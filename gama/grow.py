r"""grow — 外部錨で律する自己改善ループ (RSI) を gama 自身の config 空間に回す.

gama は「組み合わせ」を*測る*道具だった。``grow`` はその輪を閉じる: 組み合わせを**提案し**、
同じ決定的ベンチで**測り**、**held-out split が確認したときだけ**チャンピオンを差し替える。

  提案(変異) ──▶ search split で測る ──▶ 最良を挑戦者に ──▶ confirm split で裁定 ──▶ 台帳

`anchored-self-improvement` の第一原理をそのまま持ち込む: **良し悪しを LLM に判定させない**。
このモジュールのどこにも judge モデルは居ない。採点するのは ``benchmark.BenchCase.checker``
(コードを実行する / 厳密一致を見る) だけで、``grow`` はその数字にしか従わない。

### なぜ split を 3 つに割るのか (search / confirm / sealed)

1. **search** — 変異を測って挑戦者を選ぶ場。K 個の候補の *最大値* は上振れに偏る(多重比較)。
   だから search で勝ったことは「昇格の根拠」にならず、「挑戦権」にしかならない。
2. **confirm** — 挑戦者とチャンピオンだけを測り直す held-out。昇格の可否はここだけで決める。
3. **sealed** — 全世代を通して**一度も**判定に使わない封印。confirm も世代を跨いで再利用され続ける
   以上、じわじわ overfit する(判定に使った集合は、いずれ判定できなくなる)。最後に一度だけ開けて
   「偏りのない今の実力」を報告するために取っておく。

### なぜ margin(δ) を実測から取るのか

ローカル LLM は同じ config でも走らせるたびに点が動く。だから毎世代チャンピオンを confirm で
**測り直し**、前回との差 = そのセットアップ自身の揺れ幅を δ の下限にする。チャンピオン自身の
揺れより小さい「改善」は、改善ではなくノイズなので昇格させない。

### なぜ縮む変異 (simplify) を入れるのか

足す方向の変異(tool / ensemble / meshflow)しか持たないループは、足す方向にしか進めない。
非対称なループは想定外の方向(=無駄な構造が積み上がる)に必ず穴が開くので、**構造を剥がして
単体モデルに戻す変異**を対等な候補として同居させる。gama の thesis は "structure, not scale"
であって "more structure is better" ではない。

SECURITY: 測定は ``run_bench`` 経由なので、code / tool のケースは**モデル生成コードを実行する**
(gama の他のベンチと同じ opt-in の前提)。信頼できる backend にだけ回すこと。
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .benchmark import SUITES, BenchCase, run_bench, summarize
from .config import build_backend
from .models import ModelTier

SPLIT_NAMES = ("search", "confirm", "sealed")


# --------------------------------------------------------------------------- #
# Splits — 決定的に割る(乱数を使わない: 同じ suite なら誰が走らせても同じ split)
# --------------------------------------------------------------------------- #
def _allocate(n: int, ratio: tuple[int, int, int]) -> list[int]:
    """n 件を ratio 比で 3 分割した件数(最大剰余法)。

    素朴に「pattern を n 件に巡回で当てる」と、端数が必ず先頭(search)に落ちる。1 クラス 2 件の
    suite ではそれが「search 2 / confirm 0」になり、そのクラスは**永久に確認できない**。
    端数は剰余の大きい split(confirm -> sealed)から先に配るので、2 件あれば confirm に 1 件届く。
    **1 件しかないクラスは search だけに入る** —— それは分割では救えないので、``grow`` 側が
    「confirm に居ないクラスは変異させない」で受ける。
    """
    total = sum(ratio)
    quota = [n * r / total for r in ratio]
    base = [int(q) for q in quota]
    order = sorted(range(3), key=lambda i: (-(quota[i] - base[i]), i))
    for k in range(n - sum(base)):
        base[order[k % 3]] += 1
    return base


def _weave(counts: list[int]) -> list[str]:
    """件数配分を「並び順に散らした」split ラベル列にする。

    先頭から塊で配ると、case_id 順に並んだ suite (``brutal-*`` が後ろに固まる等) がそのまま
    split の難易度差になり、search と confirm が交換可能でなくなる。毎回いちばん配分に遅れて
    いる split を選ぶことで、難易度の並びを split 間で均す。
    """
    n = sum(counts)
    seq, taken = [], [0, 0, 0]
    for i in range(n):
        j = max(range(3), key=lambda k: (counts[k] * (i + 1) / n - taken[k], -k))
        seq.append(SPLIT_NAMES[j])
        taken[j] += 1
    return seq


def split_cases(cases: list[BenchCase],
                ratio: tuple[int, int, int] = (2, 1, 1)) -> dict[str, list[BenchCase]]:
    """``search`` / ``confirm`` / ``sealed`` の 3 分割を返す(交差なし・全件どれか 1 つ)。

    task_type ごとに独立に配分するので、**件数が足りるクラスはどの split にも入る**(クラスが
    片側に寄ると「search で強い = confirm で強い」の前提が壊れる)。2 件なら search/confirm、
    1 件なら search だけ —— 全クラスが 3 分割される保証ではないので、``grow`` は confirm に
    現れないクラスを変異対象から外して辻褄を合わせる。乱数は使わないので、同じ suite なら
    誰が走らせても同じ split になる。
    """
    if sum(ratio) <= 0 or any(r < 0 for r in ratio):
        raise ValueError(f"bad ratio {ratio!r}")

    by_class: dict[str, list[BenchCase]] = {}
    for c in cases:
        by_class.setdefault(c.task_type, []).append(c)

    out: dict[str, list[BenchCase]] = {n: [] for n in SPLIT_NAMES}
    for task_type in sorted(by_class):
        group = sorted(by_class[task_type], key=lambda c: c.case_id)
        for case, split in zip(group, _weave(_allocate(len(group), ratio))):
            out[split].append(case)

    # fail-closed は肯定形で: 「confirm が在ると証明できた」ときだけ先へ進む。
    # confirm が空のまま回ると、昇格判定が search の上振れをそのまま通す fail-open になる。
    if not out["confirm"]:
        raise ValueError(
            "confirm split is empty — need at least ~4 cases per class-pool to grow honestly "
            f"(got {len(cases)} cases across {len(by_class)} classes)")
    return out


def suite_pool(names) -> list[BenchCase]:
    """suite 名の並び (``["hard", "brutal"]``) を 1 本の case プールに連結する。"""
    pool: list[BenchCase] = []
    seen: set[str] = set()
    for n in names:
        if n not in SUITES:
            raise ValueError(f"unknown suite {n!r}; choose from {sorted(SUITES)}")
        for c in SUITES[n]:
            if c.case_id not in seen:       # 同じ case を二重に数えない(split の交差防止)
                seen.add(c.case_id)
                pool.append(c)
    return pool


# --------------------------------------------------------------------------- #
# Spec — チャンピオンは常に「gama router 1 段 + 名前付きレーン」の平らな形に正規化する
# --------------------------------------------------------------------------- #
def canonical(spec: dict) -> dict:
    """参照されていないレーンを落とした正規形を返す。

    正規化しないと「振る舞いは同じだが孤児レーンが残っている」spec が別ハッシュになり、
    archive の重複排除がすり抜けて同じ設計を何度も測り直す(=無駄に実モデルを焚く)。
    """
    if spec.get("backend") != "gama":
        return copy.deepcopy(spec)
    kw = spec.get("kwargs") or {}
    lanes = dict(kw.get("backends") or {})
    table = dict(kw.get("routing_table") or {})
    default = kw.get("default")
    # 既定レーンを指すルートは no-op。残すと「構造を剥がして種に戻った設計」が種と別ハッシュに
    # なり、台帳では何もしていないルートが表示され、archive でも別物として測り直される。
    table = {k: v for k, v in table.items() if v != default}
    used = set(table.values()) | ({default} if default else set())
    return {"backend": "gama",
            "kwargs": {"backends": {k: copy.deepcopy(v) for k, v in sorted(lanes.items())
                                    if k in used},
                       "routing_table": {k: table[k] for k in sorted(table)},
                       "default": default}}


def spec_hash(spec: dict) -> str:
    """正規形の内容ハッシュ(archive の同一性キー・台帳の証跡)。"""
    blob = json.dumps(canonical(spec), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def seed_champion(pool: dict[str, dict], default_lane: Optional[str] = None) -> dict:
    """出発点: ルーティング無しの単体レーン(=「構造ゼロ」のベースライン)。

    ここから構造が積まれて初めて「structure が効いた」と言えるので、種は必ず素の 1 モデル。
    """
    if not pool:
        raise ValueError("grow needs at least one lane in the pool")
    default_lane = default_lane or sorted(pool)[0]
    if default_lane not in pool:
        raise ValueError(f"default lane {default_lane!r} not in pool {sorted(pool)}")
    return canonical({"backend": "gama",
                      "kwargs": {"backends": {default_lane: copy.deepcopy(pool[default_lane])},
                                 "routing_table": {}, "default": default_lane}})


def ollama_pool(models, host: str = "http://localhost:11434") -> dict[str, dict]:
    """``ollama list`` のモデル名 -> レーン spec。``gama grow --models`` の糖衣。"""
    return {m: {"backend": "ollama",
                "kwargs": {"host": host, "model_by_tier": {"small": m, "medium": m, "large": m}}}
            for m in models}


def validate_pool(pool: dict[str, dict]) -> None:
    """レーン名が grow の合成レーン名前空間 ``tool(...)`` / ``ens(...)`` / ``mesh(...)`` と
    衝突していないことを確かめる。

    合成レーンの中身は spec の ``_grow_base`` から引くので、名前の解釈違いはもう起きない。
    ここで見るのは**上書き衝突**だけ: 同名の利用者定義レーンがあると変異がそれを差し替え、
    「宣言した spec とは別の backend を、その名前のまま測る」ことになる。名前を付け替えて
    回避するより入口で断る方が安全(付け替えは台帳のラベルと config の名前をずらす)。
    """
    bad = sorted(n for n in pool if _DERIVED_LANE.match(n))
    if bad:
        raise ValueError(
            f"lane names {bad} collide with grow's composite namespace (tool(...)/ens(...)/"
            "mesh(...)); rename them in the pool")


_DERIVED_LANE = re.compile(r"^(tool|ens|mesh)\(.*\)$")


def _lane_for(champion: dict, task_type: str) -> str:
    kw = champion["kwargs"]
    return kw["routing_table"].get(task_type, kw["default"])


def _atomic_lane(champion: dict, lane: str) -> Optional[str]:
    """合成レーンが包んでいる素のレーン名。素なら None。

    **名前からは読まない**。合成レーンを作るときに spec へ書いた ``_grow_base`` を引くだけ。
    以前は ``tool(a)`` / ``ens(a+b)`` / ``mesh(a->b)`` を逆パースしていたが、pool の lane 名は
    利用者の JSON 由来で任意なので、括弧・``+``・``->`` の扱いを直すたび別の取りこぼしが出た
    (同じ場所で三度パッチを当てた時点で、パッチでなく設計の問題)。名前は人が読むラベルに徹し、
    構造は spec 自身に持たせる。``build_backend`` は未知の最上位キーを見ないので構築には影響しない。
    """
    spec = (champion.get("kwargs", {}).get("backends") or {}).get(lane) or {}
    base = spec.get("_grow_base")
    return base if isinstance(base, str) else None


_DERIVED_LANE = re.compile(r"^(tool|ens|mesh)\(.*\)$")


def _lane_for(champion: dict, task_type: str) -> str:
    kw = champion["kwargs"]
    return kw["routing_table"].get(task_type, kw["default"])


def _atomic_lane(champion: dict, lane: str) -> Optional[str]:
    """合成レーンが包んでいる素のレーン名(``tool(qwen)`` -> ``qwen``)。素なら None。

    区切りは合成の種類ごとに違う(``ens(a+b)`` / ``mesh(a->b)``)。片方だけ剥がすと、
    剥がせない合成レーンが**縮む変異の効かない袋小路**になる(足す方向にしか動けなくなる)。

    判定は ``validate_pool`` と**同じ述語**(``_DERIVED_LANE``)で行う。「入口で弾く名前」と
    「合成として分解する名前」がずれると、弾かれなかった利用者のレーン名(``foo(a)`` 等)が
    分解され、``simplify:qa->a`` という嘘のラベルで別 backend へ振り替わる。
    """
    if not _DERIVED_LANE.match(lane):
        return None
    inner = lane[lane.index("(") + 1:-1]
    return inner.replace("->", "+").split("+")[0] if inner else None


# --------------------------------------------------------------------------- #
# Mutations — 決定的な候補生成器(LLM に「次に何を試すか」を考えさせない)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Candidate:
    label: str          # 例 "route:qa->qwen2.5:7b" — 台帳を人が読める単位にする
    kind: str           # route | tool | ensemble | meshflow | simplify
    spec: dict = field(compare=False)


def _with_lane(champion: dict, task_type: str, lane_name: str, lane_spec: dict) -> dict:
    """レーンを 1 本足して task_type をそこへ向けた新しい champion spec。"""
    new = copy.deepcopy(champion)
    new["kwargs"]["backends"][lane_name] = lane_spec
    new["kwargs"]["routing_table"][task_type] = lane_name
    return canonical(new)


def _by_class_rotation(items: list, classes: list[str], offset: int) -> list:
    """``[(task_type, Candidate)]`` を「クラスを 1 周ずつ」巡る順に並べ替える。

    ``offset`` は開始クラスをずらす量。種類ごとに違う offset を与えることで、width を小さく
    しても *種類* と *クラス* の両方に候補が散る(同じクラスに 4 種類まとめて刺さらない)。
    """
    by_cls: dict[str, list] = {}
    for task_type, cand in items:
        by_cls.setdefault(task_type, []).append(cand)
    rotated = [classes[(offset + i) % len(classes)] for i in range(len(classes))] if classes else []
    out, depth = [], 0
    while True:
        row = [by_cls[c][depth] for c in rotated if c in by_cls and depth < len(by_cls[c])]
        if not row:
            return out
        out += row
        depth += 1


def propose(champion: dict, pool: dict[str, dict], classes: list[str],
            width: int = 6, exclude: Optional[set] = None,
            ensemble_strategy: str = "synthesize",
            generation: int = 0) -> list[Candidate]:
    """チャンピオンから 1 手だけ動かした候補を、種類を混ぜて ``width`` 本返す。

    1 手だけなのは、勝因を測定に帰属させるため(2 手同時だとどちらが効いたか台帳から読めない)。
    種類を round-robin で混ぜるのは、``width`` を絞ったときに ``route`` 変異だけで埋まって
    構造変異が一生試されない偏りを防ぐため。

    ``exclude`` は「confirm で一度挑戦して負けた設計」のハッシュ集合。**search で測っただけの
    設計は除外しない** — search で選ばれなかっただけの候補を永久追放すると、後で本命になりうる
    踏み石(archive)を毎回捨てることになる。除外するのは決着がついた設計だけ。

    ``generation`` は世代番号で、**クラスと種類の両方の巡回開始位置**をずらす。固定だと
    2 種類の飢餓が起きる(どちらも実測):
      * クラス側 —— 挑戦者に選ばれなかった候補が種類キューの先頭に居座り続け、同じ種類の別
        クラスに永久に順番が来ない(全 suite 走で `tool:integration` が居座り、`tool:qa` は
        3 世代とも一度も提案されなかった。落ちたのではなく試されていない)。
      * 種類側 —— 種類の巡回が毎回 order の先頭から始まるので、``width`` が種類数より小さいと
        末尾の種類が落ちる。しかも `simplify` は**昇格が起きた直後に空でなくなる**ため、
        「1 つ昇格した瞬間に meshflow が候補から消える」という当たりの悪い噛み合わせになる
        (同じ走の gen1 で実際に消えた)。
    """
    validate_pool(pool)
    exclude = exclude or set()
    lanes = sorted(pool)
    ordered_classes = sorted(classes)
    # 種類ごとに (task_type, Candidate) で貯める。task_type を保つのは、最後に**クラスも跨いで**
    # 巡回させるため — クラス順に詰めたまま width で切ると、辞書順で先頭のクラスだけが延々
    # 探索され、残りのクラスは一度も触られない(小さい width ほど効く偏り)。
    buckets: dict[str, list] = {k: [] for k in
                                ("route", "tool", "ensemble", "meshflow", "simplify",
                                 "default", "deepen")}

    for task_type in ordered_classes:
        cur = _lane_for(champion, task_type)
        base = _atomic_lane(champion, cur) or cur          # 合成レーンなら中身の素モデルを基準に
        for lane in lanes:                                  # ① 別モデルへ振り替える
            if lane != cur:
                buckets["route"].append((task_type, Candidate(
                    f"route:{task_type}->{lane}", "route",
                    _with_lane(champion, task_type, lane, copy.deepcopy(pool[lane])))))
        if base in pool and not cur.startswith("tool("):     # ② PAL(コードを書かせて実行)で包む
            name = f"tool({base})"
            buckets["tool"].append((task_type, Candidate(
                f"tool:{task_type}({base})", "tool",
                _with_lane(champion, task_type, name,
                           {"backend": "tool", "_grow_base": base,
                            "kwargs": {"inner": copy.deepcopy(pool[base])}}))))
        # ③ 2 モデルの合議。既定は synthesize —— majority は**自由文では機能しない**。逐語一致が
        # まず起きないので Counter が全部 1 になり、most_common が「最初に入れたメンバー」を返す。
        # 実測(graded 20 問): majority 0.705 に対し素の 3B 単体が 0.830、synthesize は 0.975
        # (case 単位で 8 勝 0 敗)。既定のままだとこの変異は「一番安いモデルの答えを、全員ぶんの
        # 金を払って採用する」になっていた。
        for other in lanes:
            if other == base or base not in pool:
                continue
            name = f"ens({base}+{other})"
            if name == cur:
                continue
            buckets["ensemble"].append((task_type, Candidate(
                f"ensemble:{task_type}({base}+{other})", "ensemble",
                _with_lane(champion, task_type, name,
                           {"backend": "ensemble", "_grow_base": base,
                            "kwargs": dict(
                                {"members": [copy.deepcopy(pool[base]),
                                             copy.deepcopy(pool[other])],
                                 "strategy": ensemble_strategy},
                                # 統合役は「もう一方」= base でない側。既定の aggregator は
                                # members[0](=base) なので、弱い方に最終回答を書かせてしまう。
                                # synthesize 以外では使われないキーなので入れない —— 入れると
                                # build_backend が使わない backend を先に構築してしまう。
                                **({"aggregator": copy.deepcopy(pool[other])}
                                   if ensemble_strategy == "synthesize" else {}))}))))
        for other in lanes:                                 # ④ 検証で gate した段階委譲
            if other == base or base not in pool:
                continue
            name = f"mesh({base}->{other})"
            if name == cur:
                continue
            # verify は渡さない: bench が case の checker を kwargs で差し込むので、
            # エスカレーションは採点と同一の外部検証で gate される(自己申告での昇格を作らない)。
            buckets["meshflow"].append((task_type, Candidate(
                f"meshflow:{task_type}({base}->{other})", "meshflow",
                _with_lane(champion, task_type, name,
                           {"backend": "meshflow", "_grow_base": base,
                            "kwargs": {"tiers": [copy.deepcopy(pool[base]),
                                                 copy.deepcopy(pool[other])],
                                       "mesh": "union"}}))))
        if _atomic_lane(champion, cur):                     # ⑤ 構造を剥がして素に戻す
            inner = _atomic_lane(champion, cur)
            if inner in pool:
                buckets["simplify"].append((task_type, Candidate(
                    f"simplify:{task_type}->{inner}", "simplify",
                    _with_lane(champion, task_type, inner, copy.deepcopy(pool[inner])))))

    # ⑥ 既定レーンそのものを差し替える。ここまでの変異は**クラス単位の振り替えしかできず**、
    #    「ルーティングしていない全クラスが乗っている既定レーン」を一度も動かせなかった
    #    (5 クラス中 3 クラスが既定に乗っている構成でも手が無い)。1 手で config の 1 箇所しか
    #    変えない点は他の変異と同じ。
    cur_default = champion["kwargs"]["default"]
    for lane in lanes:
        if lane != cur_default:
            new_champ = copy.deepcopy(champion)
            new_champ["kwargs"]["backends"][lane] = copy.deepcopy(pool[lane])
            new_champ["kwargs"]["default"] = lane
            buckets["default"].append((ordered_classes[0], Candidate(
                f"default->{lane}", "default", canonical(new_champ))))

    # ⑦ 合成レーンの中身をもう一段包む。既存の変異は素のレーンからしか合成を作れないので、
    #    `mesh(tool(3b) -> coder7b)` のような入れ子は**構造上到達できない領域**だった。
    for task_type in ordered_classes:
        cur = _lane_for(champion, task_type)
        base = _atomic_lane(champion, cur)
        if not base or base not in pool or cur.startswith("tool("):
            continue
        spec = copy.deepcopy(champion["kwargs"]["backends"][cur])
        wrapped = {"backend": "tool", "kwargs": {"inner": copy.deepcopy(pool[base])}}
        kw = spec.get("kwargs") or {}
        if "tiers" in kw and kw["tiers"]:
            kw["tiers"][0] = wrapped
        elif "members" in kw and kw["members"]:
            kw["members"][0] = wrapped
        else:
            continue
        name = f"{cur}+tool"
        buckets["deepen"].append((task_type, Candidate(
            f"deepen:{task_type}(tool inside {cur.split('(')[0]})", "deepen",
            _with_lane(champion, task_type, name, spec))))

    order = ["simplify", "route", "tool", "ensemble", "meshflow", "default", "deepen"]
    queues = {k: _by_class_rotation(buckets[k], ordered_classes, offset + generation)
              for offset, k in enumerate(order)}

    champ_hash = spec_hash(champion)
    out: list[Candidate] = []
    emitted: set = set()
    i = 0
    while len(out) < width and any(queues[k] for k in order):
        kind = order[(i + generation) % len(order)]
        i += 1
        # width が数えるのは「出した候補」であって「試した回数」ではない。除外(決着済み・重複)を
        # 引いた時に枠を消費して次の種類へ進むと、**その除外の後ろに並んでいる候補が永久に出番を
        # 失う**: 実測(run J)では gen0 で決着した `simplify:qa` がキュー先頭に残り続け、
        # `simplify:research` が 3 世代とも一度も提案されなかった —— チャンピオンが research
        # レーンを持ち続けている理由を、ループは一度も問われないまま終わった。
        while queues[kind]:
            cand = queues[kind].pop(0)
            h = spec_hash(cand.spec)
            if h == champ_hash or h in exclude or h in emitted:
                continue                      # 除外は席を消費しない: 同じ種類から次を引く
            emitted.add(h)
            out.append(cand)
            break
    return out


# --------------------------------------------------------------------------- #
# Measurement — 決定的チェッカだけを信じる
# --------------------------------------------------------------------------- #
@dataclass
class Measurement:
    score: float
    success_rate: float
    latency_s: float
    n: int
    cases: int


def measure(spec: dict, cases: list[BenchCase], tier: ModelTier = ModelTier.LARGE,
            repeats: int = 1, unit_cost: Optional[dict] = None,
            label: str = "candidate") -> Measurement:
    """1 つの spec を case 集合で測る。例外は ``run_bench`` 側で 0 点に落ちる(掃引を止めない)。"""
    if not cases:
        raise ValueError("measure() needs at least one case (an empty split has no score, "
                         "and returning 0.0 would read as 'measured and failed')")
    backend = build_backend(spec)
    records = run_bench({label: backend}, suite=cases, tier=tier, repeats=repeats,
                        unit_cost=unit_cost or {}, run_id="grow")
    agg = summarize(records)["overall"][label]
    return Measurement(score=agg["score"], success_rate=agg["success_rate"],
                       latency_s=agg["latency_s"], n=agg["n"], cases=len(cases))


# --------------------------------------------------------------------------- #
# The gate — 昇格を決める唯一の関数(肯定形: 「昇格してよいと証明できた」ときだけ True)
# --------------------------------------------------------------------------- #
def promote_gate(champion_search: float, challenger_search: float,
                 champion_confirm: float, challenger_confirm: float,
                 delta: float) -> tuple[bool, str]:
    """3 条件が**すべて**証明できたときだけ昇格。理由は台帳に残せる文字列で返す。

    ① 挑戦権: search で本当に上回ったか(上回っていない候補は挑戦者になれない)
    ② held-out: confirm でも上回ったか(search の上振れは confirm を通らない)
    ③ 幅超え: 差が δ 以上か。δ = max(confirm 1 問ぶん, チャンピオン自身の測り直しの揺れ)。
       1 問未満の差は部分点の揺らぎでしかなく、揺れの内側の差は改善ではない。
    """
    if not challenger_search > champion_search:
        return False, "search-not-better"
    if not challenger_confirm > champion_confirm:
        return False, "confirm-not-better"
    # 「1 問ぶんちょうど」は通す約束なので、比較には浮動小数の遊びを持たせる。scores は
    # summarize が 4 桁に丸めた値、delta は 1/n の生の値なので、素の >= だと 0.0666 >= 0.066667
    # が偽になり、**丸め方次第で同じ 1 問ぶんの改善が通ったり落ちたりする**。1e-9 は 4 桁丸めの
    # 世界では絶対に効いてこない幅。
    if (challenger_confirm - champion_confirm) < delta - 1e-9:
        return False, f"below-margin(delta={round(delta, 4)})"
    return True, "promote"


_COMPOSITES = ("tool", "ensemble", "meshflow", "trinity", "abmcts")


def code_stamp() -> dict:
    """どのコードがこの数字を出したかを台帳に残すための刻印。

    この loop は稼働しながら**判定そのもの**を何度も変えてきた(探索の巡回・ensemble の集約・
    削減の門・その許容幅)。台帳に走行条件(suites/ratio/repeats)しか無いと、同じ条件で取った
    はずの数字が実は別の門で出ていた、という比較を後から検出できない。checkout から動いている
    ときは git の短縮 SHA を、そうでなければ version だけを残す(取れないこと自体は異常でない)。
    """
    import subprocess

    sha = None
    try:
        proc = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent),
                               "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5)
        if proc.returncode == 0:
            sha = proc.stdout.strip() or None
    except Exception:      # git が無い / インストール済みパッケージ / 何であれ致命ではない
        sha = None
    from . import __version__

    return {"version": __version__, "commit": sha}


def _structure_size(spec: dict) -> int:
    """How much structure the champion carries, counted **per routed class**.

    Counting composite *lanes* instead would miss the common case: two classes sharing one
    `tool(...)` lane, where returning one of them to the bare model leaves the lane standing —
    a real reduction that would score "not simpler" and never be allowed through.
    """
    kw = spec.get("kwargs") or {}
    lanes = kw.get("backends") or {}
    composite = {n for n, v in lanes.items() if v.get("backend") in _COMPOSITES}
    routed = sum(1 for v in (kw.get("routing_table") or {}).values() if v in composite)
    return routed + (1 if kw.get("default") in composite else 0)


def shrink_band(margin_floor: float, drift: float) -> float:
    """削減を許す幅。**追加の δ を流用してはいけない。**

    追加側の ``delta = max(床, drift)`` は「ノイズが大きい日は昇格しにくい」= 保守側に働く。
    同じ式を削減に使うと**逆向き**に効く: ノイズが大きい日ほど「区別できない」幅が広がり、
    構造を捨てやすくなる。実測(run I gen0)では drift 0.114 の世代に、confirm を 0.065
    (1.5 問ぶん)下げる削減が「測って悪いとは言えない」で通り、**7/8 走で昇格し per-case で
    機構まで確認済みの `tool:qa` が、1 世代のノイズを根拠に剥がされた**。

    削減が許されるのは、低下が**分解能とノイズの両方の下**にあるときだけ: ``min(床, drift)``。
    ``drift`` は測り直しの差の絶対値なので常に非負、したがって band も非負。
    drift が 0 と出た世代は「低下が無いときだけ剥がせる」になる。厳しいが、根拠を持って
    入れたものを外す側の基準としてはその厳しさが正しい。
    """
    return min(margin_floor, drift)


def simplify_gate(champion_confirm: float, challenger_confirm: float, band: float,
                  champion_structure: int, challenger_structure: int) -> tuple[bool, str]:
    """削減の門。追加とは**違う問い**で裁く。

    追加は「測って良くなったか」だが、削減は「測って悪くなっていないか」。同じ門(厳密改善)で
    裁くと、削減は原理的にほぼ通らない —— 実測でも 6 走で simplify 候補は挑戦者にすらなれず
    (提案 4 回・挑戦 0 回)、その間チャンピオンのコストは 0.61s→1.73s/問 と単調に増えた。
    「足す方向にしか進めないループを作らない」と宣言しておきながら、非対称は変異の層ではなく
    門の層に残っていた。

    条件: ①構造が厳密に減る ②confirm の低下が ``band``(= ``shrink_band``) **以下**
    (追加側が「δ ちょうどで昇格」なのと対称。``band`` が 0 の世代は「低下ゼロなら可」)。
    差が測れないなら簡単な方を採る、という Occam の適用であって、「同点だから通す」ではない。
    """
    if not challenger_structure < champion_structure:
        return False, "not-simpler"
    # band ちょうどまでは通す(追加側が「δ ちょうどで昇格」なのと対称)。band が 0 の世代は
    # 「低下がゼロなら剥がしてよい」になり、コストだけが減る削除は最後まで許される。
    if (champion_confirm - challenger_confirm) > band + 1e-9:
        return False, f"measurably-worse(band={round(band, 4)})"
    return True, "simplify"


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
def load_checkpoint(ledger_path) -> Optional[dict]:
    """台帳の最後の checkpoint を返す(無ければ None)。

    落ち方はこちらの都合を聞かないので、書きかけで切れた JSON 行が末尾に残ることがある。
    壊れた行は黙って飛ばす —— そこで例外を投げると、再開できるはずの走行が再開できなくなる。
    """
    last = None
    try:
        lines = Path(ledger_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("event") == "checkpoint":
            last = row
    return last


def _ledger_splits(ledger_path) -> Optional[dict]:
    """再開先の台帳が使っていた split(case id)。分割が違えば再開してはいけない。"""
    try:
        for line in Path(ledger_path).read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("event") == "seed":
                return row.get("splits")
    except OSError:
        return None
    return None


def grow(pool: dict[str, dict], *, classes: Optional[list[str]] = None,
         cases: Optional[list[BenchCase]] = None, suites=("wide", "hard", "brutal"),
         ratio: tuple[int, int, int] = (2, 1, 1), generations: int = 3, width: int = 6,
         repeats: int = 1, tier: ModelTier = ModelTier.LARGE,
         min_margin: Optional[float] = None, resume_from=None,
         patience: int = 2, seed_spec: Optional[dict] = None,
         ledger_path: Optional[str] = None, unit_cost: Optional[dict] = None,
         ensemble_strategy: str = "synthesize",
         on_event: Optional[Callable[[dict], None]] = None) -> dict:
    """RSI を ``generations`` 世代回し、最終チャンピオンと全世代の台帳を返す。

    ``patience`` 世代続けて昇格が出なければ収束とみなして打ち切る(空回りで実モデルを焚かない)。
    ``sealed`` split は世代中一切参照せず、最後に**一度だけ**開けて種と最終チャンピオンを測る。

    ``min_margin`` を省くと **confirm 1 問ぶん** (``1/len(confirm)``) が下限になる。
    """
    validate_pool(pool)
    pool_cases = cases if cases is not None else suite_pool(suites)
    splits = split_cases(pool_cases, ratio=ratio)
    # 変異させるクラスは「confirm にも case が在るクラス」だけに絞る。confirm に無いクラスを
    # いじった候補は search でしか動かず、昇格ゲートを構造的に通れない(= 実モデルを焚くだけ
    # 無駄になる)。肯定形の絞り込み: 確認できると証明できたクラスだけを触る。
    searchable = {c.task_type for c in splits["search"]}
    confirmable = {c.task_type for c in splits["confirm"]}
    asked = set(classes) if classes else searchable
    dropped = sorted(asked - confirmable)
    # 呼び出し側がクラスを明示しても絞り込みは外さない。ここを bypass できると、
    # 「confirm に居ないクラスを変異させる」= 昇格し得ない候補に実モデルを焚く経路が
    # Python API 側にだけ開く(CLI で閉じた穴が API で開く、が典型的な穴の空き方)。
    classes = sorted(asked & confirmable)
    if not classes:
        raise ValueError("no task class appears in BOTH the search and confirm splits — "
                         "widen the case pool (e.g. --suites wide,hard,brutal)")

    # 昇格の下限は「confirm 1 問ぶん」を既定にする(ちょうど 1 問ぶんは通る = 以上判定)。
    # 部分点を返す checker があるので刻みが常に 1/n という訳ではない。狙いはそこではなく、
    # **1 問を丸ごと動かせていない改善は採らない**という線引きの方。ここを定数(旧既定 0.05)
    # にすると、confirm 8 問なら半問で通り、30 問なら丸 1 問勝っても落ちる —— **suite を
    # 差し替えた瞬間に門の意味が黙って変わる**ので、下限は case 数から導く。実測 drift の
    # 方が大きい日はそちらが勝つ(max)。
    if min_margin is not None and min_margin < 0:
        raise ValueError(f"min_margin must be >= 0 (got {min_margin}); a negative floor "
                         "disables the gate instead of loosening it")
    one_case = 1.0 / len(splits["confirm"])
    margin_floor = one_case if min_margin is None else min_margin
    # confirm が小さいと「1 問ぶん」がそのまま高い壁になる。実測: confirm 5 問(=1 問 0.2)で
    # 他の 4 走すべてが昇格させた本物の改善が +0.075 にしか見えず、床に弾かれた。床は宣言
    # どおり「1 問未満は証明できない」と言っただけで、悪いのは split の細さの方。ただしそれが
    # **走り終わってから**分かるのでは遅いので、入口で粗さを申告する。
    coarse = one_case > 0.1

    resume = load_checkpoint(resume_from) if resume_from else None
    if resume_from:
        before = _ledger_splits(resume_from)
        now = {k: [c.case_id for c in v] for k, v in splits.items()}
        # 分割が違う台帳から再開したら、前半で sealed だった case が後半では search に居る、
        # という混線が起きる。封印は「一度も判定に使っていない」が成り立って初めて意味を持つので、
        # ここは断る一択。
        if before is not None and before != now:
            raise ValueError("cannot resume: the ledger was written with a different split "
                             "(different --suites/--ratio); its sealed cases are not sealed "
                             "under this one")
        if resume is None:
            raise ValueError(f"no checkpoint in {resume_from} — nothing to resume from")

    champion = canonical(resume["champion"] if resume else (seed_spec or seed_champion(pool)))
    # 参照先の無いルートを持つ seed は、GamaBackend が既定レーンへ黙って落とすので
    # 「台帳に記録した spec」と「実際に測った backend」がずれる。記録が実態と一致することが
    # この loop の全部なので、入口で断る。
    _lanes = champion["kwargs"]["backends"]
    _missing = sorted({v for v in champion["kwargs"]["routing_table"].values() if v not in _lanes}
                      | ({champion["kwargs"]["default"]} - set(_lanes)))
    if _missing:
        raise ValueError(f"seed spec routes to lanes that do not exist: {_missing}")
    seed = copy.deepcopy(champion)
    archive: dict[str, dict] = {}
    history: list[dict] = []
    ledger = Path(ledger_path) if ledger_path else None
    if ledger:
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text("", encoding="utf-8")

    def emit(row: dict) -> None:
        if ledger:
            with ledger.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        if on_event:
            on_event(row)

    if resume:
        champ_search = Measurement(**resume["champion_search"])
        champ_confirm = Measurement(**resume["champion_confirm"])
        archive.update(resume.get("archive") or {})
        start_gen = resume["gen"] + 1
    else:
        champ_search = measure(champion, splits["search"], tier, repeats, unit_cost, "champion")
        champ_confirm = measure(champion, splits["confirm"], tier, repeats, unit_cost, "champion")
        start_gen = 0
    emit({"event": "resumed" if resume else "seed", "hash": spec_hash(champion),
          "resumed_from_gen": resume["gen"] if resume else None,
          "search": asdict(champ_search), "confirm": asdict(champ_confirm),
          "classes": classes, "classes_unconfirmable": dropped,
          "code": code_stamp(),
          "margin_floor": round(margin_floor, 4),
          "margin_floor_source": "auto(one confirm case)" if min_margin is None else "explicit",
          "margin_floor_coarse": coarse,
          "splits": {k: [c.case_id for c in v] for k, v in splits.items()}})

    # 種の測定が終わった時点でも checkpoint を打つ。世代ごとの checkpoint だけだと、
    # **最初の世代が終わるまでの区間が丸ごと空白**になる: 実測でその区間(search 40 + confirm 20
    # を repeats 2 = 120 コール、gemma4 で約 20 分)にセッションごと落ちて、台帳 0 行で全損した。
    # resume の守備範囲は、いちばん長い空白から埋める。
    if not resume:
        emit({"event": "checkpoint", "gen": -1, "champion": champion,
              "champion_search": asdict(champ_search), "champion_confirm": asdict(champ_confirm),
              "challenged": [], "archive": {}, "stale": 0})

    stale = resume["stale"] if resume else 0
    # confirm で決着がついた設計(勝っても負けても二度は問わない)
    challenged: set = set(resume["challenged"]) if resume else set()
    for gen in range(start_gen, generations):
        t0 = time.time()
        cands = propose(champion, pool, classes, width=width, exclude=challenged,
                        ensemble_strategy=ensemble_strategy, generation=gen)
        if not cands:
            emit({"event": "stop", "gen": gen, "reason": "no-new-candidates"})
            break

        scored = []
        for c in cands:
            h = spec_hash(c.spec)
            cached = archive.get(h)
            # 同じ設計を二度測らない(実モデルを無駄に焚かない)。search 側の測定揺れはここで
            # 固定されるが、上振れした候補も confirm で必ず測り直されるので昇格判定は汚れない
            # (代償は「上振れ candidate に挑戦権を 1 回使う」ことだけ)。揺れ自体を抑えたい場合は
            # repeats を上げる。
            if cached:
                m = Measurement(**cached["search"])
            else:
                m = measure(c.spec, splits["search"], tier, repeats, unit_cost, "candidate")
                archive[h] = {"label": c.label, "kind": c.kind, "search": asdict(m)}
                emit({"event": "candidate", "gen": gen, "label": c.label, "kind": c.kind,
                      "hash": h, "search": asdict(m)})
            scored.append((c, m))

        # チャンピオンを毎世代 confirm で測り直す。差 δ の下限は「同じ config を測り直したときの
        # 揺れ」そのもの — 自分の揺れより小さい改善を採らないための実測アンカー。
        champ_confirm_now = measure(champion, splits["confirm"], tier, repeats, unit_cost,
                                    "champion")
        drift = abs(champ_confirm_now.score - champ_confirm.score)
        delta = max(margin_floor, drift)

        # 追加と削減は別々に選ぶ。混ぜると、構造を剥がした候補が search 最高点の勝負に放り込まれ、
        # 「良くなったか」という**別の問い**の門で焼かれる(そして二度と挑戦できない)。
        additive = [(c, m) for c, m in scored if c.kind != "simplify"]
        row = {"event": "generation", "gen": gen, "champion_hash": spec_hash(champion),
               "champion_search": champ_search.score,
               "champion_confirm": champ_confirm_now.score,
               "drift": round(drift, 4), "delta": round(delta, 4),
               "candidates": len(cands)}
        ok, reason = False, "no-additive-candidate"
        if additive:
            # 同点はコスト(=実測レイテンシ)の低い方、それも同じならラベル順。決定的に選ぶ。
            challenger, chal_search = min(additive, key=lambda t: (-t[1].score, t[1].latency_s,
                                                                    t[0].label))
            chal_confirm = measure(challenger.spec, splits["confirm"], tier, repeats, unit_cost,
                                   "challenger")
            ok, reason = promote_gate(champ_search.score, chal_search.score,
                                      champ_confirm_now.score, chal_confirm.score, delta)
            challenged.add(spec_hash(challenger.spec))
            row.update({"challenger": challenger.label, "kind": challenger.kind,
                        "challenger_hash": spec_hash(challenger.spec),
                        "challenger_search": chal_search.score,
                        "challenger_confirm": chal_confirm.score})
        row.update({"verdict": "promote" if ok else "reject", "reason": reason})
        if ok:
            champion, champ_search, champ_confirm = challenger.spec, chal_search, chal_confirm
            stale = 0
        else:
            champ_confirm = champ_confirm_now       # 次世代の drift 基準は常に最新の実測
            stale += 1
        # 追加が却下された世代だけ、削減にも 1 枠まわす。追加が通った世代は見送る:
        # 1 世代 1 手を守らないと、どちらが効いたのか台帳から帰属できなくなる。
        if not ok:
            shrinks = [(c, m) for c, m in scored
                       if c.kind == "simplify"
                       and _structure_size(c.spec) < _structure_size(champion)
                       and spec_hash(c.spec) not in challenged]
            if shrinks:
                cand, cand_search = min(shrinks, key=lambda t: (-t[1].score, t[0].label))
                cand_confirm = measure(cand.spec, splits["confirm"], tier, repeats, unit_cost,
                                       "simplifier")
                band = shrink_band(margin_floor, drift)
                s_ok, s_reason = simplify_gate(champ_confirm_now.score, cand_confirm.score,
                                               band, _structure_size(champion),
                                               _structure_size(cand.spec))
                challenged.add(spec_hash(cand.spec))
                row["simplify_challenger"] = cand.label
                row["simplify_confirm"] = cand_confirm.score
                row["simplify_verdict"] = "promote" if s_ok else "reject"
                row["simplify_reason"] = s_reason
                row["simplify_band"] = round(band, 4)
                if s_ok:
                    champion, champ_search, champ_confirm = cand.spec, cand_search, cand_confirm
                    stale = 0
        row["champion_after"] = spec_hash(champion)
        row["structure_size"] = _structure_size(champion)
        row["elapsed_s"] = round(time.time() - t0, 2)     # 削減の測定まで含めた世代の実時間
        history.append(row)
        emit(row)
        # 世代ごとの checkpoint。実走は数時間かかり、実際に OOM で 43 問 x 4 候補を測り終えた
        # 直後に落ちて全部消えた。台帳に判定は残っていたのに再開できなかったのは、**再開に要る
        # 状態(チャンピオンの spec・決着済み・archive)を残していなかった**ため。
        emit({"event": "checkpoint", "gen": gen, "champion": champion,
              "champion_search": asdict(champ_search), "champion_confirm": asdict(champ_confirm),
              "challenged": sorted(challenged), "archive": archive, "stale": stale})
        if stale >= patience:
            emit({"event": "stop", "gen": gen, "reason": f"no-promotion-for-{patience}-gens"})
            break

    # 封印を開けるのはここだけ。判定には一切使っていないので、この数字だけが偏っていない。
    # case が足りず封印を作れなかった場合は None を返す —— 判定に使った数字を「偏っていない
    # 数字」の欄に流用するのが、この設計でいちばんやってはいけないこと。
    if splits["sealed"]:
        sealed_seed = measure(seed, splits["sealed"], tier, repeats, unit_cost, "seed")
        sealed_champ = (sealed_seed if spec_hash(seed) == spec_hash(champion) else
                        measure(champion, splits["sealed"], tier, repeats, unit_cost, "champion"))
        sealed = {"seed": asdict(sealed_seed), "champion": asdict(sealed_champ)}
    else:
        sealed = None
    result = {
        "champion": champion, "champion_hash": spec_hash(champion),
        "seed": seed, "seed_hash": spec_hash(seed),
        # 「チャンピオンが実際に入れ替わった世代」を数える。片方の門の verdict を数えると、
        # 削減で入れ替わった世代が reject 扱いのまま promotions に乗らず、**台帳が実態と食い違う**
        # (しかも門を足すたび同じ穴が空く)。
        "promotions": sum(1 for h in history if h["champion_hash"] != h["champion_after"]),
        # 昇格の回数と、走行の**正味の成果**は別物。実測(run L)では ensemble を足した次の世代に
        # 同じレーンを外し、"promotions: 2" のままチャンピオンが種と同一に戻った。回数だけを
        # 読むと「2 つ良くなった」と誤読する。
        "net_change": spec_hash(champion) != spec_hash(seed),
        "generations_run": len(history),
        "search": asdict(champ_search), "confirm": asdict(champ_confirm),
        "sealed": sealed,
        "splits": {k: [c.case_id for c in v] for k, v in splits.items()},
        "params": {"suites": list(suites) if cases is None else "custom", "ratio": list(ratio),
                   "tier": tier.value, "repeats": repeats, "width": width,
                   "generations": generations, "patience": patience,
                   "min_margin": round(margin_floor, 4),
                   "min_margin_source": "auto(one confirm case)" if min_margin is None
                                        else "explicit",
                   "ensemble_strategy": ensemble_strategy, "code": code_stamp()},
        "history": history, "archive_size": len(archive),
    }
    emit({"event": "final", **{k: v for k, v in result.items() if k != "history"}})
    return result


# --------------------------------------------------------------------------- #
# Recipe emission — 育った成果を gama の recipe library に還す
# --------------------------------------------------------------------------- #
def write_recipe(result: dict, directory, name: Optional[str] = None,
                 hardware: str = "(fill in: box, RAM, GPU)") -> Path:
    """勝ったチャンピオンを ``config.json`` + ``recipe.md`` として書き出す。

    数字は必ず sealed(判定に使っていない封印)を主役に書く。search/confirm の点は選抜に使った
    以上どうしても上振れするので、そこを見出しに置くと「育った」ではなく「盛れた」になる。
    """
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    name = name or d.name
    config = {"_comment": f"GROWN by `gama grow` — champion {result['champion_hash']}. "
                          "Numbers below are from a sealed split never used for any decision.",
              "system": result["champion"],
              "grow": {"seed_hash": result["seed_hash"], "promotions": result["promotions"],
                       "generations_run": result["generations_run"],
                       # the conditions travel with the numbers: a sealed score means nothing
                       # without the suites/ratio/repeats it was produced under
                       "params": result.get("params", {}),
                       "splits": result["splits"], "sealed": result["sealed"]}}
    (d / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
    sealed = result["sealed"]
    lines = [
        f"# {name} (grown by `gama grow`)", "",
        f"Hardware: {hardware}", "",
        "| | seed (no structure) | grown champion |",
        "|---|---|---|",
    ]
    if sealed:
        lines.append(
            f"| sealed score (n={sealed['champion']['cases']} cases, never used for a decision) "
            f"| **{sealed['seed']['score']}** | **{sealed['champion']['score']}** |")
    else:
        lines.append("| sealed score | _no sealed split — every number below fed a decision, "
                     "so read them as optimistic_ | |")
    lines += [
        f"| confirm score (the split that decided promotions) | — | {result['confirm']['score']} |",
        f"| search score (selection — biased upward, do not quote) | — | "
        f"{result['search']['score']} |", "",
        f"- promotions: {result['promotions']} over {result['generations_run']} generations "
        f"({result['archive_size']} designs measured)",
        "- every promotion required a held-out `confirm` win larger than the champion's own "
        "re-measurement drift; no LLM judged anything.",
        f"- grown with: {json.dumps(result.get('params', {}), ensure_ascii=False)}",
        "- spot-check the champion (this is NOT a reproduction of the numbers above, which "
        "come from the splits recorded in config.json): "
        "`gama bench --backends system --config config.json --suite hard`", "",
        "## What grew", "",
    ]
    for h in result.get("history", []):     # a result round-tripped through JSON may omit it
        mark = "✅" if h["verdict"] == "promote" else "·"
        lines.append(f"- {mark} gen{h['gen']} `{h['challenger']}` — search "
                     f"{h['champion_search']}→{h['challenger_search']}, confirm "
                     f"{h['champion_confirm']}→{h['challenger_confirm']} "
                     f"(δ={h['delta']}) → {h['reason']}")
    (d / "recipe.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return d
