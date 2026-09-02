r"""grow — 外部錨で律する自己改善ループ (RSI) を gama 自身の config 空間に回す.

gama は「組み合わせ」を*測る*道具だった。``grow`` はその輪を閉じる: 組み合わせを**提案し**、
同じ決定的ベンチで**測り**、**held-out split が確認したときだけ**チャンピオンを差し替える。

  提案(変異) ──▶ search split で測る ──▶ 最良を挑戦者に ──▶ confirm split で裁定 ──▶ 台帳

`anchored-self-improvement` の第一原理をそのまま持ち込む: **良し悪しを LLM に判定させない**。
このモジュールのどこにも judge モデルは居ない。採点するのは ``benchmark.BenchCase.checker``
(コードを実行する / 厳密一致を見る) だけで、``grow`` はその数字にしか従わない。

### なぜ split を 3 つに割るのか (search / confirm / sealed)

1. **search** — 変異を測って挑戦者を選ぶ場。K 個の候補の *最大値* は上振れに偏る(多重比較)。
   だから search で勝ったことは「昇格の根拠」にならず、「挑戦権」にしかならない。逆も同じで、
   search で **1 問ぶん以内**の負けは負けではない(search はその分解能を持たない。チャンピオン
   の search 点は昇格時の一回きりの最大値で、測り直されない)。挑戦権を失うのは 1 問を超えて
   負けたときだけで、その候補は confirm を測らずに落とす(``search_gate``)。
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
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .benchmark import SUITES, BenchCase, run_bench, summarize
from .backends import (_BACKENDS, ToolBackend, note_served, reset_served, reset_tool_stats,
                       served_conflicts, served_map)
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
    1 件なら ratio で最も重い split だけ —— 全クラスが 3 分割される保証ではないので、``grow`` は
    confirm に現れないクラスを変異対象から外して辻褄を合わせる。乱数は使わないので、同じ suite
    なら誰が走らせても同じ split になる。
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

    # fail-closed は肯定形で: 「search と confirm が在ると証明できた」ときだけ先へ進む。
    # confirm が空のまま回ると、昇格判定が search の上振れをそのまま通す fail-open になる。
    # search が空になるのは全クラスが 1 件で ratio が confirm 寄りのときで、そのまま進むと
    # 挑戦権の帯(1/n_search)が割り算で落ちる。「起きないはず」を配分の実装に頼らず、ここで止める。
    for name in ("search", "confirm"):
        if not out[name]:
            raise ValueError(
                f"{name} split is empty — need at least ~4 cases per class-pool to grow honestly "
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


# grow が鋳造するレーン名。括弧の後ろの ``+tool`` / ``+pf`` は同じレーンへの上乗せ(deepen・prefill)。
_DERIVED_LANE = re.compile(r"^(tool|ens|mesh)\(.*\)(\+\w+)*$")


def _prefillable(inner_spec) -> bool:
    """prefill を受け取れる inner か。backend クラスの ``supports_prefill``(肯定形の宣言)で判定する。

    構築せずに spec から引くのは、propose が決定的・オフラインだから。合成(ensemble 等)は
    登録表に無いので False —— ToolBackend も同じ条件で構築を拒むから、ここで出さない候補は
    ちょうど「作れない候補」に一致する(提案してから build で落ちる、を作らない)。
    """
    if not isinstance(inner_spec, dict):
        return False
    cls = _BACKENDS.get(inner_spec.get("backend"))
    return bool(getattr(cls, "supports_prefill", False))


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
    if isinstance(base, str):
        return base
    # 手書きの種 config は ``_grow_base`` を持たない。grow 自身の名前空間(``_DERIVED_LANE``。
    # ``validate_pool`` が利用者のレーン名を締め出している範囲)に限っては名前が構造そのものなので、
    # そこだけ逆パースする。範囲外の名前(``foo(a)`` や deepen の ``mesh(a->b)+tool``)は読まない —
    # 前者は嘘のラベルになり、後者は spec が答えを持っている。
    # 2026-09-02 まで、この関数の**名前解析版が同じファイルの後ろに残っていて**こちらを上書き
    # していた(deepen レーンに乗ったクラスは simplify も tool も一切提案されない袋小路だった)。
    if not _DERIVED_LANE.match(lane):
        return None
    inner = lane[lane.index("(") + 1:lane.rindex(")")]
    return inner.replace("->", "+").split("+")[0] if inner else None


# --------------------------------------------------------------------------- #
# Mutations — 決定的な候補生成器(LLM に「次に何を試すか」を考えさせない)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Candidate:
    label: str          # 例 "route:qa->qwen2.5:7b" — 台帳を人が読める単位にする
    kind: str           # route | tool | ensemble | meshflow | simplify
    spec: dict = field(compare=False)
    # この候補が治療するクラス(症状は tool レーンの「コードが出なかった call」、治療は prefill)。
    # 鋳造した側が書く: 処方かどうかは並び順だけでなく confirm の挑戦者の優先順位にも効くので、
    # ラベルの末尾を読み直して決めない(ラベルの形が変わっても、この欄は変わらない)。
    remedy: Optional[str] = field(default=None, compare=False)


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


def _class_of(label: str) -> str:
    """候補ラベル ``kind:class(...)`` / ``kind:class->lane`` のクラス名。"""
    body = label.split(":", 1)[1] if ":" in label else label
    return re.split(r"[(\->]", body, 1)[0]


def propose(champion: dict, pool: dict[str, dict], classes: list[str],
            width: int = 6, exclude: Optional[set] = None,
            ensemble_strategy: str = "synthesize",
            generation: int = 0,
            additive_classes: Optional[list[str]] = None,
            allow_default: bool = True,
            no_code_by_class: Optional[dict] = None,
            archived: Optional[set[str]] = None) -> list[Candidate]:
    """チャンピオンから 1 手だけ動かした候補を、種類を混ぜて**新顔** ``width`` 本返す。

    ``archived`` は search で測定済みの設計のハッシュ集合。測定済みは ``width`` の席を消費せず、
    除外されていない限り**全部**返す(呼び側は archive の点をそのまま使うのでコール 0)。
    以前は測定済みも席を 1 つ使っていて、同点だらけの走行(run W: search 32 問中 4 問しか
    残っていない)では世代が進むほど新顔が減った。W の gen0 checkpoint から 6 世代を再生する
    (gama-runs/replay-propose.py・全候補同点の仮定)と新顔は 4,3,2,3,1,0、席を使わなければ
    4,4,4,3,0,0(このチャンピオンから 1 手で行ける設計 15 本を gen4 で測り切る)。しかも
    踏み石(帯の内側で最高点でなかった設計)が次に挑戦できるのは巡回が同じ設計をもう一度出した
    時だけで、gen2 に出た research の prefill は 6 世代で一度も出直せなかった。``width`` は
    「この世代に測る設計の数」であって、測定済みを数えると幅が黙って痩せる。並ぶ測定済みは
    今のチャンピオンから 1 手で行ける設計に限られる(この関数は近傍しか生成せず、archive の
    大きさに関わらず近傍は毎回全部生成している。archive が増えて増えるのは台帳の行の長さで、
    生成も測定も増えない)。チャンピオンが替わって近傍の外に出た設計は、生成されないので並ばない。

    ``no_code_by_class`` はチャンピオンの測定が残した診断(クラス → tool レーンでコードが
    出なかった call 数)。症状のあるクラスの ``+prefill`` が処方(``_prescribed``)で、tool 枠の
    先頭に出すだけでなく、種類の巡回を待たずに最初の席を取る。巡回は「どこも均等に触る」
    ための順で、症状の場所を測定が既に言っているのに順番待ちさせるのはループが自分のデータを
    無視すること(run W: seed の confirm が research で no_code 8/72 を出していても gen1 の
    tool 枠は巡回で qa の prefill に行き、research のは checkpoint から propose を再生すると
    gen2。診断で gen0 に来る。順番の出どころを巡回の開始位置でなく測定にする)。並ぶだけでは
    足りず、挑戦者の同点処理(``_challenger_key``)も処方を先にする: W の種から症状 research=8
    で 5 世代を再生すると、処方は gen0 に並んでも同点処理に入れなければ route の差し替え・
    ensemble・default の後ろに回り(同じ大きさでラベル順が先)、5 世代で一度も挑戦者にならない。
    入れれば gen0 で挑戦する。診断が無ければ巡回のまま。

    1 手だけなのは、勝因を測定に帰属させるため(2 手同時だとどちらが効いたか台帳から読めない)。
    種類を round-robin で混ぜるのは、``width`` を絞ったときに ``route`` 変異だけで埋まって
    構造変異が一生試されない偏りを防ぐため。

    ``exclude`` は「決着がついた設計」のハッシュ集合: confirm で挑戦して負けたもの(走行を通じて)、
    および search でチャンピオンに帯(1 問)を超えて負けたもの(そのチャンピオンの間だけ。search は
    替わるまで動かないので、その負けは覆らない)。**帯の内側にいて最高点でなかっただけの設計は除外しない** —
    最高点の候補が confirm で落ちた次の世代に、archive の点のまま(追加コール無しで)挑戦者に
    なれる踏み石だから。永久追放するのは、決着がついた設計だけ。

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
    # 「足す変異」と「削る変異」で対象クラスを分ける。門が非対称だから対象も非対称になる:
    # 足す側は**良くなること**を要求されるので伸びしろの無いクラスでは通りようがないが、
    # 削る側は**悪くならないこと**しか要求しない。むしろ満点のクラスこそ「その構造は
    # 何も買っていない」ことが言える場所で、そこを削れなくするのは逆向き。
    additive = sorted(additive_classes) if additive_classes is not None else ordered_classes
    # 種類ごとに (task_type, Candidate) で貯める。task_type を保つのは、最後に**クラスも跨いで**
    # 巡回させるため — クラス順に詰めたまま width で切ると、辞書順で先頭のクラスだけが延々
    # 探索され、残りのクラスは一度も触られない(小さい width ほど効く偏り)。
    buckets: dict[str, list] = {k: [] for k in
                                ("route", "tool", "ensemble", "meshflow", "simplify",
                                 "default", "deepen")}

    for task_type in ordered_classes:
        cur = _lane_for(champion, task_type)
        cur_spec = champion["kwargs"]["backends"].get(cur) or {}
        base = _atomic_lane(champion, cur) or cur          # 合成レーンなら中身の素モデルを基準に
        for lane in lanes:                                  # ① 別モデルへ振り替える
            if lane != cur:
                buckets["route"].append((task_type, Candidate(
                    f"route:{task_type}->{lane}", "route",
                    _with_lane(champion, task_type, lane, copy.deepcopy(pool[lane])))))
        if base in pool and cur_spec.get("backend") != "tool":  # ② PAL(コードを書かせて実行)で包む
            name = f"tool({base})"
            buckets["tool"].append((task_type, Candidate(
                f"tool:{task_type}({base})", "tool",
                _with_lane(champion, task_type, name,
                           {"backend": "tool", "_grow_base": base,
                            "kwargs": {"inner": copy.deepcopy(pool[base])}}))))
        elif base in pool and not (cur_spec.get("kwargs") or {}).get("prefill") \
                and _prefillable((cur_spec.get("kwargs") or {}).get("inner")):
            # ②' すでに tool レーンなら、次の 1 手は「返答の冒頭に ```python を置く」(prefill)。
            # 素のモデルに戻す・別モデルに振り替える・合議にする、のどれも「コードを書かない」
            # 症状には効かない(run V: crux research で tool レーンが 1 度もコードを出さず、素の
            # 思考文が採点されていた)。prefill は実測で 0/3 → 2-3/3 のコード到達だが、もともと
            # コードを書けている問題群でどう転ぶかは未測定なので、既定にせず 1 手の変異にして
            # 門に通す。中身(inner)はそのまま、足すのは prefill だけ(2 手同時にしない)。
            # 届く範囲は素の tool レーンだけで、deepen で合成の内側に入った tool 段(⑦)には
            # 出さない: そこは「合成の中身を 1 手動かす」変異がまだ無く、ここだけ足すと
            # 内側の tool 段への他の手(剥がす・差し替える)と非対称になる。
            name = f"{cur}+pf"
            buckets["tool"].append((task_type, Candidate(
                f"tool:{task_type}({base})+prefill", "tool",
                _with_lane(champion, task_type, name,
                           {"backend": "tool", "_grow_base": base,
                            "kwargs": {"inner": copy.deepcopy(cur_spec["kwargs"]["inner"]),
                                       "prefill": ToolBackend.PREFILL}}),
                remedy=task_type)))
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
        if not base or base not in pool:
            continue
        spec = copy.deepcopy(champion["kwargs"]["backends"][cur])
        if spec.get("backend") == "tool":
            continue                                        # 素の tool レーンは包む段が無い
        wrapped = {"backend": "tool", "kwargs": {"inner": copy.deepcopy(pool[base])}}
        kw = spec.get("kwargs") or {}
        stages = kw.get("tiers") or kw.get("members")
        # 先頭段がもう tool なら包み直さない(名前でなく spec で見る: deepen 済みのレーン名は
        # ``mesh(a->b)+tool`` で、``tool(`` では始まらない)。
        if not stages or (stages[0].get("backend") == "tool"):
            continue
        stages[0] = wrapped
        name = f"{cur}+tool"
        buckets["deepen"].append((task_type, Candidate(
            f"deepen:{task_type}(tool inside {cur.split('(')[0]})", "deepen",
            _with_lane(champion, task_type, name, spec))))

    order = ["simplify", "route", "tool", "ensemble", "meshflow", "default", "deepen"]
    # 足す変異だけを additive なクラスに絞る(削る変異 ⑤ はそのまま全クラスに残す)。
    # ここで一括して落とすのは、①〜④⑦ の生成側に条件を撒くと足し忘れた種類から穴が開くため。
    add_set = set(additive)
    for kind in ("route", "tool", "ensemble", "meshflow", "deepen"):
        buckets[kind] = [(tt, c) for (tt, c) in buckets[kind] if tt in add_set]
    # ⑥ 既定レーンの差し替えだけはクラス単位で切れない。**既定に落ちている全クラスへ同時に
    # 効く**ので、「各クラス単独では門を越えないが合計では越える」場合がある(confirm 20 問・
    # δ=0.1 なら門は 2 問。1.5 問ずつ余っている 2 クラスは個別には飽和でも、合計 3 問で昇格
    # しうる)。だから足し合わせの判断は呼び側(伸びしろを持っている grow)に置き、ここは
    # その結論だけを受け取る。
    if not allow_default:
        buckets["default"] = []

    queues = {k: _by_class_rotation(buckets[k], ordered_classes, offset + generation)
              for offset, k in enumerate(order)}
    # 診断で先頭に出すのは prefill だけ(no_code は tool レーンの症状で、処方も prefill だけ)。
    # 症状の多いクラスから、同数はクラス名順。他の種類の順は触らない(診断が名指しするのは
    # 治療 1 本で、「どの種類を試すか」の根拠ではない)。処方の席は下で、巡回の前に取る。
    symptoms = {c: n for c, n in (no_code_by_class or {}).items()
                if isinstance(n, int) and not isinstance(n, bool) and n > 0}
    n_lead = 0
    if symptoms and queues["tool"]:
        rank = {c: (-n, c) for c, n in symptoms.items()}
        first = sorted((c for c in queues["tool"] if _prescribed(c, rank)),
                       key=lambda c: rank[c.remedy])
        lead = {id(c) for c in first}             # 同一性で除く(Candidate は値で等しくなりうる)
        queues["tool"] = first + [c for c in queues["tool"] if id(c) not in lead]
        n_lead = len(first)

    champ_hash = spec_hash(champion)
    archived = set() if archived is None else archived      # spec_hash の集合(exclude と同じ)
    out: list[Candidate] = []
    emitted: set = set()
    new = 0
    # 処方は種類の巡回を待たない。測定がクラスと治療を名指ししているのに tool の番が回るまで
    # 寝かせるのは、巡回の開始位置が順番を決める欠陥を種類の段でやり直すだけ(7 種類・width 4
    # だと gen3〜5 には tool の席が無く、その世代に昇格した champion の処方は gen6 まで待つ。
    # width 2 なら gen0 の席は simplify と route で、処方は次の世代)。席を食うのは新顔の処方
    # だけ: 測定済みなら消費せず、却下されれば exclude で消えるので、幅を食うのは一度きり。
    for _ in range(n_lead):
        if new >= width:
            break
        cand = queues["tool"].pop(0)
        h = spec_hash(cand.spec)
        if h == champ_hash or h in exclude or h in emitted:
            continue
        emitted.add(h)
        out.append(cand)
        if h not in archived:
            new += 1
    i = 0
    while new < width and any(queues[k] for k in order):
        kind = order[(i + generation) % len(order)]
        i += 1
        # width が数えるのは「出した新顔」であって「試した回数」ではない。除外(決着済み・重複)を
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
            if h in archived:
                continue                      # 測定済みも席を消費しない: 同じ種類から新顔を引く
            new += 1
            break
    # 幅を使い切った後も、測定済みの踏み石は残らず出す。巡回の位置次第で出たり出なかったり
    # すると、「archive の点のまま挑戦できる」が実際には巡回の運になる(上の docstring の実測)。
    for kind in order:
        for cand in queues[kind]:
            h = spec_hash(cand.spec)
            if h in archived and h != champ_hash and h not in exclude and h not in emitted:
                emitted.add(h)
                out.append(cand)
    return out


# --------------------------------------------------------------------------- #
# Measurement — 決定的チェッカだけを信じる
# --------------------------------------------------------------------------- #
def _state(m: "Measurement") -> dict:
    """checkpoint に残す形。**per_case を落とさない**。

    台帳の各行(`_meas`)と checkpoint は要件が逆。行は毎測定ごとに出るので 112 個の生の点を
    並べると読めなくなるが、checkpoint は 1 世代 1 行で、しかも**再開に要る状態**そのもの。
    ここで per_case を落としたせいで、再開後に静かに死ぬ機能をこのセッションで 3 回作った
    (対応のある比較・飽和判定・search 側の飽和判定)。3 回目はパッチでなく分離で直す:
    「見せるための形」と「続きを走らせるための形」を別の関数にして、名前で取り違えを防ぐ。
    """
    d = asdict(m)
    d["error_cases"] = sorted(m.error_cases)      # JSON に載る形へ(frozenset は載らない)
    return d


def _meas(m: "Measurement") -> dict:
    """台帳に出す形。``per_case`` と ``error_cases`` は落とす。

    生の case 別得点は「対応のある比較」を **その場で** するための素材で、台帳に毎回
    112 個並べると走行記録が読めなくなり、O〜T 走との形も変わって比較できなくなる。
    判定に使った要約(勝敗数と p 値)は generation 行に別途残るので、監査経路は切れない。
    """
    d = asdict(m)
    d.pop("per_case", None)
    d.pop("error_cases", None)
    return d


@dataclass
class Measurement:
    score: float
    success_rate: float
    latency_s: float
    n: int
    cases: int
    errors: int = 0            # 例外で 0 点になった call 数(モデルが間違えた 0 点とは別物)
    error_rate: float = 0.0
    # case ごとの得点(repeats 平均)。集計値だけ持ち帰ると**同じ問題での勝ち負け**が消え、
    # 「平均が上がった」以上のことが何も言えなくなる。床を 1/n と drift から作っている限り
    # 支配的な誤差である「どの問題がその split に入ったか」を見ていないので、対応のある比較
    # (同一 case で champion と challenger を突き合わせる)ができる素材をここで残す。
    per_case: dict = field(default_factory=dict)
    # 例外で 0 点になった case。0 点は「モデルが間違えた」ではないので、対応のある比較で
    # 「負け」と数えると測定失敗が証拠に化ける。除外できるように id を持ち帰る。
    error_cases: frozenset = frozenset()
    # tool レーンが実際にコードを走らせた回数 / 通った回数。fell_back が多いレーンは
    # 「道具として測れていない」ので、低い得点をモデルの弱さと読んではいけない。
    tool_calls: int = 0
    tool_ran: int = 0
    tool_no_code: int = 0      # ```python が出てこなかった(prompt/モデル側の問題)
    tool_empty_out: int = 0    # コードは走ったが何も print しなかった(生成コード側の問題)
    # クラス別の「コードが出てこなかった」call 数。合計は「どこかで出ていない」としか言わず、
    # 処方(prefill はクラス単位の変異)を向ける先が読めない。空 dict は「tool レーンを通った
    # call でコードが出なかった事は無い」で、古い checkpoint(この欄が無い)も同じ形で復元する。
    tool_no_code_by_class: dict = field(default_factory=dict)


def measure(spec: dict, cases: list[BenchCase], tier: ModelTier = ModelTier.LARGE,
            repeats: int = 1, unit_cost: Optional[dict] = None,
            label: str = "candidate") -> Measurement:
    """1 つの spec を case 集合で測る。例外は ``run_bench`` 側で 0 点に落ちる(掃引を止めない)。"""
    if not cases:
        raise ValueError("measure() needs at least one case (an empty split has no score, "
                         "and returning 0.0 would read as 'measured and failed')")
    backend = build_backend(spec)
    reset_tool_stats()
    records = run_bench({label: backend}, suite=cases, tier=tier, repeats=repeats,
                        unit_cost=unit_cost or {}, run_id="grow")
    agg = summarize(records)["overall"][label]
    # `run_bench` は例外を握って 0 点にする(1 つの backend が掃引を止めないため)。その 0 点は
    # 「モデルが間違えた」ではなく「**測れなかった**」で、区別を捨てると死んだ backend の
    # 全ゼロが実測値として判定に入る。件数を持ち帰る。
    errors = sum(1 for r in records if r.get("error"))
    by_case: dict[str, list[float]] = {}
    for r in records:
        by_case.setdefault(r["case_id"], []).append(r["score"])
    per_case = {cid: sum(v) / len(v) for cid, v in by_case.items()}
    error_cases = frozenset(r["case_id"] for r in records if r.get("error"))
    # tool の計数は合計もクラス別も**記録の差分から**足す(1 つの出所)。合計だけグローバルの
    # 累積値から読むと「クラス別の和 = 合計」が reset のタイミングに依存する偶然になり、
    # reset を挟み忘れた呼び出し経路からずれが入る(codex diag-r1)。
    ts = {"calls": 0, "ran": 0, "no_code": 0, "empty_out": 0}
    no_code_by_class: dict[str, int] = {}
    for r in records:
        d = r.get("tool") or {}
        for k in ts:
            ts[k] += d.get(k, 0)
        if d.get("no_code"):
            no_code_by_class[r["task_type"]] = no_code_by_class.get(r["task_type"], 0) + d["no_code"]
    return Measurement(score=agg["score"], success_rate=agg["success_rate"],
                       latency_s=agg["latency_s"], n=agg["n"], cases=len(cases),
                       errors=errors, error_rate=round(errors / len(records), 4) if records else 0.0,
                       per_case=per_case, error_cases=error_cases,
                       tool_calls=ts["calls"], tool_ran=ts["ran"],
                       tool_no_code=ts["no_code"], tool_empty_out=ts["empty_out"],
                       tool_no_code_by_class=dict(sorted(no_code_by_class.items())))


# --------------------------------------------------------------------------- #
# 対応のある比較 — 「平均が上がった」と「同じ問題で勝った」を分ける
# --------------------------------------------------------------------------- #
def paired_gain(champion: Measurement, challenger: Measurement,
                tol: float = 1e-9) -> tuple[int, int, int]:
    """同一 case での勝ち/負け/引き分けを数える。両者に在る case だけを対象にする。

    平均差は「どの問題が split に入ったか」に強く依存するが、**同じ問題での勝敗**は
    その依存を打ち消す(対応のある比較)。ここで返す数は次の sign test の素材で、
    この関数自体は何も判定しない。
    """
    # 片方でも測れなかった case は落とす。0 点は「測れなかった」の印であって負けではなく、
    # 混ぜると死んだ backend ほど「相手の勝ち」を大量生産する。
    shared = ((set(champion.per_case) & set(challenger.per_case))
              - champion.error_cases - challenger.error_cases)
    wins = losses = ties = 0
    for cid in shared:
        d = challenger.per_case[cid] - champion.per_case[cid]
        if d > tol:
            wins += 1
        elif d < -tol:
            losses += 1
        else:
            ties += 1
    return wins, losses, ties


def sign_test(wins: int, losses: int) -> float:
    """符号検定の片側 p 値(厳密)。「差の出た問題のうち勝ちに偏った」が偶然でない確からしさ。

    引き分けは情報を持たないので落とす(McNemar と同じ扱い)。帰無仮説は「勝ち負けは五分」。
    scipy は使わない(stdlib 縛り)。返すのは p 値だけで、閾値の判断は呼び側に置く。
    """
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(wins, n + 1))
    return tail / (2 ** n)


# --------------------------------------------------------------------------- #
# The gate — 昇格を決める唯一の関数(肯定形: 「昇格してよいと証明できた」ときだけ True)
# --------------------------------------------------------------------------- #
# 3 つの門が共有する境界の遊び。score は ``summarize`` が 4 桁に丸めた値で、門の幅(1/n や
# drift)は生の値。丸めた 2 点の差は真の差から最大 1e-4 ずれるので、「1 問ぶんちょうど」を
# 通す約束は 1e-4 の遊びが無いと**丸めの向きで通ったり落ちたりする**(15 問なら 11/15 − 10/15
# が 0.0666 と出て 0.066667 に届かない。72 問なら 0.0138 と 0.0139 の両方が起きる)。以前の
# 1e-9 は浮動小数の表現誤差しか吸わず、この丸めには足りていなかった。1e-4 を超える差は
# 4 桁の世界で「本当に違う」差なので、どちらの向きにも誤判定を作らない。
SCORE_TOL = 1e-4


def search_gate(champion_search: float, challenger_search: float,
                band: float) -> tuple[bool, str]:
    """① 挑戦権。search で **band を超えて**負けていなければ通す(上回る必要はない)。

    search は挑戦者を**選ぶ**ための split で、昇格を決める split ではない(決めるのは confirm)。
    だからここは証拠の門ではなく**コストの門**: confirm(search の 2 倍の問数)を焚く価値が
    無いほど負けている候補だけを落とす。かつては「厳密に上回ること」を要求していたが、
    それは 2 つの理由で挑戦者を不当に落としていた(どちらも run V の台帳で実測):

    * チャンピオンの search 点は**昇格時の一回きりの測定**で、その後測り直されない。昇格時は
      width 本の最大値なので上振れしている(同じ走で confirm 側は毎世代測り直され、
      0.7838 → 0.7592 と下がり続けた。search 側の 0.901 だけがその補正を受けない)。
    * search は 32 問しかなく、同じ設計を測り直すと丸 1 問ぶん動いた(0.8958 と 0.8646)。
      その分解能で「0.5 問負け」を負けと読み、confirm で 4勝1敗・4勝0敗(+1.6 問・+1.7 問)
      だった候補を confirm も見ずに捨てていた。

    ``band`` は search の分解能(1 問ぶん = 1/n_search)。band ちょうどまでは通す(削減の門と
    同じ向き)。band=0 なら「同点は通し、少しでも負けたら落とす」。
    """
    if (champion_search - challenger_search) > band + SCORE_TOL:
        return False, f"search-worse(band={round(band, 4)})"
    return True, "search"


def promote_gate(champion_search: float, challenger_search: float,
                 champion_confirm: float, challenger_confirm: float,
                 delta: float, paired: Optional[tuple[int, int, int]] = None,
                 max_paired_p: Optional[float] = None,
                 search_band: float = 0.0) -> tuple[bool, str]:
    """3 条件が**すべて**証明できたときだけ昇格。理由は台帳に残せる文字列で返す。

    ① 挑戦権: search で ``search_band`` を超えて負けていないか(``search_gate``)。search は
       選抜用で 1 問未満の差を分けられないので、上回ることは要求しない。既定の band 0.0 でも
       **同点は通る**(以前の「厳密に上回れ」は撤回。チャンピオンの search 点は昇格時の一回きりの
       最大値なので、同点を負けと読むと上振れした古い点に永久に勝てない)。
    ② held-out: confirm で上回ったか(search の上振れは confirm を通らない)
    ③ 幅超え: 差が δ 以上か。δ = max(confirm 1 問ぶん, チャンピオン自身の測り直しの揺れ)。
       1 問未満の差は部分点の揺らぎでしかなく、揺れの内側の差は改善ではない。
    """
    ok, why = search_gate(champion_search, challenger_search, search_band)
    if not ok:
        return False, why
    if not challenger_confirm > champion_confirm:
        return False, "confirm-not-better"
    # 「1 問ぶんちょうど」は通す約束なので、比較には丸めの遊び(SCORE_TOL)を持たせる。
    if (challenger_confirm - champion_confirm) < delta - SCORE_TOL:
        return False, f"below-margin(delta={round(delta, 4)})"
    # ④(任意) 対応のある証拠: 同一 case での勝敗が偶然に見えないか。既定は無効で、
    #    有効にすると門が一気に厳しくなる(実測: alpha=0.05 は 5勝0敗級を要求し、過去の
    #    昇格はほぼ全部止まる)。判断は運用者に置き、p 値は常に台帳へ出す。
    #    ①〜③ より**後**に見る: 平均差の床も割っている候補を "paired-not-significant" と
    #    記録すると、台帳の理由が根本原因を隠す。理由は常に最も基本的な落ち方を名指しする。
    if max_paired_p is not None:
        # 検定を要求されたのに材料が無いのは fail-open。肯定形で止める(「有意だと証明
        # できた」ときだけ通す)。黙って素通りする経路を作らない。
        if paired is None:
            raise ValueError("promote_gate(max_paired_p=...) requires `paired` win/loss counts; "
                             "without them the condition would silently pass")
        w, l, _ = paired
        pv = sign_test(w, l)
        if pv > max_paired_p:
            return False, f"paired-not-significant(p={pv:.3f},{w}w-{l}l)"
    return True, "promote"


_COMPOSITES = ("tool", "ensemble", "meshflow", "trinity", "abmcts")


def code_stamp(where: Optional[Path] = None) -> dict:
    """どのコードがこの数字を出したかを台帳に残すための刻印。

    この loop は稼働しながら**判定そのもの**を何度も変えてきた(探索の巡回・ensemble の集約・
    削減の門・その許容幅)。台帳に走行条件(suites/ratio/repeats)しか無いと、同じ条件で取った
    はずの数字が実は別の門で出ていた、という比較を後から検出できない。checkout から動いている
    ときは git の短縮 SHA を、そうでなければ version だけを残す(取れないこと自体は異常でない)。

    SHA だけでは足りなかった。run X(2026-08-30)は HEAD ffdb5bf の上に未コミットの grow.py を
    載せた tree を import して走り、刻印は ffdb5bf と言った。数字を出したのは HEAD でなく tree
    なので、package dir に HEAD と違う物があるかを `dirty` として並べて残す(True は「SHA が
    全部ではない」の意味。None は git で確かめられなかった)。見るのは `where` の配下だけで、
    実走では package dir: README や tests の編集は数字を変えない。未追跡のファイルも数える。
    import されうる新しい module は commit に無いし、どれが読まれるかをここで知る手は無いので、
    `dirty` は「振る舞いが違う」でなく「commit と違う」を言う。
    `where` はテストが自前の repo を指すためのもの。repo root を渡せば root 配下全部を見る。
    """
    import subprocess

    where = Path(__file__).resolve().parent if where is None else Path(where)
    sha, dirty = None, None
    try:
        proc = subprocess.run(["git", "-C", str(where), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5)
        if proc.returncode == 0:
            sha = proc.stdout.strip() or None
    except Exception:      # git が無い / インストール済みパッケージ / 何であれ致命ではない
        sha = None
    if sha:
        # 取れた SHA は status が転んでも手放さない(dirty だけ None に残る)。
        try:
            proc = subprocess.run(["git", "-C", str(where), "status", "--porcelain", "--", "."],
                                  capture_output=True, text=True, timeout=5)
            if proc.returncode == 0:
                dirty = bool(proc.stdout.strip())
        except Exception:
            dirty = None
    from . import __version__

    return {"version": __version__, "commit": sha, "dirty": dirty}


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
    if (champion_confirm - challenger_confirm) > band + SCORE_TOL:
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
    """再開先の台帳が使っていた split(case id)。分割が違えば再開してはいけない。

    seed 行だけでなく resumed 行からも読む: 別ファイルへ再開した台帳(と、この修正より前に
    同じファイルへ再開して truncate された台帳)は resumed 行から始まり、そこから再開すると
    split の検査が空振りしていた。
    """
    try:
        for line in Path(ledger_path).read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            # split の無い行(旧形式・手で直した行)は飛ばして、持っている最初の行を採る
            if row.get("event") in ("seed", "resumed") and row.get("splits") is not None:
                return row["splits"]
    except OSError:
        return None
    return None


MAX_ERROR_RATE = 0.2


class MeasurementFailure(RuntimeError):
    """測定そのものが壊れているときに投げる。**スコアが低いのとは別の事故**。

    実際に起きたこと(run S): 走行中に配信サーバが別モデルへ載せ替えられ、以降の全コールが
    503 を返した。``run_bench`` は例外を握って 0 点にするので、loop から見ると「チャンピオンも
    挑戦者も 0.0、drift も 0.0」という**整合の取れた実測値**に見え、門は全部正しく動いた上で
    「レーンを外してもコストゼロ」と判断して構造を捨てた。最終チャンピオンも sealed も、
    死んだ backend の産物だった。台帳は一見正常に見える。

    低いスコアと測れないことは別なので、後者は走行を止める。checkpoint があるので、backend が
    戻ってから ``--resume`` で続けられる。
    """


def _guard_measurement(m: "Measurement", what: str) -> None:
    if m.error_rate > MAX_ERROR_RATE:
        raise MeasurementFailure(
            f"{what}: {m.errors} of {m.n} calls raised ({m.error_rate:.0%}). That is a broken "
            "measurement, not a low score — refusing to decide on it. Fix the backend and "
            "--resume from this ledger.")
    # 相手が途中で入れ替わっていないか。run S は載せ替えが 503 を返したので error_rate の床に
    # 引っかかったが、200 を返しながら中身が変わる載せ替えは何も鳴らさず、世代 0 と世代 5 が
    # 別モデルの比較になる。要求名 -> 供給名の対応が割れた時点で走行を止める。
    conflicts = served_conflicts()
    if conflicts:
        detail = "; ".join(f"{k} served as {' and '.join(v)}" for k, v in conflicts.items())
        raise MeasurementFailure(
            f"{what}: the backend changed under the run ({detail}). Generations measured before "
            "and after that point are not comparable, so this run cannot be decided. Restart the "
            "server on the intended model and --resume from this ledger.")


def promoted_gain_cases(history: list, n_confirm: Optional[int]) -> Optional[float]:
    """門が昇格の**その時点で**認めた confirm 上の伸びの合計(問)。

    チャンピオンが実際に入れ替わった世代だけを数える(``promotions`` と同じ定義)。足した手は
    ``gain_cases``、削った手は ``simplify_confirm`` と当該世代の champion 実測との差で、どちらも
    その世代の confirm 上で門を通った量そのもの。読めない行(古い台帳・測っていない世代)は
    0 ではなく None を返す: 「主張が無い」と「主張が読めない」を同じ数にすると、後者が
    「検定力あり」に化ける。

    これは sealed が検定する仮説の大きさ**ではない**。sealed が見るのは最終形と種の差で、その
    confirm 側の対応物は ``confirm_claim`` が出す「最終チャンピオンの confirm − 種の confirm」。
    手ごとの認定を足し上げると、それより大きく出る: 昇格した次の世代から champion は毎世代
    測り直され、点は種の側へ戻っていく(run V: 昇格時に +1.10 問を認定、測り直しの平均は種より
    −0.08 問。run T: +1.25 に対し +0.84)。上書き・相殺する手があっても同じ。だからこちらは
    「門が何を認めたか」の記録として、判定の横に**並記**する。
    丸めない: 丸めは表示の仕事で、判定側が丸めた値で分岐しないため。
    """
    if not isinstance(n_confirm, int) or isinstance(n_confirm, bool) or n_confirm <= 0:
        return None
    total, seen = 0.0, False
    for h in history or []:
        # 両キーが無い行は「入れ替わっていない」でなく「読めない」(None == None を未変更と
        # 読むと、古い台帳の認定合計が 0 に化ける。codex r4)。
        if not isinstance(h, dict) or "champion_hash" not in h or "champion_after" not in h:
            return None
        if h["champion_hash"] == h["champion_after"]:
            continue
        seen = True
        if h.get("simplify_verdict") == "promote":
            a, b = _num(h.get("simplify_confirm")), _num(h.get("champion_confirm"))
            if a is None or b is None:
                return None
            total += (a - b) * n_confirm
        elif _num(h.get("gain_cases")) is not None:
            total += _num(h["gain_cases"])
        else:
            return None
    return total if seen else 0.0


def _num(v) -> Optional[float]:
    """JSON 由来の値を数として読む。bool は数ではない(台帳の true が 1 問に化ける)。"""
    return (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v) else None)


def confirm_claim(seed_scores: list, champion_scores: list, n_confirm: int,
                  promotion_score: Optional[float] = None) -> dict:
    """最終形が confirm で種より何問上に立つか = sealed が検定する仮説の confirm 側の大きさ。

    sealed は種と最終形を一度ずつ測って比べる。confirm 側でその対応物を出すとき、どの測定を
    使うかで ±1 問(=帯そのもの)動く。run T は種を 3 回測って 0.8619/0.8485/0.8440、最終形は
    3 回とも 0.8664: 種の初回だけ見れば +0.25 問、昇格直前の値を見れば +1.25 問、平均なら
    +0.84 問。1 回の測定が 1 問揺れることは drift の測定自体が毎世代示しているので、単発の
    一対で決めない。

    推定は**選択に使っていない測定の平均**で行う:
      * 種は測った全部(種は confirm で選ばれた側になったことが無い)。
      * 最終形は**昇格の後の測り直し**の全部。昇格時の測定は候補の中で高かったから残った値で
        (run V: 昇格時 0.7838、以後 0.7792/0.7637/0.7608/0.7592)、含めると主張が上振れる。
        測り直しが無い(最終世代で昇格した)ときだけ、それしか無いので昇格時の値を使い、
        ``promotion_only`` で申告する。
    最終形が種と同一(何も通らなかった・足して戻した)なら主張は 0: 同じ設計を測り直した差は
    揺れであって主張ではなく、そこから検定力を語らない。
    """
    def _mean(xs):
        return sum(xs) / len(xs)

    # 読める数だけで立てる。種の測定が一つも無ければ主張は「無い」でなく「読めない」(None)で、
    # ``sealed_verdict`` はそれを「not recorded」と読む。ここで割り算を落とすと成果物の生成ごと
    # 落ちる(codex r4)。
    seed_scores = [x for x in (_num(v) for v in (seed_scores or [])) if x is not None]
    champion_scores = [x for x in (_num(v) for v in (champion_scores or [])) if x is not None]
    promotion_score = _num(promotion_score)
    seed_mean = _mean(seed_scores) if seed_scores else None
    if not champion_scores and promotion_score is None:
        # 種のまま終わった走行。champion の測定は種の測定そのものなので、別個に測った数としては
        # 0 と書く(同じ数を両方に書くと、種を n 回・champion を n 回測ったように読める)。
        return {"cases": None if seed_mean is None else 0.0,
                "seed_mean": None if seed_mean is None else round(seed_mean, 4),
                "seed_measurements": len(seed_scores),
                "champion_mean": None if seed_mean is None else round(seed_mean, 4),
                "champion_measurements": 0,
                "promotion_only": False, "same_as_seed": True}
    promotion_only = not champion_scores
    champ_mean = promotion_score if promotion_only else _mean(champion_scores)
    # 種が読めなければ主張は「読めない」(None)。champion 側の材料は捨てずに残す(codex r5)。
    return {"cases": None if seed_mean is None else (champ_mean - seed_mean) * n_confirm,
            "seed_mean": None if seed_mean is None else round(seed_mean, 4),
            "seed_measurements": len(seed_scores),
            "champion_mean": round(champ_mean, 4),
            "champion_measurements": 1 if promotion_only else len(champion_scores),
            "promotion_only": promotion_only, "same_as_seed": False}

def sealed_verdict(sealed: Optional[dict], claimed_gain_cases: Optional[float] = None,
                   confirm_cases: Optional[int] = None,
                   promoted_gain_cases: Optional[float] = None,
                   claim_basis: Optional[str] = None) -> dict:
    """封をした split が、この走行そのものについて何と言っているか。

    ここまでのゲートはすべて confirm の上で判定していて、confirm は世代をまたいで**繰り返し
    選択に使われる**から上振れする(実測 3 走で confirm +6.25/+8.85/+0.45% に対し sealed は
    +1.65/+1.39/-1.79%)。sealed は一度しか開けないので、走行全体に対する唯一の外部の目になる。

    判定の帯は sealed 自身の分解能(1 問)。sealed も 28 問しかなく、0.5 問の下落は害の証明に
    ならない。だから 3 値で返す: 効いた / 分からない / 悪くなった。**「分からない」を
    「効いた」に丸めない**ことがこの関数の全部で、run T はまさにそこを黙って通していた
    (sealed 0.8393 -> 0.8214 なのに「昇格 1・成功」として champion を出した)。

    ``write_recipe`` は任意の result に対してこれを呼ぶので、**全域関数**にしてある。読めない
    入力で例外を投げると、判定を落とすだけで済むはずが成果物ごと落ちる。区別は 2 語に分けた:
    ``unsealed`` = そもそも split が無い / ``unjudgeable`` = 在るが比べられない。

    「分からない」は一語だが中身は三つある(2026-09-02・7 走の台帳を並べて判明):
      * **検定力が無かった**: 最終形が confirm で種より上に立つ量を sealed の問数に直すと
        帯(1 問)の内側。run T は +0.84/56 問 → sealed 28 問なら 0.42 問。完全に転移していても
        「分からない」にしかならない走行で、その判定は**封を開ける前から決まっていた**。
      * **検定力はあったが転移しなかった**: run R は +4.0/48 → 2.0 問ぶん、run Q は
        +2.5/40 → 1.25 問ぶんを主張して、sealed ではどちらも +0.33 問。これは confirm 上の
        選択が上振れだった証拠で、「分からない」ではなく「confirm の伸びは本物でなかった」と
        読むべき結果。
      * **主張が走行の中で蒸発していた**: 門は昇格時に伸びを認定するが、champion は次の世代
        から毎世代測り直され、点は種の側へ戻る。run V は昇格時に +1.10 問を認定しながら、最終
        形の測り直し 4 回の平均は種より −0.08 問。sealed が検定する仮説(最終形 vs 種)は封を
        開ける前に走行自身の数字が取り下げていて、sealed に見せるものが無かった。
    三つとも同じ ``not-separable`` を返すが、``power`` と note で言い分ける。判定の値を増やさない
    のは、recipe/README の分類(3 値)を壊さないためと、どれも「champion を出荷しない」という
    行動は同じだから。違うのは**次に何をするか**(split を広げる / 手を疑う / 門を疑う)で、
    それは note が言う。

    検定力は ``claimed_gain_cases`` = 最終チャンピオンの confirm − 種の confirm(問、
    ``confirm_claim`` の推定)で決める。sealed が比べるのがまさに最終形と種だから、confirm 側の
    主張もその形で測る。昇格時の認定の合計 ``promoted_gain_cases`` は判定に使わず並記する
    (codex r2: 手ごとの認定を足すと、測り直しで戻った分や相殺した分だけ最終形の主張より大きく
    見積もる)。``claim_basis`` は主張がどの測定から出たかの一句で、note にそのまま入る。
    ``claimed_gain_cases`` と ``confirm_cases`` が無ければ従来どおり(power は None)。
    ``power`` は verdict の下位分類ではない: confirm の主張が sealed に何を予告していたかの札で、
    verdict と独立に付く。improved なのに underpowered は「sealed の伸びが confirm の主張より
    大きい」、regressed で evaporated は「二つの集合が同じ向きを言っている」で、どちらも矛盾では
    なく情報。not-separable の理由として読むのは、not-separable の時だけ。
    """
    def _score(m) -> Optional[float]:
        v = m.get("score")
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) \
            and math.isfinite(v) else None

    def _cases(m) -> Optional[int]:
        v = m.get("cases")
        return int(v) if isinstance(v, int) and not isinstance(v, bool) and v > 0 else None

    if not sealed or not isinstance(sealed, dict):
        return {"verdict": "unsealed", "delta_cases": None, "band_cases": None,
                "note": "no sealed split: every number fed a decision, so read them as optimistic",
                "claimed_confirm_cases": None, "promoted_confirm_cases": None,
                "expected_cases": None, "power": None}

    def _unjudgeable(note):
        return {"verdict": "unjudgeable", "delta_cases": None, "band_cases": None, "note": note,
                "claimed_confirm_cases": None, "promoted_confirm_cases": None,
                "expected_cases": None, "power": None}

    seed_m, champ_m = sealed.get("seed"), sealed.get("champion")
    if not isinstance(seed_m, dict) or not isinstance(champ_m, dict):
        return _unjudgeable("the sealed block does not hold a seed and a champion measurement, "
                            "so this run is not judged")
    s_score, c_score = _score(seed_m), _score(champ_m)
    s_n, c_n = _cases(seed_m), _cases(champ_m)
    if None in (s_score, c_score, s_n, c_n):
        return _unjudgeable("the sealed measurements are missing a usable score or case count, "
                            "so this run is not judged")
    if s_n != c_n:
        # 分母が違えば差は問数に直せない。丸めずに「比べられない」と言う。
        return _unjudgeable(f"the sealed split was measured over {s_n} cases for the seed and "
                            f"{c_n} for the champion: those two numbers are not a comparison, "
                            "so this run is not judged")

    delta, band = c_score - s_score, 1.0 / s_n
    # 主張した伸びを sealed の問数に直す。confirm で +G 問なら sealed(n 問)では G*n/n_confirm 問が
    # 「完全に転移した時に見えるはずの量」。帯(1 問)に届かなければ、この走行は sealed に何も
    # 言わせられない設計だった。
    expected = power = None
    claimed, promoted = _num(claimed_gain_cases), _num(promoted_gain_cases)
    # confirm の問数は主張を sealed の問数に直す分母。無い/壊れている呼び出しでは主張の問数を
    # 言わない(codex r3: 分母なしで `of None confirm cases` と印字する経路があった)。
    has_confirm_n = (isinstance(confirm_cases, int) and not isinstance(confirm_cases, bool)
                     and confirm_cases > 0)
    if claimed is not None and has_confirm_n:
        # 判定は丸める前の値で。表示用に 2 桁へ丸めた値で分岐すると 1.004 問が 1.00 になって
        # 「検定力なし」に化ける(この repo が門で一度踏んだ罠と同じ形)。
        exact = claimed * s_n / confirm_cases
        expected = round(exact, 2)
        if claimed > 0:
            power = "underpowered" if exact <= 1.0 else "powered"
        elif promoted is not None and promoted > 0:
            # 門は伸びを認めたが、最終形は種より上に立っていない: 主張は走行の中で蒸発した。
            # 「何も主張していない」と同じ札にすると、門が認定を出したという事実が消える。
            power = "evaporated"
        else:
            power = "nothing-claimed"
    # 昇格時の認定の合計は判定に使わず、note で主張の横に並べる(読めない台帳ではその旨)。
    certified = (f"its promotions certified {promoted:+.2f} at promotion time"
                 if promoted is not None else "the promotion-time total is not recorded")
    if claim_basis:
        certified = f"{claim_basis}; {certified}"
    if delta > band:
        v, note = "improved", "the held-out split agrees the champion is better than the seed"
    elif delta < -band:
        v, note = ("regressed",
                   "the held-out split says the champion is WORSE than the seed it started "
                   "from: the gains measured on confirm did not survive contact with cases "
                   "that never fed a decision. Do not adopt this champion")
    else:
        v = "not-separable"
        if power == "underpowered":
            # 分離に要る sealed の問数(この主張のまま)と、この sealed で分離に要る主張の大きさ。
            # 「どちらを動かすか」は運用者の判断なので両方言う。
            need_n = int(confirm_cases / claimed) + 1
            need_g = round(confirm_cases / s_n, 2)
            note = (f"the held-out split cannot tell the champion from the seed, and it never "
                    f"could have: on confirm the champion stands {claimed:+.2f} of "
                    f"{confirm_cases} cases over the seed ({certified}), which is {expected:g} "
                    f"of the {s_n} sealed cases, not beyond the one case this split resolves. This "
                    f"verdict was fixed before the split was opened. To be separable at this "
                    f"gain the sealed split needs at least {need_n} cases; on this split the "
                    f"champion would have to stand more than {need_g:g} confirm cases over "
                    f"the seed")
        elif power == "powered":
            note = (f"the held-out split cannot tell the champion from the seed, although it "
                    f"could have: on confirm the champion stands {claimed:+.2f} of "
                    f"{confirm_cases} cases over the seed ({certified}), which should show as "
                    f"{expected:g} of the {s_n} sealed cases if it were real, and the split "
                    f"shows {delta * s_n:+.2f}. The confirm gains did not transfer: read the "
                    f"promotions as selection on confirm, not as an improvement")
        elif power == "evaporated":
            note = (f"the held-out split cannot tell the champion from the seed, and the run's "
                    f"own numbers had already retracted the claim before the split was opened: "
                    f"the promotions certified {promoted:+.2f} of {confirm_cases} confirm cases "
                    f"when they were promoted, but re-measured at the end the champion stands "
                    f"{claimed:+.2f} over the seed. The certified gains evaporated on "
                    f"re-measurement: read them as selection on confirm, and the champion as no "
                    f"better than the seed")
        elif claimed is None and promoted is not None and promoted > 0:
            # 主張が読めない台帳(seed の confirm が残っていない再開)でも、門が認定を出した事実は
            # 読める。検定力は言えないので言わない。
            of_n = f"of {confirm_cases} confirm cases" if has_confirm_n else "on confirm"
            note = (f"the held-out split cannot tell the champion from the seed at this case "
                    f"count, and whether it could have is not recorded: the promotions "
                    f"certified {promoted:+.2f} {of_n} at promotion time, but the seed's "
                    f"confirm score was not carried to the end of this run, so how much of "
                    f"that the champion still claims is unknown")
        else:
            note = ("the held-out split cannot tell the champion from the seed at this case "
                    "count. The run neither proved nor disproved an improvement")
    return {"verdict": v, "delta_cases": round(delta * s_n, 2), "band_cases": 1.0, "note": note,
            "claimed_confirm_cases": None if claimed is None else round(claimed, 2),
            "promoted_confirm_cases": None if promoted is None else round(promoted, 2),
            "expected_cases": expected, "power": power}


def class_headroom(m: "Measurement", cases: list) -> dict:
    """クラスごとに「まだ取れていない点」を問数で返す。per_case が無ければ空(=判定しない)。

    ここが 1 問未満のクラスは、**どんな変異を当てても昇格の床を越えられない**。床は
    「confirm 1 問ぶん」で、変異が触れるのはそのクラスの case だけだから、取りうる最大の
    伸びが 1 問に満たないなら証明として通らない。推定ではなく算術で言える。
    """
    if not m.per_case:
        return {}
    out: dict = {}
    for c in cases:
        got = m.per_case.get(c.case_id)
        if got is None:
            continue
        out[c.task_type] = out.get(c.task_type, 0.0) + (1.0 - got)
    return out


def _default_swap_viable(champion: dict, classes: list, headroom: dict,
                         gate_cases: float) -> bool:
    """既定レーンの差し替えが昇格しうるか。クラス単位の飽和では切れない唯一の変異。

    レーン変異は 1 クラスしか触らないので「そのクラスの伸びしろ < 門」で切れるが、既定の
    差し替えは**既定に落ちている全クラスへ同時に効く**。confirm 20 問・δ=0.1 なら門は 2 問で、
    1.5 問ずつ余っている 2 クラスは個別には飽和でも合計 3 問ぶん動きうる。足してから比べる。

    伸びしろが測れない(``headroom`` が空)ときは True —— 飽和は証明できたときだけ主張する。
    """
    if not headroom:
        return True
    default_lane = (champion.get("kwargs") or {}).get("default")
    under = [c for c in classes if _lane_for(champion, c) == default_lane]
    # 測れていないクラスが 1 つでも既定の下に在るなら、合計は下から押さえられない。
    # 欠損を 0 と読むと「測っていない」が「伸びしろ無し」に化ける。
    if any(c not in headroom for c in under):
        return True
    if sum(headroom[c] for c in under) < gate_cases:
        return False
    return True


def _restore(d: dict) -> "Measurement":
    """checkpoint の dict を Measurement に戻す。古い台帳(per_case 無し)もそのまま読める。"""
    d = dict(d)
    d["error_cases"] = frozenset(d.get("error_cases") or ())
    # この分離より前に書かれた台帳には per_case が無い。空で復元すると search 側の飽和判定は
    # 昇格が起きて champ_search が測り直されるまで効かないが、それが正しい向き —— 飽和は
    # 証明できたときだけ主張する。証拠が無いことを「余地なし」の側に丸めない。
    d.setdefault("per_case", {})
    # 診断の無い古い checkpoint は空で復元する。confirm は毎世代測り直すので、再開後の最初の
    # 世代から診断は戻る。search 側の分だけは次の昇格(champ_search の測り直し)まで無い。
    d.setdefault("tool_no_code_by_class", {})
    return Measurement(**d)


def _prescribed(cand: "Candidate", symptoms: dict) -> bool:
    """その候補が、チャンピオンの診断が指した処方か(症状のあるクラスへの ``+prefill``)。

    診断は tool レーンの「コードが出なかった call 数」(クラス別)で、処方は今のところ prefill
    だけ。propose の並び順と挑戦者の同点処理が**同じ判定**を使うために分けてある(片方だけ
    直すと、列の先頭に出した処方が同点処理で後回しになる、という食い違いが起きる)。
    どのクラスを治療する候補かは鋳造時に ``Candidate.remedy`` へ書かれていて、ここはそれと
    診断を突き合わせるだけ(ラベルは人が読む単位で、判定の根拠にしない)。
    """
    return cand.remedy is not None and cand.remedy in symptoms


def _challenger_key(cand: "Candidate", m: "Measurement", prescribed: bool = False) -> tuple:
    """同点の候補をどう並べるか。**再現しない量を読まない**ことがこの関数の要件。

    点数(``m.score``)は読む —— 同じ spec を同じ case 集合で測れば同じ値になるので、
    走行をまたいで再現する。読まないのは壁時計のように**測るたびに変わる量**の方。

    以前はここに実測レイテンシが入っていて、走行が決定的にならなかった(壁時計は走るたび
    違い、共有 GPU では他人の負荷でも動く。determinism テストが 80 回中 4 回落ちた)。
    レイテンシで「安い方」を選ぶ意図自体は正しかったが、その代理として測定ゆらぎを
    判定に入れてしまっていた。構造の大きさは spec だけで決まり、追加コールを生んでいる
    当のものなので、意図を保ったまま再現する。
    分けて名前を与えてあるのは、この性質を文字列検査でなく**振る舞いとして**試験できる
    ようにするため。並び順は「点の高い順 → 処方が先 → 構造の小さい順 → ラベル順」。

    ``prescribed`` は「チャンピオンの診断が指した処方」(``_prescribed``)。同点の中で先に
    confirm を測るのは処方: search はそのクラスをほぼ取り切っていて payoff を見せられず
    (run W の search は research の未解決が 1 問)、外しは confirm 側に残っている(同 5 問)。
    ここに入れないと、列の先頭に出した処方が同点処理で構造とラベルの順に戻される。W の種
    (= X の種)から症状 research=8 で 5 世代を再生する(gama-runs/replay-propose.py・全候補
    同点の仮定)と、処方を同点処理に入れない場合の挑戦者は route:content → ensemble:research
    → default → route:research → route:qa で、gen0 から毎世代並んでいる処方は一度も confirm を
    測られない。入れれば gen0 で挑戦する。処方かどうかはラベルと診断だけで決まるので走行を
    またいで再現する。
    """
    return (-m.score, 0 if prescribed else 1, _structure_size(cand.spec), cand.label)


def grow(pool: dict[str, dict], *, classes: Optional[list[str]] = None,
         cases: Optional[list[BenchCase]] = None, suites=("wide", "hard", "brutal"),
         ratio: tuple[int, int, int] = (2, 1, 1), generations: int = 3, width: int = 6,
         repeats: int = 1, tier: ModelTier = ModelTier.LARGE,
         min_margin: Optional[float] = None, resume_from=None,
         patience: int = 2, seed_spec: Optional[dict] = None,
         ledger_path: Optional[str] = None, unit_cost: Optional[dict] = None,
         ensemble_strategy: str = "synthesize",
         max_paired_p: Optional[float] = None,
         on_event: Optional[Callable[[dict], None]] = None) -> dict:
    """RSI を ``generations`` 世代回し、最終チャンピオンと全世代の台帳を返す。

    ``patience`` 世代続けて昇格が出なければ収束とみなして打ち切る(空回りで実モデルを焚かない)。
    ``sealed`` split は世代中一切参照せず、最後に**一度だけ**開けて種と最終チャンピオンを測る。

    ``min_margin`` を省くと **confirm 1 問ぶん** (``1/len(confirm)``) が下限になる。

    ``max_paired_p`` を与えると、平均差の床に加えて「同一 case での勝敗が偶然に見えない」
    ことも要求する(符号検定の片側 p)。既定は None = 無効。有効にすると門は大幅に厳しくなる
    ので、まず無効のまま台帳の ``paired_p`` を読み、自分の case 数で何が通るかを見てから使う。
    """
    # 観測履歴は走行に属する。同一プロセスで 2 回 grow すると前回の実体が
    # 混ざり、seed の直後に偽陽性を出す。測定を 1 回もしないうちに落とす。
    reset_served()
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

    # 刻印は走行につき 1 回。run W は seed 行に e7e5e63、recipe に ffdb5bf と書いた:
    # 走行中に commit が進み、終わりの行が書かれる時に HEAD を読み直したから。プロセスが
    # 動かしているのは最初に import した tree で、途中の commit はこの数字に何も寄与していない。
    stamp = code_stamp()
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
    # 同じ台帳へ続ける再開(--out と --resume が同じファイル)では前半の行を消さない。台帳は
    # この走行の唯一の証拠で、checkpoint を読んだ直後に truncate すると seed 行と前半の世代が
    # 消え、2 回目の再開では split の検査まで空振りする(seed 行が無いので)。落ちた時の
    # 書きかけ行の末尾に足すと再開行まで壊れるので、改行で区切ってから足す。
    # 実在しない --resume は上で「checkpoint が無い」と断っている(台帳に触る前)ので、ここの
    # ``resume_from`` は読めた台帳。
    continuing = bool(ledger and resume_from and ledger.exists()
                      and ledger.resolve() == Path(resume_from).resolve())
    if ledger:
        ledger.parent.mkdir(parents=True, exist_ok=True)
        if continuing:
            with ledger.open("rb") as fh:
                try:
                    fh.seek(-1, 2)
                    last = fh.read(1)
                except OSError:          # 空の台帳: 区切る行が無い
                    last = b"\n"
            if last != b"\n":
                with ledger.open("a", encoding="utf-8") as fh:
                    fh.write("\n")
        else:
            ledger.write_text("", encoding="utf-8")

    def emit(row: dict) -> None:
        if ledger:
            with ledger.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        if on_event:
            on_event(row)

    # 同一性が「連続して確かめられた」と言えるのは、それを証明できたときだけ。この変更より
    # 前に書かれた台帳には served が無く、黙って空から再開すると**検査を通ったこと**にされる。
    # 否定形(「食い違いが無い」)でなく肯定形(「同じだと確かめられた」)で持つ。
    resumed_blind = False
    if resume:
        champ_search = _restore(resume["champion_search"])
        champ_confirm = _restore(resume["champion_confirm"])
        archive.update(resume.get("archive") or {})
        start_gen = resume["gen"] + 1
        # 中断前に測っていた実体を復元する。復元は resumed イベントより**前**に済ませる
        # (イベントを見る側が、空の状態を再開直後の真値だと読まないように)。
        prior = resume.get("served")
        if prior:
            for dest, seen in prior.items():
                for one in seen:
                    note_served(dest, one)
        else:
            # 古い台帳からの再開(この機能より前の run O〜T には served が無い)。突き合わせる
            # 相手が無いので、この境界だけは検査できない。ただし「名乗らない backend だから
            # 空」なのか「古い形式だから空」なのかは、backend 名の許可リストで決めない
            # (名前で判定すると新しい backend が黙って検査対象から漏れる)。走ってみて実体を
            # 1 つでも観測したなら、名乗れる相手なのに突き合わせられなかった、と後で分かる。
            resumed_blind = True
        # 未検証の境界は**系統に属する**。blind な再開のあと落ちてもう一度再開すると、その
        # checkpoint には(以降の世代で観測した)served が載っているので、何もしなければ
        # 「突き合わせる相手が居た」ことになり、未検証だった事実がすすがれて消える。
        # 一度開いた穴は、その系統の台帳が続くかぎり引き継ぐ。
        if resume.get("identity_blind"):
            resumed_blind = True
    else:
        champ_search = measure(champion, splits["search"], tier, repeats, unit_cost, "champion")
        _guard_measurement(champ_search, "seed on the search split")
        champ_confirm = measure(champion, splits["confirm"], tier, repeats, unit_cost, "champion")
        _guard_measurement(champ_confirm, "seed on the confirm split")
        start_gen = 0
    # sealed の判定に渡す confirm 側の主張(``confirm_claim``)の材料。種の測定と最終形の測り直しを
    # 別々に貯める。再開した走行の種は**再開点のチャンピオン**(上で ``seed`` を champion から
    # 複製しているとおり)で、sealed も最後にそれと最終形を比べる。だから主張の基準もその点
    # (checkpoint から復元した champion の confirm)で、元の種まで遡らない。遡ると sealed が
    # 比べていない相手に対する主張になり、検定力の算術が合わなくなる。
    seed_scores: list = [champ_confirm.score]
    champ_scores: list = []          # 昇格後の測り直し(昇格時の値は入れない: 選ばれた値)
    champ_promo: Optional[float] = None
    emit({"event": "resumed" if resume else "seed", "hash": spec_hash(champion),
          "resumed_from_gen": resume["gen"] if resume else None,
          "search": _meas(champ_search), "confirm": _meas(champ_confirm),
          "classes": classes, "classes_unconfirmable": dropped,
          # 台帳をイベントログとして読む人が、復元後の実体を row だけで読めるようにする
          # (global state を覗きに行かないと分からない、という形にしない)。
          "served": served_map(), "identity_blind": resumed_blind,
          "code": stamp,
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
              "champion_search": _state(champ_search),
              "champion_confirm": _state(champ_confirm),
              "challenged": [], "settled": [], "archive": {}, "stale": 0,
              "served": served_map(), "identity_blind": resumed_blind})

    stale = resume["stale"] if resume else 0
    # confirm で決着がついた設計(勝っても負けても二度は問わない)
    challenged: set = set(resume["challenged"]) if resume else set()
    # search で帯を超えて負けた設計。challenged と分けるのは、この負けが**チャンピオンの search 点
    # に対する**負けだから: 新チャンピオンの search は旧より帯 1 つぶん低くてよいので、旧に帯超えで
    # 負けた設計が新には帯の内側ということがある。昇格で空にし、archive の点のまま再挑戦させる
    # (search を測り直さないので、再判定はコール 0 で済む)。古い checkpoint には無いキー。
    # challenged の方は昇格で空にしない: confirm の基準は昇格で**上がる一方**なので、旧に confirm で
    # 負けた設計は新にはなお負けている(search の基準が帯ぶん下がりうるのと逆向きの非対称)。
    settled: set = set(resume.get("settled", [])) if resume else set()
    for gen in range(start_gen, generations):
        t0 = time.time()
        # 伸びしろの尽きたクラスは変異させない。実測(run U seed): integration は 8/8 満点で
        # 伸びしろ 0.00 問なのに `tool:integration` が候補に出て、確実に無駄と分かっている
        # 測定に実モデルを焚いていた。取りうる最大の伸びが門の幅に満たないクラスは、
        # **どの変異でも昇格しえない**ので、除外は推定ではなく算術。
        #
        # チャンピオンの confirm 測定をここ(propose の前)でやるのは、飽和判定を
        # **復元した状態に依存させない**ため。毎世代どのみち測るものなので、順番を前に
        # 出すだけで測定回数は変わらず、再開の有無に関係なく新鮮な値で判定できる。
        # (checkpoint 自体は `_state()` で per_case を保つようになったが、search 側と違って
        #  confirm は毎世代測り直すので、そもそも復元値に頼る必要が無い。)
        champ_confirm_now = measure(champion, splits["confirm"], tier, repeats, unit_cost,
                                    "champion")
        _guard_measurement(champ_confirm_now, f"champion on confirm (gen {gen})")
        # 種のままなら種の測定、そうでなければ最終形候補の測り直し。足して戻した設計は種と
        # 同じ hash なので種側に戻る。
        (seed_scores if spec_hash(champion) == spec_hash(seed) else champ_scores).append(
            champ_confirm_now.score)
        # 差 δ の下限は「同じ config を測り直したときの揺れ」そのもの — 自分の揺れより
        # 小さい改善を採らないための実測アンカー。
        drift = abs(champ_confirm_now.score - champ_confirm.score)
        delta = max(margin_floor, drift)
        # 飽和の閾値も δ で見る。床だけで見ると、揺れが床より大きい世代に「床は越えられるが
        # δ は越えられない」クラスが候補に残り、通りようのない測定を焚くことになる。
        gate_cases = delta * len(splits["confirm"])
        # 挑戦権の帯は search の分解能(1 問ぶん)。search 側はチャンピオンを測り直さないので
        # drift は取れず、床だけになる。
        search_band = 1.0 / len(splits["search"])
        headroom = class_headroom(champ_confirm_now, splits["confirm"])
        # search 側は見ない。以前は「チャンピオンが search のそのクラスを取り切っていれば、
        # 変異は search を 1 点も上げられず挑戦権が取れない」として飽和に数えていたが、それは
        # 挑戦権が「厳密に上回ること」だった頃の算術。今の門 ①(search_gate)は同点を通すので、
        # 取り切ったクラスへの変異も confirm に伸びしろがある限り挑戦できる(run V では qa が
        # confirm に 1.25 問残しながら search 側の理由だけで 5 世代とも試されなかった)。
        saturated = sorted(c for c in classes if c in headroom and headroom[c] < gate_cases)
        if saturated:
            emit({"event": "saturated", "gen": gen,
                  "classes": {c: round(headroom.get(c, 0.0), 2) for c in saturated},
                  "gate_cases": round(gate_cases, 2),
                  "note": "no additive mutation on these classes can be promoted: the confirm "
                          "headroom is below the gate"})
        # 飽和したクラスも propose には渡す。削る変異は「悪くならないこと」しか要求しないので、
        # むしろ満点のクラスこそ「その構造は何も買っていない」と言える場所になる。
        # 診断は search と confirm の両方から(どちらも今のチャンピオンの測定)。
        symptoms: dict[str, int] = {}
        for m in (champ_search, champ_confirm_now):
            for c, n in (m.tool_no_code_by_class or {}).items():
                symptoms[c] = symptoms.get(c, 0) + n
        # 測定済み(archive)は幅の外で全部戻ってくる: search の点は設計に付くものでチャンピオンが
        # 替わっても動かないので、帯の内側に残った設計はコール 0 で毎世代挑戦者の候補になる。
        # 世代の初めに写しを取るのは、この世代に測った新顔と、前から archive に居た踏み石を
        # 台帳で区別するため(後で archive を見ると新顔も入っている)。
        archived_before = set(archive)
        cands = propose(champion, pool, classes, width=width, exclude=challenged | settled,
                        ensemble_strategy=ensemble_strategy, generation=gen,
                        additive_classes=[c for c in classes if c not in saturated],
                        allow_default=_default_swap_viable(champion, classes, headroom,
                                                           gate_cases),
                        no_code_by_class=symptoms, archived=archived_before)
        if not cands:
            # ここまでに champion を confirm で測っている。止まるからといって捨てると、
            # 最終結果も再開状態も「測る前の値」のまま残る。checkpoint も**実際に出す**
            # (load_checkpoint が読むのは checkpoint イベントだけなので、変数を更新した
            # だけでは再開したときに古い値へ戻る)。
            champ_confirm = champ_confirm_now
            emit({"event": "checkpoint", "gen": gen, "champion": champion,
                  "champion_search": _state(champ_search),
                  "champion_confirm": _state(champ_confirm),
                  "challenged": sorted(challenged), "settled": sorted(settled),
                  "archive": archive, "stale": stale,
                  "served": served_map(), "identity_blind": resumed_blind})
            emit({"event": "stop", "gen": gen, "reason": "no-new-candidates"})
            break

        scored = []
        measured = 0          # この世代に search を焚いた本数(台帳の new_candidates)
        for c in cands:
            h = spec_hash(c.spec)
            cached = archive.get(h)
            # 同じ設計を二度測らない(実モデルを無駄に焚かない)。search 側の測定揺れはここで
            # 固定されるが、上振れした候補も confirm で必ず測り直されるので昇格判定は汚れない
            # (代償は「上振れ candidate に挑戦権を 1 回使う」ことだけ)。揺れ自体を抑えたい場合は
            # repeats を上げる。
            if cached:
                m = _restore(cached["search"])
            else:
                m = measure(c.spec, splits["search"], tier, repeats, unit_cost, "candidate")
                _guard_measurement(m, f"candidate {c.label}")
                # archive も**状態**。実測(112問・5世代・width 4・候補20本)で台帳全体 75.5KB、
                # うち checkpoint が 84%。この規模なら per_case を持たせても問題にならないので、
                # 「重そう」で痩せさせない(痩せさせた結果が下の 4 例目)。
                # ここを表示用の形で入れると、キャッシュから昇格した
                # 候補の champ_search に per_case が無くなり、search 側の飽和判定がその走行の
                # 残り全部で黙って効かなくなる(per_case を落として機能が死ぬのはこれで 4 例目。
                # 台帳の行だけが痩せていればよく、決定に使うものは痩せさせない)。
                archive[h] = {"label": c.label, "kind": c.kind, "search": _state(m)}
                measured += 1
                emit({"event": "candidate", "gen": gen, "label": c.label, "kind": c.kind,
                      "hash": h, "search": _meas(m)})
            scored.append((c, m))


        # 追加と削減は別々に選ぶ。混ぜると、構造を剥がした候補が search 最高点の勝負に放り込まれ、
        # 「良くなったか」という**別の問い**の門で焼かれる(そして二度と挑戦できない)。
        additive = [(c, m) for c, m in scored if c.kind != "simplify"]
        row = {"event": "generation", "gen": gen, "champion_hash": spec_hash(champion),
               "champion_search": champ_search.score,
               "champion_confirm": champ_confirm_now.score,
               "drift": round(drift, 4), "delta": round(delta, 4),
               # どちらが敷居を決めたか。ノイズ律速なら repeats を上げる/測定を安定させる、
               # 分解能律速なら confirm の case を増やす —— 打つ手が正反対なので、台帳が
               # 「δ=0.05 だった」しか言わないと、次に何を変えればいいか読み取れない。
               "bound_by": "floor" if margin_floor >= drift else "drift",
               # candidates は挑戦者の候補に並んだ数、new_candidates がこの世代に search を
               # 測った数(= 焚いたコールの側。上の測定ループで実際に数える)。同点だらけの
               # 走行では前者だけ増えていく。
               "candidates": len(cands), "new_candidates": measured}
        # 分数のままだと大きさが読めない。床は「1 問ぶん」なので、**case 相当数**で出す。
        # ここが効くのは、床 1/n と効果 S/n の比が **S そのもの**で n に依存しない点:
        # 触っていないクラスの case を足しても比は 1 ミリも動かない。増やして意味があるのは
        # **変異が作用するクラスの case** だけで、判定は結局「丸 1 問ぶん動いたか」に尽きる。
        n_confirm = len(splits["confirm"])
        ok, reason = False, "no-additive-candidate"
        if additive:
            # 同点は構造の小さい方、それも同じならラベル順。
            #
            # ここは実測レイテンシで割っていたが、**それだと走行が決定的にならない**。壁時計は
            # 走るたびに違い、共有 GPU なら他人の負荷でも動くので、同点の候補が「たまたま空いて
            # いる時に測られた」だけで勝てた(determinism テストが 21 回に 1 回落ちて露見)。
            # 再現性はこの repo の売りそのものなので、判定に測定ゆらぎを一切入れない。
            # 構造の大きさは spec だけで決まり、追加コールを生んでいる当のものなので、
            # 「同点なら安い方」の意図もそのまま保つ(shrink 側の Occam と同じ向き)。
            challenger, chal_search = min(
                additive, key=lambda t: _challenger_key(*t, prescribed=_prescribed(t[0],
                                                                                    symptoms)))
            # search で band を超えて負けた設計は、**このチャンピオンの下では決着済み**:
            # 挑戦権 ① は search だけで決まり、チャンピオンの search はチャンピオンが替わるまで
            # 測り直されない(= 同じ hash の候補は archive の同じ点を返し続ける)。だから confirm を
            # 測っても門 ① で必ず落ちるし、来世代に提案し直しても同じ負けを繰り返すだけ。
            # 実測(run V gen1)では search で負けた候補に confirm 130 コール(9 分)を使ってから
            # 落としていた。決着済みの設計は settled に入れて枠も焚かない(昇格で空になる)。
            # 挑戦者 1 本だけでなく、この世代に search で測った**全候補**を見る: 次点以下も
            # 同じチャンピオンの同じ点に負けているので、同じ結論が今この場で出せる。
            # band の内側の候補は入れない(最高点が confirm で落ちた次の世代に、archive の点の
            # まま挑戦者になれる踏み石)。**削る候補も入れない**(削減の門は search を見ないので、
            # search で負けていても confirm で「悪くなっていない」を示せば通る)。
            for c, m in additive:
                if not search_gate(champ_search.score, m.score, search_band)[0]:
                    settled.add(spec_hash(c.spec))
            chal_hash = spec_hash(challenger.spec)
            row.update({"challenger": challenger.label, "kind": challenger.kind,
                        "challenger_hash": chal_hash,
                        "challenger_search": chal_search.score,
                        # 踏み石から上がった挑戦者か、この世代の新顔か。archive の点で挑戦する
                        # 設計はこの世代に search を測っていない、と台帳だけで読めるように。
                        "challenger_from": "archive" if chal_hash in archived_before else "new",
                        "search_band": round(search_band, 4)})
            s_ok, s_why = search_gate(champ_search.score, chal_search.score, search_band)
            if not s_ok:
                # 門 ① を通れないと分かっている候補の confirm は測らない。理由は promote_gate が
                # 返すのと同じ関数から取り、台帳の読み方を二通りにしない。
                ok, reason = False, s_why
            else:
                chal_confirm = measure(challenger.spec, splits["confirm"], tier, repeats,
                                       unit_cost, "challenger")
                _guard_measurement(chal_confirm, f"challenger {challenger.label}")
                w, l, t = paired_gain(champ_confirm_now, chal_confirm)
                ok, reason = promote_gate(champ_search.score, chal_search.score,
                                          champ_confirm_now.score, chal_confirm.score, delta,
                                          paired=(w, l, t), max_paired_p=max_paired_p,
                                          search_band=search_band)
                challenged.add(spec_hash(challenger.spec))
                row.update({"challenger_confirm": chal_confirm.score,
                            "gain_cases": round(
                                (chal_confirm.score - champ_confirm_now.score) * n_confirm, 2)})
                # 対応のある比較を**記録だけ**する(この時点では門にしない)。平均差の床
                # (1/n と drift)は「どの問題が split に入ったか」という支配的な誤差を見ていない
                # 疑いがあり、実測 3 走で confirm の伸びが sealed で 4〜6 倍しぼみ、小さい伸びでは
                # 符号ごと反転した。門にする前に「今までの昇格が何本引っかかるか」を先に測る。
                row.update({"paired_wins": w, "paired_losses": l, "paired_ties": t,
                            "paired_p": round(sign_test(w, l), 4)})
        row.update({"verdict": "promote" if ok else "reject", "reason": reason})
        if ok:
            champion, champ_search, champ_confirm = challenger.spec, chal_search, chal_confirm
            stale = 0
            settled = set()          # search の決着は旧チャンピオンの点に対するものだった
            champ_scores, champ_promo = [], chal_confirm.score
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
                _guard_measurement(cand_confirm, f"simplification {cand.label}")
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
                    champ_scores, champ_promo = [], cand_confirm.score
        row["champion_after"] = spec_hash(champion)
        row["structure_size"] = _structure_size(champion)
        row["elapsed_s"] = round(time.time() - t0, 2)     # 削減の測定まで含めた世代の実時間
        history.append(row)
        emit(row)
        # 世代ごとの checkpoint。実走は数時間かかり、実際に OOM で 43 問 x 4 候補を測り終えた
        # 直後に落ちて全部消えた。台帳に判定は残っていたのに再開できなかったのは、**再開に要る
        # 状態(チャンピオンの spec・決着済み・archive)を残していなかった**ため。
        emit({"event": "checkpoint", "gen": gen, "champion": champion,
              "champion_search": _state(champ_search),
              "champion_confirm": _state(champ_confirm),
              "challenged": sorted(challenged), "settled": sorted(settled),
              "archive": archive, "stale": stale,
              "served": served_map(), "identity_blind": resumed_blind})
        if stale >= patience:
            emit({"event": "stop", "gen": gen, "reason": f"no-promotion-for-{patience}-gens"})
            break

    # 封印を開けるのはここだけ。判定には一切使っていないので、この数字だけが偏っていない。
    # case が足りず封印を作れなかった場合は None を返す —— 判定に使った数字を「偏っていない
    # 数字」の欄に流用するのが、この設計でいちばんやってはいけないこと。
    if splits["sealed"]:
        sealed_seed = measure(seed, splits["sealed"], tier, repeats, unit_cost, "seed")
        _guard_measurement(sealed_seed, "seed on the sealed split")
        sealed_champ = (sealed_seed if spec_hash(seed) == spec_hash(champion) else
                        measure(champion, splits["sealed"], tier, repeats, unit_cost, "champion"))
        sealed = {"seed": _meas(sealed_seed), "champion": _meas(sealed_champ)}
    else:
        sealed = None
    # sealed が検定する仮説の confirm 側の対応物: 最終形が confirm で種よりどれだけ上に立つか。
    # 昇格時の認定を足し上げた量ではない(昇格の次の世代から champion は測り直され、点は種の
    # 側へ戻る。run V は認定 +1.10 問に対し測り直しの平均は −0.08 問)。推定の中身は
    # ``confirm_claim``。
    n_confirm = len(splits["confirm"])
    same_as_seed = spec_hash(champion) == spec_hash(seed)
    claim = confirm_claim(seed_scores, [] if same_as_seed else champ_scores, n_confirm,
                          None if same_as_seed else champ_promo)
    # 主張の根拠を note に残す。種のままで終わった走行では「champion の測定」は種の測定その
    # ものなので、別個に測った数のように読ませない(codex r3)。
    if same_as_seed:
        claim_basis = (f"the champion is the seed, measured {claim['seed_measurements']} "
                       f"times on confirm")
    elif claim["promotion_only"]:
        claim_basis = ("the champion measured once, at its promotion, against "
                       f"{claim['seed_measurements']} seed measurements")
    else:
        claim_basis = (f"means of {claim['champion_measurements']} champion and "
                       f"{claim['seed_measurements']} seed measurements")
    claim_cases = None if claim["cases"] is None else round(claim["cases"], 4)
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
        # 走行全体でどちらが律速だったか。分解能律速(floor)の走行で「もっと repeats を」と
        # 言っても何も変わらない。逆も同じ。
        "bound_by": {"floor": sum(1 for h in history if h.get("bound_by") == "floor"),
                     "drift": sum(1 for h in history if h.get("bound_by") == "drift")},
        "generations_run": len(history),
        "search": _meas(champ_search), "confirm": _meas(champ_confirm),
        # 最終形が confirm で種(再開した走行では再開点のチャンピオン)より何問上か、と
        # その推定の材料。sealed が検定した仮説の大きさそのもの。
        "confirm_claim": dict(claim, cases=claim_cases),
        "sealed": sealed,
        # 走行そのものの合否。confirm 上の昇格数は「何手通したか」であって「良くなったか」
        # ではない。封をした split に一度だけ言わせる。最終形の主張と昇格時の認定を渡して、
        # 「分からない」が検定力の無さなのか、転移しなかったのか、走行の中で蒸発したのかを
        # 言い分けさせる。
        "sealed_verdict": sealed_verdict(
            sealed, claim["cases"], n_confirm,
            promoted_gain_cases=promoted_gain_cases(history, n_confirm),
            claim_basis=claim_basis),
        # クラスごとの伸びしろ(問)。「どこに case を足すべきか」を一般論でなく実測で言う。
        # **変異できるクラスだけ**に絞る(confirm には居るが search に居ないクラスは
        # そもそも触れないので、そこに余地が無いと警告しても打てる手が無い)。
        # 丸めは 2 桁でなく 4 桁: 表示の丸めを判定に持ち込むと、0.995 問が「1 問ある」に
        # 化けて「1 問未満」の警告から漏れる。表示の丸めは呼び側の仕事。
        "headroom": {c: round(v, 4)
                     for c, v in sorted(class_headroom(champ_confirm,
                                                       splits["confirm"]).items())
                     if c in classes},
        "splits": {k: [c.case_id for c in v] for k, v in splits.items()},
        "params": {"suites": list(suites) if cases is None else "custom", "ratio": list(ratio),
                   "tier": tier.value, "repeats": repeats, "width": width,
                   "generations": generations, "patience": patience,
                   "min_margin": round(margin_floor, 4),
                   "min_margin_source": "auto(one confirm case)" if min_margin is None
                                        else "explicit",
                   "ensemble_strategy": ensemble_strategy,
                   "max_paired_p": max_paired_p, "code": stamp},
        # この走行が実際に測った相手。空なら同一性を名乗れない backend(echo 等)で、
        # 「確認できなかった」ことがそのまま読める(黙って保証したことにしない)。
        "served": served_map(),
        # 走行を通して同じ相手を測ったと**言い切れる**か。False は「違った」ではなく
        # 「確かめられなかった」で、その区別を残さないと未検証が検証済みに化ける。
        # 名乗れる相手を測ったのに、再開の境界で突き合わせる記録が無かった場合だけ False。
        # 名乗らない backend(echo 等)は「怪しい」のではなく「確かめようがない」ので、
        # 偽の警告を出さない。
        "identity_verified": not (resumed_blind and bool(served_map())),
        "history": history, "archive_size": len(archive),
        # 昇格した手の対応のある証拠。平均差だけ見ていると「confirm では伸びたが sealed では
        # しぼんだ/反転した」が説明できない。弱い証拠のまま通った手をここで名指しする。
        "promotion_evidence": [
            {"gen": h["gen"], "challenger": h.get("challenger"),
             "gain_cases": h.get("gain_cases"),
             "wins": h.get("paired_wins"), "losses": h.get("paired_losses"),
             "p": h.get("paired_p")}
            for h in history if h.get("verdict") == "promote"
        ],
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
    ]
    # 「Kimi-48B で測った」だけでは再現できない(量子化違いは別の相手)。応答が名乗った実体を
    # そのまま載せる。名乗らない backend なら、その旨を書いて保証を作らない。
    served = result.get("served") or {}
    if served:
        lines.append("Measured against (as reported by the server on every call):")
        lines += [f"- `{k}` → `{', '.join(v)}`" for k, v in sorted(served.items())]
        lines.append("")
    verdict = result.get("sealed_verdict") or sealed_verdict(result.get("sealed"))
    _label = {"improved": "IMPROVED", "regressed": "REGRESSED — do not adopt",
              "not-separable": "NOT SEPARABLE", "unsealed": "UNSEALED",
              "unjudgeable": "NOT JUDGED"}
    lines += [
        f"**Held-out verdict: {_label.get(verdict['verdict'], verdict['verdict'])}** "
        + (f"({verdict['delta_cases']:+} cases on the sealed split, which resolves "
           f"{verdict['band_cases']:g})" if verdict.get("delta_cases") is not None else ""),
        "",
        verdict["note"] + ".",
        "",
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
    ]
    # 主張の材料(無いのは古い result)。この二つの差が、sealed が検定した仮説の大きさそのもの。
    claim = result.get("confirm_claim")
    if isinstance(claim, dict) and claim.get("seed_mean") is not None:
        if claim.get("same_as_seed"):
            how = f"the champion is the seed: {claim.get('seed_measurements')} measurements"
        elif claim.get("promotion_only"):
            how = "champion measured once, at its promotion"
        else:
            how = (f"means over the run: {claim.get('seed_measurements')} seed / "
                   f"{claim.get('champion_measurements')} champion measurements")
        lines.append(f"| confirm score the sealed claim was made from ({how}) | "
                     f"{claim.get('seed_mean')} | {claim.get('champion_mean')} |")
    lines += [
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
