"""calib — verifier calibration: 門は「置いたか」でなく「どれだけ知っているか」で測る（v3a/T8）。

meshflow の verify は昇格と採択を決める門である。門の価値はその判定 V が正しさ C について
持つ情報量で上から抑えられる（Sagawa–Ueda 型: 測定で得た情報以上の仕事は引き出せない。
DPI: 門は情報を製造しない）。ここで測るのは:

- 混同行列（件数）: verdict ∈ {pass, fail} × correct ∈ {1, 0} の 2x2。
- I(V; C) bits: 物理側の目盛り。I≈0 の verify は「置いてあるだけの門」。
- **selection_ceiling（件数・達成可能上限）**: V だけを見て答えを採る/捨てるどんな方策も、
  正解率は MAP（各判定値で多数クラスに賭ける）を超えられない。Fano 型の下界より緩みが無く
  （達成可能なので tight）、件数のまま扱えるので verdict の規律（丸め値で判定しない）に合う。

**照合の向き**: verify を使う配管が selection_ceiling を超える正解率を報告したら、それは
発見ではなく**測定の誤り**（DPI 違反は結論でなくバグ検出器。docs/topology-design.md T8）。

full output が要るのでオフライン較正は不可（bench ledger は output_preview 200 字しか
持たない）。calibrate_verifier は suite を実走し、同じ出力に case checker（正誤）と
対象 verifier（判定）を同時適用する。
"""
from __future__ import annotations

import math
from typing import Callable, Optional

from .benchmark import score_output
from .models import ModelTier


def confusion_counts(pairs: list) -> dict:
    """(verdict_pass: bool, correct: bool) の列から 2x2 の件数。キーは固定名。

    tp = pass かつ正解 / fp = pass だが不正解 / fn = fail だが正解 / tn = fail かつ不正解。
    """
    tp = fp = fn = tn = 0
    for v, c in pairs:
        if v and c:
            tp += 1
        elif v and not c:
            fp += 1
        elif not v and c:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": tp + fp + fn + tn}


def _h(ps: list) -> float:
    return -sum(p * math.log2(p) for p in ps if p > 0)


def mutual_information_bits(cm: dict):
    """I(V; C) を bits で。n=0 は None（数を出すと診断に読まれる）。

    I = H(C) − H(C|V)。verify の「知っている量」の物理目盛りで、ラベルの向きに依存しない
    （pass/fail を逆に読む verifier でも I は同じ。だから**運用の上限は I でなく
    selection_ceiling で読む**: あちらは向きの取り違えも含めて達成可能な最大に立つ）。"""
    n = cm["n"]
    if n == 0:
        return None
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    h_c = _h([(tp + fn) / n, (fp + tn) / n])
    p_pass = (tp + fp) / n
    h_c_given_v = 0.0
    if tp + fp > 0:
        h_c_given_v += p_pass * _h([tp / (tp + fp), fp / (tp + fp)])
    if fn + tn > 0:
        h_c_given_v += (1 - p_pass) * _h([fn / (fn + tn), tn / (fn + tn)])
    return round(h_c - h_c_given_v, 4) + 0.0   # -0.0 を 0.0 へ（表示が診断に見える）


def selection_ceiling_counts(cm: dict) -> dict:
    """V だけで選ぶ方策の正解「件数」の達成可能上限（MAP）と、比較用の基準件数。

    ceiling_k = max(tp, fp) + max(fn, tn): 各判定値の bucket で多数クラスに賭ける方策が
    達成する。**V だけを根拠にケースごとの採否/賭けを決めるどんな方策**の正解件数も
    これを超えない（「pass だけ採る」通常の門は tp を数える方策で、tp <= max(tp,fp) <=
    ceiling_k と自明に下側にいる。つまり門の読み方によらず上限として安全）。
    base_k = max(正解総数, 不正解総数): V を見ずに常に同じ側へ賭ける最善（= 情報ゼロの床）。
    gate が V から引き出せる余地は高々 ceiling_k − base_k 件。"""
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    ceiling_k = max(tp, fp) + max(fn, tn)
    base_k = max(tp + fn, fp + tn)
    return {"ceiling_k": ceiling_k, "base_k": base_k, "headroom_k": ceiling_k - base_k,
            "n": cm["n"]}


def budget_check(observed_k: int, n: int, cm: dict) -> dict:
    """照合: 観測された「V 由来の選択の正解件数」が予算内か。超過は測定バグの検出。

    件数で比べる（丸め率で判定しない）。observed_k はこの較正と**同じ case 集合**上の
    件数であること（違う suite の数字を持ち込むと照合にならない）。"""
    if n != cm["n"]:
        raise ValueError(f"case counts differ: observed n={n} vs calibration n={cm['n']} "
                         "(same-suite comparison only)")
    sel = selection_ceiling_counts(cm)
    exceeds = observed_k > sel["ceiling_k"]
    return {
        "observed_k": observed_k, "ceiling_k": sel["ceiling_k"], "base_k": sel["base_k"],
        "within_budget": not exceeds,
        "verdict": ("MEASUREMENT BUG: observed selection beats what the verdict can know "
                    "(DPI). Check the pipeline's scoring, not the gate."
                    if exceeds else "within budget"),
    }


def calibrate_verifier(backend, verify: Callable[[str], object], suite: list,
                       tier: ModelTier = ModelTier.LARGE,
                       task_types: Optional[set] = None,
                       pass_score: float = 1.0) -> dict:
    """suite を実走し、同じ full output に case checker（正誤）と verify（判定）を当てる。

    判定の二値化は meshflow の門と同じ式に合わせる: _normalize_score(verify(out)) >=
    pass_score（既定 1.0 = MeshflowBackend の既定）。較正と実運用で閾値がずれると
    照合が照合でなくなる。verify の例外は fail（門が落ちたら通さない、の保守側。
    meshflow._score も同じ向き）。backend が例外を投げたケースは記録から除かず、
    **verify を呼ばずに** (verdict=fail, correct=False) で数える（出力が存在しないのに
    定数 True の verifier が fp を作る、という較正の汚染を防ぐ）。per_case の
    backend_error でどのケースがそれかを追える。"""
    from .meshflow import _normalize_score

    pairs = []
    per_case = []
    for case in suite:
        if task_types and case.task_type not in task_types:
            continue
        backend_error = False
        try:
            out = backend.complete(case.prompt, tier, task_type=case.task_type)
        except Exception:
            out, backend_error = "", True
        if backend_error:
            v, correct = False, False
        else:
            correct = score_output(case, out) >= 1.0
            try:
                v = _normalize_score(verify(out)) >= pass_score
            except Exception:
                v = False
        pairs.append((bool(v), bool(correct)))
        per_case.append({"case_id": case.case_id, "verdict_pass": bool(v),
                         "correct": bool(correct), "backend_error": backend_error})
    cm = confusion_counts(pairs)
    return {"confusion": cm, "i_bits": mutual_information_bits(cm),
            "selection": selection_ceiling_counts(cm), "per_case": per_case}
