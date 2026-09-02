"""Backend benchmark — the *external anchor* for vendor routing.

Measures each model backend on a small suite of task-class cases with **deterministic
checkers** (exact numbers, executed code, required-element presence — never an
LLM judge), then proposes a ``routing_table`` mapping each task class to the backend
that scored best. The proposal is written to a file for a human to ratify and adopt
via ``--config`` (same discipline as ``tehai calibrate``): routing fires on *measured*
performance, not a model's self-report.

Honesty notes:
- The *writing* class is scored by a coarse deterministic proxy (does it contain the
  required elements / shape), NOT a quality judgement. Treat its score as a floor.
- The *code* class **executes model-generated code** in-process to check it. Only run
  the live bench on inputs you trust (this is opt-in, like ``tehai run --sandbox``).
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
import re
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from ._json import _extract_json
from .logger import ExecutionLogger, LogRecord
from .models import ModelTier


# --------------------------------------------------------------------------- #
# Case model
# --------------------------------------------------------------------------- #
@dataclass
class BenchCase:
    case_id: str
    task_type: str                       # a TaskType value -> becomes a routing_table key
    prompt: str
    checker: Callable[[str], object]     # returns bool or float in [0,1]


# --------------------------------------------------------------------------- #
# Deterministic checkers
# --------------------------------------------------------------------------- #
def _strip_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines)
    return s


_FENCE_RE = re.compile(r"```[A-Za-z0-9_+-]*\n(.*?)```", re.DOTALL)


def _extract_code(s: str) -> str:
    """Pull runnable code out of a model reply: the longest fenced ``` block if any
    (so reasoning/verbose models that wrap code in prose are scored fairly), else the
    de-fenced text. Without this, a model that writes 'Here is the code:\\n```py...```'
    scores 0 purely on output format, not correctness."""
    blocks = _FENCE_RE.findall(s or "")
    if blocks:
        return max(blocks, key=len)
    return _strip_fences(s)


def _last_int(s: str) -> Optional[int]:
    nums = re.findall(r"-?\d+", (s or "").replace(",", ""))
    return int(nums[-1]) if nums else None


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


# 生成コードの実行は必ず子プロセスに隔離する。理由は cancellation の一点:
# in-process exec だと非停止コード 1 件で採点側(bench sweep / yoriai measure)全体が
# 止まり、SIGALRM は main thread 限定な上に生成コード側の except Exception に飲まれる
# (2026-08-31 実測: TinyLlama の wide-code-twosum 生成物はガード無しで 10 分待っても
# 返らなかった。gama-runs/kl-vs-beta の逸脱1)。真に打ち切れるのは kill できる境界だけ。
_CHECK_TIMEOUT_S = 10.0


# 子が返してよい 1 件あたりの戻り値サイズ。正解値は常に小さな plain data なので、
# これを超える戻り値はどのみち不正解。キャップが無いと生成コードが巨大 bytes を
# 返すだけで親側の受信/unpickle がメモリ爆発する(kill できる境界にしても、戻り値
# サイズ経由の穴は子側で塞ぐしかない)。
_RESULT_BYTES_CAP = 1_000_000

# sandbox が環境要因で立たないとき、全 code ケースの沈黙 0 点化と区別する為の一度きり警告
_SANDBOX_START_WARNED = False


class _PlainUnpickler(pickle.Unpickler):
    """find_class(GLOBAL/STACK_GLOBAL)を全拒否: plain data だけが復元できる。

    REDUCE opcode 自体は残るが、REDUCE が呼べる callable は untrusted バイト列では
    GLOBAL 経由でしか積めないため、拒否点はここ 1 箇所で実行経路が閉じる。

    親側の unpickle は生成コード由来のバイト列に触れる唯一の場所で、素の pickle だと
    __reduce__ を持つ戻り値が親プロセスで任意コードを実行できる(隔離の自己矛盾)。
    find_class を全拒否すると None/bool/int/float/str/bytes/list/tuple/dict/set しか
    復元できず、実行経路が消える。正解値は常に plain data なので採点意味論は保たれる
    (カスタム __eq__ で正解に化けるオブジェクトだけは err に落ちるが、それを得点に
    しない方が採点として正しい)。
    """

    def find_class(self, module, name):  # noqa: D102
        raise pickle.UnpicklingError(f"non-plain data refused: {module}.{name}")


def _plain_loads(blob: bytes):
    import io as _io
    return _PlainUnpickler(_io.BytesIO(blob)).load()


def _sandbox_apply(code: str, func_name: str, argss: list, conn) -> None:
    """Child-process side: exec untrusted code, apply fn to each args tuple.

    合否判定は意図的に親側へ残す(判定則は checker ごとに違う。ここへ複製すると
    _chk_longest_pal 型の特殊判定が二重化して drift する)。結果は 1 件ずつ
    ("cell", i, ...) で送る: 途中の 1 件が非停止でも、そこまでの正解は親に届いて
    いて部分点が生きる(summarize/routing は平均 score を使うので、まとめて最後に
    送る形だと hang 1 件が全テストの点を消す)。転送は send_bytes(生バイト)のみ:
    Queue だと транспорト層が親側で無条件に unpickle するため、乗っ取られた子が
    仕込んだオブジェクトをそのまま実行してしまう。バイト列なら親が制限付き
    unpickler で選んで復元できる。
    """
    def _send(msg) -> None:
        conn.send_bytes(pickle.dumps(msg))

    # 最初に「起動できた」印を送る: これより前に子が死ぬのは forkserver/spawn の
    # bootstrap 失敗(親の __main__ 再 import 不能)で、後に死ぬのは生成コード自身の
    # 選択(os._exit・segfault・hang)。親はこの印の有無で両者を区別し、前者だけ warn する。
    _send(("started", None, None))
    ns: dict = {}
    try:
        exec(compile(_extract_code(code), "<bench>", "exec"), ns)  # noqa: S102
    except BaseException:  # 生成コードは SystemExit/KeyboardInterrupt も投げうる
        _send(("exec_error", None, None))
        return
    fn = ns.get(func_name)
    if not callable(fn):
        _send(("no_func", None, None))
        return
    for i, args in enumerate(argss):
        try:
            v = fn(*args)
            # 渡せない/大きすぎる値は、この子ごと落ちて timeout に化ける前に
            # 「その 1 件の失敗」へ縮めておく(正解値は常に小さな plain data)。
            # pickle は 1 回だけ: サイズ判定も送信も同じ blob を使う。
            blob = pickle.dumps(("cell", i, ("ok", v)))
            if len(blob) > _RESULT_BYTES_CAP:
                raise ValueError("result too large")
            conn.send_bytes(blob)
        except BaseException:
            _send(("cell", i, ("err", None)))
    _send(("done", None, None))


def _apply_func_sandboxed(code: str, func_name: str, argss: list,
                          timeout: Optional[float] = None):
    """Run _sandbox_apply in a killable subprocess; None means timeout.

    待ち方は q.get(timeout) を先にする。p.join(timeout) を先にすると、戻り値が
    pipe バッファを超えたとき子が put でブロックしたまま「生きている」ので、
    正常な結果を timeout と誤判定して kill してしまう。
    """
    if timeout is None:
        # def 時でなく呼び出し時に読む: テスト・運用が module 定数の差し替えで
        # 上限を変えられるように(既定束縛だと差し替えが効かない)。
        timeout = _CHECK_TIMEOUT_S
    # 締切は子の起動より前から数える: forkserver/spawn の起動が詰まる環境でも
    # 「採点側を止めない」の上限に含める(start() 自体のブロックだけは残余リスク)。
    deadline = time.monotonic() + timeout
    try:
        # start method の選択には二つの相反するリスクがある:
        #  (1) 多スレッド親(yoriai server の measure)からの fork は、他スレッドが握った
        #      lock を子が継承して deadlock し、正解コードまで 0.0 に化ける。
        #  (2) forkserver/spawn は子で親の __main__ を再 import する。main が実ファイルで
        #      ないと(python -c / heredoc / REPL)FileNotFoundError で子が bootstrap 中に
        #      死に、これも正解コードが 0.0 になる(2026-08-31 実測: reference 40/40 が
        #      stdin 経由だと 32/40 に化けた)。しかも __main__ ガードの外の module 直下の
        #      コードが子で再実行される。
        # 既定は fork にする: 単スレッド親(CLI bench・解析スクリプト・pytest)では (1) は
        # 起きず、(2) の main 再 import も無いのであらゆる起動経路で一様に動く。多スレッド
        # 親のときだけ forkserver(→spawn)へ逃がす。そこは deadlock を避けるのが優先で、
        # かつ現実の多スレッド親(uvicorn/yoriai)は実 module から起動されるので (2) は起きない。
        # 見えない native thread で (1) を取りこぼしても、症状は deadlock=timeout kill で
        # 0.0 になるだけ(誤って正解を捏造する方向には壊れない)。
        if hasattr(os, "fork") and threading.active_count() <= 1:
            ctx = multiprocessing.get_context("fork")
        else:
            try:
                ctx = multiprocessing.get_context("forkserver")
            except ValueError:
                ctx = multiprocessing.get_context("spawn")
        rx, tx = ctx.Pipe(duplex=False)
        p = ctx.Process(target=_sandbox_apply, args=(code, func_name, argss, tx),
                        daemon=True)
        p.start()
    except Exception as e:
        # 環境要因(daemon プロセス配下・start method 不成立など)で sandbox 自体が
        # 立たない場合、例外を伝播させると 1 ケースでなく sweep 全体が落ちる。
        # 0.0 に折るが、全 code ケースが静かに 0 点化する事故と区別できるように
        # プロセスあたり 1 回だけ stderr へ出す(測定は赤くならず沈黙で死ぬ、への保険)。
        global _SANDBOX_START_WARNED
        if not _SANDBOX_START_WARNED:
            _SANDBOX_START_WARNED = True
            print(f"[gama.bench] sandbox unavailable, code checks score 0: {e}",
                  file=sys.stderr)
        return None
    tx.close()  # 親側の書き端を閉じる: 子の死で rx.poll が確実に EOF を見るため
    # 待ちは「done or 子の死 or 締切」の三条件で、届いた cell を逐次拾う。
    # 一本待ちだと、生成コードが os._exit() やハードクラッシュで何も送らずに
    # 死んだとき毎回満額 timeout を待つ。逆に子の死だけ見ると送信直後の正常終了と
    # 競合するので、死や EOF を見た後も読める分は飲み干してから確定する。
    results = [("err", None)] * len(argss)
    fatal = None  # exec_error / no_func: 部分点の対象外(関数が存在しない)
    started = False  # 子が exec 直前まで到達したか(bootstrap 失敗との区別)

    def _absorb() -> str:
        nonlocal fatal, started
        try:
            blob = rx.recv_bytes(_RESULT_BYTES_CAP + 65536)
        except EOFError:
            return "eof"
        except OSError:
            return "eof"  # 上限超過などで接続が読めなくなった: 以後は打ち切り
        try:
            kind, i, payload = _plain_loads(blob)
        except Exception:
            return "cont"  # 復元を拒否した cell は err のまま(実行はしない)
        if kind == "started":
            started = True
            return "cont"
        if kind == "cell":
            if isinstance(i, int) and 0 <= i < len(results):
                results[i] = payload
            return "cont"
        if kind == "done":
            return "done"
        fatal = kind
        return "done"

    state = "cont"
    hit_deadline = False      # 締切で打ち切ったか(= 正当な timeout。bootstrap 失敗と区別)
    while state == "cont":
        if rx.poll(0.1):
            state = _absorb()
            continue
        if not p.is_alive():
            while rx.poll(0.2) and state == "cont":
                state = _absorb()
            break
        if time.monotonic() >= deadline:
            hit_deadline = True
            break
    if p.is_alive():
        p.kill()
    p.join(1.0)
    rx.close()
    # started すら来ずに子が死んだ = forkserver/spawn の bootstrap 失敗(多スレッド親 +
    # 非ファイル __main__ は fork へ逃がせず、子が親の main 再 import で死ぬ)。これは全
    # code ケースの静かな 0 点化になるので沈黙させず、プロセスあたり 1 回だけ warn する。
    # started が来た後の死(生成コードの os._exit・segfault・hang)は user コードの正当な
    # 挙動なので warn しない(codex 指摘: 両者を印で区別)。timeout(hit_deadline)も対象外。
    # _SANDBOX_START_WARNED の global 宣言は関数冒頭の start 失敗ハンドラ側にある。
    if not started and fatal is None and state != "done" and not hit_deadline:
        if not _SANDBOX_START_WARNED:
            _SANDBOX_START_WARNED = True
            print("[gama.bench] sandbox child died during bootstrap; code checks score 0 "
                  f"(exitcode={p.exitcode}). Likely a multithreaded non-file __main__ "
                  "entrypoint; run the bench from a real module/script.", file=sys.stderr)
        return None
    if fatal is not None:
        return None
    # timeout / started 後の死 でも、届いた分の部分点は返す(未着は err のまま)。
    return results


def _check_func(code: str, func_name: str, cases) -> float:
    """Exec model code in a sandboxed subprocess and check a function against cases.

    SECURITY: executes model output (in a killable child; see _CHECK_TIMEOUT_S).
    Only used by the opt-in live benchmark. タイムアウト/即死は「結果が届かなかった
    テストだけ不合格」(hang より前に完走したテストの部分点は保持される。summarize/
    routing が平均 score を使うため)。exec 不能・関数不在は従来どおり 0.0。どの経路
    でもタイムアウトが score を上げる方向は無い。
    """
    argss = [args for args, _ in cases]
    results = _apply_func_sandboxed(code, func_name, argss)
    if results is None:
        return 0.0
    ok = 0
    for (st, v), (_, expected) in zip(results, cases):
        if st == "ok" and v == expected:
            ok += 1
    return ok / len(cases)


def _chk_palindrome(out: str) -> float:
    return _check_func(out, "is_palindrome", [
        (("A man, a plan, a canal: Panama",), True),
        (("hello",), False),
        (("racecar",), True),
    ])


def _chk_fizzbuzz(out: str) -> float:
    return _check_func(out, "fizzbuzz", [
        ((15,), "FizzBuzz"), ((3,), "Fizz"), ((5,), "Buzz"), ((7,), "7"),
    ])


def _chk_arith(out: str) -> float:
    return 1.0 if _last_int(out) == 396 else 0.0


def _chk_speed(out: str) -> float:
    return 1.0 if _last_int(out) == 80 else 0.0


def _chk_syllogism(out: str) -> float:
    return 1.0 if re.search(r"\byes\b", (out or "").lower()) else 0.0


def _chk_seq(out: str) -> float:
    return 1.0 if _last_int(out) == 42 else 0.0


def _chk_haiku(out: str) -> float:
    o = out or ""
    lines = [ln for ln in o.splitlines() if ln.strip()]
    return (int("rain" in o.lower()) + int(len(lines) >= 3)) / 2.0


def _chk_summary(out: str) -> float:
    o = (out or "").strip()
    sentences = [x for x in o.split(".") if x.strip()]
    one_sentence = o.endswith(".") and len(sentences) == 1
    has_terms = ("cat" in o.lower()) and ("mouse" in o.lower())
    return (int(one_sentence) + int(has_terms)) / 2.0


def _chk_tool_json(out: str) -> float:
    try:
        data = _extract_json(out)
    except Exception:
        return 0.0
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        return 0.0
    return (int(str(data.get("name")) == "tehai") + int(data.get("count") == 3)) / 2.0


def _chk_tool_call(out: str) -> float:
    norm = _norm_ws(out).replace("add(8,34)", "add(8, 34)")
    return 1.0 if "add(8, 34)" in norm else 0.0


# --------------------------------------------------------------------------- #
# Default suite — 5 task classes x 2 cases. task_type values are real TaskType
# members so the proposed routing_table keys slot straight into config.
# --------------------------------------------------------------------------- #
DEFAULT_SUITE: list[BenchCase] = [
    BenchCase("code-palindrome", "code_implementation",
              "Write a Python function `is_palindrome(s)` that returns True iff s reads "
              "the same forwards and backwards, ignoring case and non-alphanumeric "
              "characters. Return ONLY the function definition, no prose.", _chk_palindrome),
    BenchCase("code-fizzbuzz", "code_implementation",
              "Write a Python function `fizzbuzz(n)` returning 'Fizz' if n is divisible by "
              "3, 'Buzz' if divisible by 5, 'FizzBuzz' if divisible by both, else str(n). "
              "Return ONLY the function definition.", _chk_fizzbuzz),
    BenchCase("math-arith", "qa",
              "Compute 17 * 23 + 5. Reply with ONLY the integer.", _chk_arith),
    BenchCase("math-speed", "qa",
              "A train travels 60 km in 45 minutes. What is its average speed in km/h? "
              "Reply with ONLY the integer.", _chk_speed),
    BenchCase("reason-syllogism", "research",
              "All Bloops are Razzies. All Razzies are Lazzies. Are all Bloops necessarily "
              "Lazzies? Answer ONLY 'yes' or 'no'.", _chk_syllogism),
    BenchCase("reason-seq", "research",
              "What number continues this sequence: 2, 6, 12, 20, 30, ? Reply with ONLY "
              "the integer.", _chk_seq),
    BenchCase("write-haiku", "content",
              "Write a haiku about autumn rain. It must mention 'rain' and be exactly "
              "three lines. Output only the haiku.", _chk_haiku),
    BenchCase("write-summary", "content",
              "Summarize the following in ONE sentence ending with a period: 'The cat sat "
              "on the mat, then chased a mouse across the kitchen.' Output only the "
              "sentence.", _chk_summary),
    BenchCase("tool-json", "integration",
              'Return a JSON object with keys "name" (string "tehai") and "count" '
              "(integer 3). Output ONLY the JSON.", _chk_tool_json),
    BenchCase("tool-call", "integration",
              "You can call the function add(a, b). To add 8 and 34, output ONLY the call "
              "exactly as: add(8, 34)", _chk_tool_call),
]


# --------------------------------------------------------------------------- #
# Hard suite — break the ceiling effect of DEFAULT_SUITE (everyone hits 1.0, so
# it can't tell backends apart). Ported from tehai's experiments/hard_bench.py:
# executed algorithms, modular arithmetic, non-obvious sequences, acrostic/format
# constraints, strict nested JSON — cases hard enough to *discriminate* backends,
# so gama's thesis ("a structured combination of small models ties a big one")
# becomes measurable rather than a ceiling. Same deterministic-checker discipline;
# task_type values stay within the 5 real classes so the proposed routing_table
# keys slot straight into a config.
# --------------------------------------------------------------------------- #
def _chk_longest_pal(out: str) -> float:
    # 実行は _apply_func_sandboxed に寄せ、判定(回文性・部分文字列・長さ)だけを
    # ここに残す。等値比較でない checker なので _check_func には畳めない。
    tests = {"babad": 3, "cbbd": 2, "a": 1, "forgeeksskeegfor": 10, "racecarx": 7}
    items = list(tests.items())
    results = _apply_func_sandboxed(out, "longest_palindrome",
                                    [(s,) for s, _ in items])
    if results is None:
        return 0.0
    ok = 0
    for (st, r), (s, length) in zip(results, items):
        if (st == "ok" and isinstance(r, str) and r in s and r == r[::-1]
                and len(r) == length):
            ok += 1
    return ok / len(tests)


def _chk_merge(out: str) -> float:
    return _check_func(out, "merge_intervals", [
        (([[1, 3], [2, 6], [8, 10], [15, 18]],), [[1, 6], [8, 10], [15, 18]]),
        (([[1, 4], [4, 5]],), [[1, 5]]),
        (([[1, 4], [0, 4]],), [[0, 4]]),
        (([[1, 4], [2, 3]],), [[1, 4]]),
    ])


def _chk_mult(out: str) -> float:
    return 1.0 if _last_int(out) == 1059 else 0.0          # 37*43 - 28*19


def _chk_modexp(out: str) -> float:
    return 1.0 if _last_int(out) == 9 else 0.0             # 7^100 mod 13


def _chk_weekday(out: str) -> float:
    return 1.0 if re.search(r"\bfriday\b", (out or "").lower()) else 0.0


def _chk_lookandsay(out: str) -> float:
    return 1.0 if _last_int(out) == 312211 else 0.0


def _chk_acrostic(out: str) -> float:
    lines = [ln.strip() for ln in (out or "").splitlines() if ln.strip()]
    if len(lines) != 4:
        return 0.0
    firsts = "".join(ln[0].upper() for ln in lines if ln)
    return (int(firsts == "CODE") + int(all("machine" in ln.lower() for ln in lines))) / 2.0


def _chk_primelist(out: str) -> float:
    return 1.0 if _norm_ws(out).replace(" ", "") == "53,59,61,67" else 0.0


def _chk_json_nested(out: str) -> float:
    try:
        d = _extract_json(out)
    except Exception:
        return 0.0
    if isinstance(d, list) and d:
        d = d[0]
    if not isinstance(d, dict):
        return 0.0
    a = d.get("args") or {}
    return 1.0 if (d.get("tool") == "search" and isinstance(a, dict)
                   and a.get("query") == "gama" and a.get("limit") == 5
                   and d.get("tags") == ["a", "b", "c"]) else 0.0


def _chk_json_squares(out: str) -> float:
    try:
        d = _extract_json(out)
    except Exception:
        return 0.0
    return 1.0 if d == [1, 4, 9, 16, 25] else 0.0


HARD_SUITE: list[BenchCase] = [
    BenchCase("hard-code-longpal", "code_implementation",
              "Write a Python function `longest_palindrome(s)` returning the longest "
              "contiguous palindromic substring of s (return any one if there is a tie). "
              "Return ONLY the function definition, no prose.", _chk_longest_pal),
    BenchCase("hard-code-merge", "code_implementation",
              "Write a Python function `merge_intervals(intervals)` that merges all "
              "overlapping intervals and returns them sorted ascending by start, as a list "
              "of [start, end] lists. Return ONLY the function definition, no prose.", _chk_merge),
    BenchCase("hard-math-mult", "qa",
              "Compute 37 * 43 - 28 * 19. Reply with ONLY the integer.", _chk_mult),
    BenchCase("hard-math-modexp", "qa",
              "What is the remainder when 7^100 is divided by 13? Reply with ONLY the "
              "integer.", _chk_modexp),
    BenchCase("hard-reason-weekday", "research",
              "The day before two days after the day before tomorrow is Saturday. What day "
              "is it today? Answer with ONLY the weekday name.", _chk_weekday),
    BenchCase("hard-reason-lookandsay", "research",
              "Give the next term of this sequence: 1, 11, 21, 1211, 111221, ? Reply with "
              "ONLY the integer.", _chk_lookandsay),
    BenchCase("hard-write-acrostic", "content",
              "Write exactly 4 lines. The first letters of the four lines must spell C, O, "
              "D, E in that order. Every line must contain the word 'machine'. Output ONLY "
              "the 4 lines.", _chk_acrostic),
    BenchCase("hard-write-primelist", "content",
              "Output every prime number strictly between 50 and 70 as a comma-separated, "
              "ascending list with no spaces and no other text.", _chk_primelist),
    BenchCase("hard-tool-json-nested", "integration",
              'Output ONLY a JSON object with: "tool"="search"; "args" an object with '
              '"query"="gama" and "limit"=5; "tags"=["a","b","c"]. No prose.', _chk_json_nested),
    BenchCase("hard-tool-json-squares", "integration",
              "Output ONLY a JSON array of the squares of the integers 1 through 5, as "
              "integers.", _chk_json_squares),
]


# --------------------------------------------------------------------------- #
# Brutal suite — a frontier-split probe: even strong models miss some of these.
# Ported from tehai's hard_bench.py BRUTAL_SUITE. Use when `hard` no longer
# discriminates your strongest backends.
# --------------------------------------------------------------------------- #
def _chk_trailzeros(out: str) -> float:
    return 1.0 if _last_int(out) == 24 else 0.0           # trailing zeros in 100!


def _chk_powmod(out: str) -> float:
    return 1.0 if _last_int(out) == 624 else 0.0          # 2^50 mod 1000


def _chk_distinct3(out: str) -> float:
    return 1.0 if _last_int(out) == 648 else 0.0          # 3-digit all-distinct count


def _chk_knights(out: str) -> float:
    toks = re.findall(r"[a-z]+", (out or "").lower())
    return 1.0 if toks and toks[-1] == "knight" else 0.0


def _chk_palindrome_sentence(out: str) -> float:
    t = re.sub(r"[^a-z0-9]", "", (out or "").lower())
    return 1.0 if len(t) >= 11 and t == t[::-1] else 0.0


def _chk_p_alliteration(out: str) -> float:
    words = re.findall(r"[A-Za-z']+", out or "")
    return 1.0 if len(words) == 8 and all(w[0].lower() == "p" for w in words) else 0.0


BRUTAL_SUITE: list[BenchCase] = [
    BenchCase("brutal-qa-trailzeros", "qa",
              "How many trailing zeros does 100! (100 factorial) have? Reply with ONLY the "
              "integer.", _chk_trailzeros),
    BenchCase("brutal-qa-powmod", "qa",
              "Compute 2^50 mod 1000. Reply with ONLY the integer.", _chk_powmod),
    BenchCase("brutal-research-knights", "research",
              "On an island each person is a knight (always tells the truth) or a knave "
              "(always lies). A says 'B is a knave'. B says 'A and I are the same type'. Is "
              "A a knight or a knave? Answer with ONLY the single word: knight or knave.",
              _chk_knights),
    BenchCase("brutal-research-distinct", "research",
              "How many 3-digit numbers (100-999) have all three digits distinct? Reply "
              "with ONLY the integer.", _chk_distinct3),
    BenchCase("brutal-content-palindrome", "content",
              "Write a sentence that is a palindrome (reads identically forwards and "
              "backwards when ignoring case, spaces and punctuation) and is at least 11 "
              "letters long. Output ONLY the sentence.", _chk_palindrome_sentence),
    BenchCase("brutal-content-alliteration", "content",
              "Write a sentence of exactly 8 words where every single word begins with the "
              "letter 'p'. Output ONLY the sentence.", _chk_p_alliteration),
]


# --------------------------------------------------------------------------- #
# Wide suite — same difficulty band as `hard`, but *broad*: 8 cases in each of the
# 5 classes. Difficulty was never the only thing missing. With 26 cases total the
# discriminating suites could rank two backends, but they could not carry a
# three-way split: `gama grow` ends up deciding promotions on 5 confirm cases,
# where one case is 0.2 of the score and a real 3-point improvement is invisible.
# Breadth is what buys *resolution*, so this suite exists to be split, not to be
# harder. The existing three suites are deliberately left untouched: the numbers in
# README / recipes were measured on them, and silently changing a suite would make
# every published score incomparable to the next one.
#
# Answers here were computed (not recalled) before being written down, and
# tests/test_suite_integrity.py re-derives every one of them from a reference
# solution — a case with a wrong expected answer silently penalises correct models,
# which is the one failure a benchmark must not have.
# --------------------------------------------------------------------------- #
def _eq_int(n: int) -> Callable[[str], float]:
    """Last integer in the reply equals n (the convention every suite here uses)."""
    return lambda out: 1.0 if _last_int(out) == n else 0.0


def _eq_norm(expected: str) -> Callable[[str], float]:
    """Reply equals expected after whitespace-normalising and lower-casing.

    Case and surrounding whitespace are free; *internal* spacing is not, so a prompt
    that demands "no spaces" is still scored on that.
    """
    want = _norm_ws(expected).lower()
    return lambda out: 1.0 if _norm_ws(out).lower() == want else 0.0


def _eq_nospace(expected: str) -> Callable[[str], float]:
    """Reply equals expected once ALL whitespace is deleted.

    For call-syntax answers, where `f(6,7)` and `f(6, 7)` are the same answer: the existing
    `tool-call` case makes the same allowance, and scoring the space would measure
    formatting pedantry rather than whether the model produced the call.
    """
    want = "".join(expected.split()).lower()
    return lambda out: 1.0 if "".join((out or "").split()).lower() == want else 0.0


def _last_digits(expected: str) -> Callable[[str], float]:
    """返答の**最後の数字列**が expected と文字列として一致する。

    基数表記のように「数値」でなく「桁の並び」が答えのとき用。`_eq_int` は int に直して
    比べるので `3,530` や `03530` まで通してしまい、base-7 の答えとしては別物を正解にする。
    最後を採るのは数値 case と同じ規約(散文で包んだ返答を救い、言い直しは後勝ち)。
    """
    want = expected.strip()
    def chk(out: str) -> float:
        runs = re.findall(r"\d+", out or "")
        return 1.0 if runs and runs[-1] == want else 0.0
    return chk


def _last_hyphenated(expected: str) -> Callable[[str], float]:
    """返答に現れる**最後のハイフン連結語**が expected と一致する。

    「答えだけ返せ」と書いた上で厳密一致にすると、正しく解いたうえで一言添えたモデルを
    0 点にする ── それは能力でなく指示追従を測っている(README の「公平な答え抽出」と同じ話)。
    かといって「どこかに含まれていれば可」にすると、`X ではなく Y です` のような**否定**まで
    通る。数値 case と同じ「最後が答え」規約に合わせると、包んだ返答は救えて言い直しは
    正しく後勝ちになる。
    """
    want = expected.strip().lower()
    pat = re.compile(r"[a-z]+(?:-[a-z]+)+")
    def chk(out: str) -> float:
        toks = pat.findall((out or "").lower())
        return 1.0 if toks and toks[-1] == want else 0.0
    return chk


def _eq_exact(expected: str) -> Callable[[str], float]:
    """Reply equals expected after stripping the ends only (inner whitespace is scored).

    For the machine-consumed formats: a prompt that says "one line, single space" is not
    satisfied by two lines, and a checker looser than its own prompt measures nothing in
    particular. Case is still free.
    """
    want = expected.strip().lower()
    return lambda out: 1.0 if (out or "").strip().lower() == want else 0.0


def _last_word(word: str) -> Callable[[str], float]:
    """The last alphabetic token equals word (one-word answers wrapped in prose)."""
    want = word.lower()

    def chk(out: str) -> float:
        toks = re.findall(r"[a-z]+", (out or "").lower())
        return 1.0 if toks and toks[-1] == want else 0.0
    return chk


def _func(name: str, cases) -> Callable[[str], float]:
    """Execute the reply as Python and check `name` against (args, expected) pairs."""
    return lambda out: _check_func(out, name, cases)


def _json_eq(expected) -> Callable[[str], float]:
    """The reply's JSON payload equals expected exactly.

    Deliberately no list-unwrapping (unlike ``_chk_tool_json``): some cases here ask for a
    JSON *array*, so unwrapping a single-element list would make "[{...}]" and "{...}"
    indistinguishable and silently pass the wrong shape.
    """
    def chk(out: str) -> float:
        try:
            d = _extract_json(out)
        except Exception:
            return 0.0
        return 1.0 if d == expected else 0.0
    return chk


def _chk_seven_words(out: str) -> float:
    o = (out or "").strip()
    words = re.findall(r"[A-Za-z']+", o)
    return 1.0 if (len(words) == 7 and o.endswith(".")) else 0.0


def _chk_acrostic_gama(out: str) -> float:
    lines = [ln.strip() for ln in (out or "").splitlines() if ln.strip()]
    if len(lines) != 4:
        return 0.0
    firsts = "".join(ln[0].upper() for ln in lines)
    return 1.0 if (firsts == "GAMA" and all("frog" in ln.lower() for ln in lines)) else 0.0


def _chk_lipogram(out: str) -> float:
    o = (out or "").strip()
    letters = re.findall(r"[A-Za-z]", o)
    return 1.0 if (len(letters) >= 25 and "e" not in o.lower()) else 0.0


def _chk_three_bullets(out: str) -> float:
    lines = [ln.rstrip() for ln in (out or "").splitlines() if ln.strip()]
    if len(lines) != 3:
        return 0.0
    return 1.0 if all(ln.startswith("- ") and "local" in ln.lower() for ln in lines) else 0.0


def _chk_s_alliteration(out: str) -> float:
    words = re.findall(r"[A-Za-z']+", out or "")
    return 1.0 if (len(words) == 6 and all(w[0].lower() == "s" for w in words)) else 0.0


def _chk_toad_thrice(out: str) -> float:
    return 1.0 if len(re.findall(r"\btoad\b", (out or "").lower())) == 3 else 0.0


WIDE_SUITE: list[BenchCase] = [
    # --- qa: one exact integer, no prose ---------------------------------- #
    BenchCase("wide-qa-pow2", "qa",
              "What is 2 to the power of 20? Reply with ONLY the integer.", _eq_int(1048576)),
    BenchCase("wide-qa-gcd", "qa",
              "What is the greatest common divisor of 1071 and 462? Reply with ONLY the "
              "integer.", _eq_int(21)),
    BenchCase("wide-qa-divisors", "qa",
              "How many positive divisors does 360 have? Reply with ONLY the integer.",
              _eq_int(24)),
    BenchCase("wide-qa-sum50", "qa",
              "What is the sum of all integers from 1 to 50 inclusive? Reply with ONLY the "
              "integer.", _eq_int(1275)),
    BenchCase("wide-qa-modpow", "qa",
              "What is 3^7 mod 11? Reply with ONLY the integer.", _eq_int(9)),
    BenchCase("wide-qa-binary", "qa",
              "Convert the binary number 110101 to decimal. Reply with ONLY the integer.",
              _eq_int(53)),
    BenchCase("wide-qa-seconds", "qa",
              "How many seconds are there in 3 hours and 45 minutes? Reply with ONLY the "
              "integer.", _eq_int(13500)),
    BenchCase("wide-qa-lcm", "qa",
              "What is the least common multiple of 14 and 21? Reply with ONLY the integer.",
              _eq_int(42)),

    # --- research: multi-step reasoning, single token answer --------------- #
    BenchCase("wide-research-choose", "research",
              "In how many ways can you choose 3 items from 8 distinct items when order does "
              "not matter? Reply with ONLY the integer.", _eq_int(56)),
    BenchCase("wide-research-seq", "research",
              "What number continues this sequence: 3, 8, 15, 24, 35, ? Reply with ONLY the "
              "integer.", _eq_int(48)),
    BenchCase("wide-research-digitsum", "research",
              "What is the sum of the digits of 2^10? Reply with ONLY the integer.", _eq_int(7)),
    BenchCase("wide-research-arrangements", "research",
              "How many distinct arrangements are there of the letters in the word 'level'? "
              "Reply with ONLY the integer.", _eq_int(30)),
    BenchCase("wide-research-ages", "research",
              "Mika is three times as old as her son. In 12 years she will be twice as old as "
              "he is then. How old is Mika now? Reply with ONLY the integer.", _eq_int(36)),
    BenchCase("wide-research-tallest", "research",
              "Ann is taller than Ben. Ceci is shorter than Ben. Dan is taller than Ann. Who "
              "is the shortest? Answer with ONLY the name.", _last_word("ceci")),
    BenchCase("wide-research-knave", "research",
              "On an island each person is a knight (always truthful) or a knave (always "
              "lying). A says 'B is a knight'. B says 'A and I are of different types'. What "
              "is B? Answer with ONLY the single word: knight or knave.", _last_word("knave")),
    BenchCase("wide-research-dice", "research",
              "Two fair six-sided dice are rolled. What is the probability that the sum is 7? "
              "Answer with ONLY a reduced fraction like a/b.", _eq_norm("1/6")),

    # --- code_implementation: the reply is executed ------------------------ #
    BenchCase("wide-code-twosum", "code_implementation",
              "Write a Python function `two_sum(nums, target)` returning the indices of the "
              "two numbers that add up to target, as a list [i, j] with i < j. Exactly one "
              "solution exists. Return ONLY the function definition.",
              _func("two_sum", [(([2, 7, 11, 15], 9), [0, 1]), (([3, 2, 4], 6), [1, 2]),
                                (([3, 3], 6), [0, 1])])),
    BenchCase("wide-code-anagram", "code_implementation",
              "Write a Python function `is_anagram(a, b)` returning True iff a and b are "
              "anagrams, ignoring case and spaces. Return ONLY the function definition.",
              _func("is_anagram", [(("Listen", "Silent"), True), (("hello", "world"), False),
                                   (("Dormitory", "dirty room"), True)])),
    BenchCase("wide-code-flatten", "code_implementation",
              "Write a Python function `flatten(x)` that flattens an arbitrarily nested list "
              "of integers into a flat list, preserving order. Return ONLY the function "
              "definition.",
              _func("flatten", [(([[1, [2, 3]], 4],), [1, 2, 3, 4]), (([],), []),
                                (([1, [2, [3, [4]]]],), [1, 2, 3, 4])])),
    BenchCase("wide-code-roman", "code_implementation",
              "Write a Python function `roman_to_int(s)` converting a Roman numeral string to "
              "an integer. Return ONLY the function definition.",
              _func("roman_to_int", [(("MCMXCIV",), 1994), (("LVIII",), 58), (("IX",), 9),
                                     (("MMXXVI",), 2026)])),
    BenchCase("wide-code-bsearch", "code_implementation",
              "Write a Python function `binary_search(arr, target)` returning the index of "
              "target in the ascending list arr, or -1 if absent. Return ONLY the function "
              "definition.",
              _func("binary_search", [(([1, 3, 5, 7, 9, 11], 7), 3), (([1, 3, 5], 4), -1),
                                      (([2], 2), 0), (([], 1), -1)])),
    BenchCase("wide-code-rotate", "code_implementation",
              "Write a Python function `rotate(lst, k)` returning lst rotated k positions to "
              "the right (k may exceed len(lst)). Return ONLY the function definition.",
              _func("rotate", [(([1, 2, 3, 4, 5], 2), [4, 5, 1, 2, 3]),
                               (([1, 2, 3], 3), [1, 2, 3]), (([1, 2, 3], 4), [3, 1, 2])])),
    BenchCase("wide-code-balanced", "code_implementation",
              "Write a Python function `is_balanced(s)` returning True iff the brackets in s "
              "(only ()[]{}) are correctly balanced and nested. Return ONLY the function "
              "definition.",
              _func("is_balanced", [(("([]{})",), True), (("(]",), False), (("(((",), False),
                                    (("",), True)])),
    BenchCase("wide-code-topword", "code_implementation",
              "Write a Python function `most_common_word(text)` returning the most frequent "
              "whitespace-separated word in text (lowercased). Return ONLY the function "
              "definition.",
              _func("most_common_word", [(("the cat the dog the bird",), "the"),
                                         (("a b b",), "b")])),

    # --- content: format constraints, checked structurally ----------------- #
    BenchCase("wide-content-sevenwords", "content",
              "Write one sentence of exactly 7 words ending with a period. Output ONLY the "
              "sentence.", _chk_seven_words),
    BenchCase("wide-content-acrostic", "content",
              "Write exactly 4 lines whose first letters spell G, A, M, A in that order. "
              "Every line must contain the word 'frog'. Output ONLY the 4 lines.",
              _chk_acrostic_gama),
    BenchCase("wide-content-lipogram", "content",
              "Write a sentence of at least 25 letters that does not contain the letter 'e' "
              "at all. Output ONLY the sentence.", _chk_lipogram),
    BenchCase("wide-content-weekdays", "content",
              "Output the five weekday names from Monday to Friday as a comma-separated list "
              "with no spaces and no other text.",
              _eq_norm("Monday,Tuesday,Wednesday,Thursday,Friday")),
    BenchCase("wide-content-bullets", "content",
              "Output exactly 3 lines. Every line must start with '- ' and contain the word "
              "'local'. Output ONLY those lines.", _chk_three_bullets),
    BenchCase("wide-content-alliteration", "content",
              "Write a sentence of exactly 6 words where every word begins with the letter "
              "'s'. Output ONLY the sentence.", _chk_s_alliteration),
    BenchCase("wide-content-toad", "content",
              "Write a short paragraph that contains the word 'toad' exactly three times. "
              "Output ONLY the paragraph.", _chk_toad_thrice),
    BenchCase("wide-content-reverse", "content",
              "Write the word 'gamabunta' backwards. Output ONLY that reversed word.",
              _eq_norm("atnubamag")),

    # --- integration: structured output another program must consume ------- #
    BenchCase("wide-tool-json-user", "integration",
              'Output ONLY a JSON object with: "name"="gama"; "tags"=["a","b"]; "active"=true.',
              _json_eq({"name": "gama", "tags": ["a", "b"], "active": True})),
    BenchCase("wide-tool-json-list", "integration",
              'Output ONLY a JSON array of two objects: the first {"id":1}, the second '
              '{"id":2}.', _json_eq([{"id": 1}, {"id": 2}])),
    BenchCase("wide-tool-call", "integration",
              "You can call the function multiply(a, b). To multiply 6 and 7, output ONLY the "
              "call exactly as: multiply(6, 7)", _eq_nospace("multiply(6, 7)")),
    BenchCase("wide-tool-csv", "integration",
              "Output ONLY one CSV line with three fields in this order: gama, toad, 3 — "
              "comma-separated, no spaces, no header, no other text.", _eq_exact("gama,toad,3")),
    BenchCase("wide-tool-json-nested", "integration",
              'Output ONLY a JSON object with "model" an object of "name"="gemma" and '
              '"tier"="large", plus "retries"=2.',
              _json_eq({"model": {"name": "gemma", "tier": "large"}, "retries": 2})),
    BenchCase("wide-tool-json-squares", "integration",
              "Output ONLY a JSON array of the squares of the integers 6 through 10, as "
              "integers.", _json_eq([36, 49, 64, 81, 100])),
    BenchCase("wide-tool-json-null", "integration",
              'Output ONLY a JSON object with "ok"=true and "error"=null.',
              _json_eq({"ok": True, "error": None})),
    BenchCase("wide-tool-kv", "integration",
              "Output ONLY the two key=value pairs separated by a single space, in this "
              "order: host=localhost port=11434", _eq_exact("host=localhost port=11434")),
]


# --------------------------------------------------------------------------- #
# Graded suite — cases whose score is a FRACTION by construction.
#
# The other suites are effectively pass/fail: a checker returns 1.0 or 0.0, so a
# backend's score can only move in whole cases. Two things need a finer signal:
#
#   * `gama grow`'s promotion floor is "one confirm case". Whether that floor
#     differs from a hand-set constant can only show up when a real improvement
#     lands *between* half a case and a whole one, which pass/fail cases cannot
#     produce (replaying 14 real generations, no decision ever landed there).
#   * an inference-time search (`abmcts`) steers on the verify score as a reward.
#     With a binary reward there is no gradient for "this draft is closer" — its
#     "go deeper" branch has nothing to refine toward, and the search degenerates
#     into best-of-N.
#
# So every case here carries several INDEPENDENTLY checkable requirements and
# scores the fraction satisfied. Still deterministic, still no LLM judge — the
# grain is finer, not softer. Existing suites are left untouched: their numbers
# are quoted in README/recipes and a suite that changes underneath them makes
# every published score incomparable.
# --------------------------------------------------------------------------- #
def _fraction(checks) -> float:
    """Fraction of independently verified requirements that hold."""
    checks = list(checks)
    return sum(1.0 for c in checks if c) / len(checks) if checks else 0.0


def _alnum_tokens(s: str) -> set:
    return set(re.findall(r"[A-Za-z0-9]+", (s or "").lower()))


def _bare_list_of(n: int) -> Callable[[str], bool]:
    """Is the reply exactly a bare comma-separated list of n items (no prose around it)?"""
    def ok(out: str) -> bool:
        items = [t.strip() for t in (out or "").strip().split(",")]
        return (len(items) == n and all(items)
                and not any(re.search(r"\s", t) for t in items))
    return ok


def _found_all(expected) -> Callable[[str], float]:
    """Fraction of the expected values present, PLUS one requirement for the shape the
    prompt asks for (a bare comma-separated list of the right length).

    Value matching is deliberately position-free, so "2 of the 3 sub-answers are right"
    scores 2/3 instead of collapsing to pass/fail the moment the model annotates its list.
    But the prompts here do say "comma-separated, nothing else", and a checker that ignores
    a constraint its own prompt makes is measuring something nobody asked for — so the format
    is scored as one more independently checked requirement rather than silently dropped.
    """
    want = [str(e).lower() for e in expected]
    shaped = _bare_list_of(len(want))
    return lambda out: _fraction([w in _alnum_tokens(out) for w in want] + [shaped(out)])


def _chk_g_order(out: str) -> float:
    """Credit per correct POSITION, not per name present.

    The question is about the order, so membership scoring would give full marks to any
    permutation — the finer grain has to measure the thing the prompt actually asks for.
    """
    got = [t.strip().lower() for t in re.split(r"[,\n]", out or "") if t.strip()]
    want = ["ann", "ben", "cid"]
    return _fraction([got[i] == want[i] if i < len(got) else False for i in range(3)])


def _chk_g_b5(out: str) -> float:
    o = (out or "").strip()
    words = re.findall(r"[A-Za-z']+", o)
    return _fraction([len(words) == 5,
                      bool(words) and all(w[0].lower() == "b" for w in words),
                      o.endswith("!")])


def _chk_g_lines(out: str) -> float:
    lines = [ln.strip() for ln in (out or "").splitlines() if ln.strip()]
    body = "\n".join(lines)
    return _fraction([len(lines) == 3,
                      bool(lines) and all(len(ln) <= 20 for ln in lines),
                      body.lower().count("gama") == 1,
                      not re.search(r"\d", body)])


def _chk_g_three_words(out: str) -> float:
    o = (out or "").strip()
    low = o.lower()
    return _fraction([len(re.findall(r"\bfrog\b", low)) == 1,
                      len(re.findall(r"\bpond\b", low)) == 1,
                      len(re.findall(r"\bnight\b", low)) == 1,
                      # "a short paragraph" and "output ONLY" are constraints too: one block,
                      # no preamble. Left unscored, a wall of text with the three words in it
                      # would take full marks off a prompt it did not follow.
                      "\n\n" not in o and len(o) <= 300])


def _chk_g_sun_moon(out: str) -> float:
    o = (out or "").strip()
    sentences = [x.strip() for x in o.split(".") if x.strip()]
    return _fraction([len(sentences) == 2,
                      bool(sentences) and "sun" in sentences[0].lower(),
                      len(sentences) > 1 and "moon" in sentences[1].lower(),
                      len(o) < 100,
                      o.count(".") == 2 and o.endswith(".")])   # "each ending with a period"


def _g_json(out):
    try:
        d = _extract_json(out)
    except Exception:
        return None
    return d


def _chk_g_json4(out: str) -> float:
    d = _g_json(out)
    if not isinstance(d, dict):
        return 0.0
    return _fraction([d.get("name") == "gama", d.get("version") == 2,
                      d.get("ok") is True, d.get("tags") == ["a", "b"]])


def _chk_g_jsonlist3(out: str) -> float:
    d = _g_json(out)
    if not isinstance(d, list):
        return 0.0
    return _fraction([isinstance(d[i], dict) and d[i].get("id") == i + 1
                      if i < len(d) else False for i in range(3)])


def _chk_g_csv(out: str) -> float:
    rows = [ln.strip() for ln in (out or "").splitlines() if ln.strip()]
    want = [["gama", "toad", "3"], ["yoriai", "knot", "5"]]
    cells = []
    for r in range(2):
        got = [c.strip().lower() for c in rows[r].split(",")] if r < len(rows) else []
        cells += [got[c] == want[r][c] if c < len(got) else False for c in range(3)]
    # "two lines, no header, no spaces" is part of the ask; scoring only the cell values
    # would give full marks to a table with a header row and padding.
    shape = len(rows) == 2 and all(len(r.split(",")) == 3 for r in rows) and " " not in "".join(rows)
    return _fraction(cells + [shape])


def _chk_g_kv4(out: str) -> float:
    o = (out or "").strip()
    toks = _norm_ws(o).lower().split()
    pairs = dict(t.split("=", 1) for t in toks if "=" in t)
    return _fraction([pairs.get("host") == "localhost", pairs.get("port") == "11434",
                      pairs.get("tls") == "off", pairs.get("retries") == "2",
                      # "on one line, single spaces, ONLY": four tokens and nothing else.
                      len(toks) == 4 and "\n" not in o])


GRADED_SUITE: list[BenchCase] = [
    # --- qa: several sub-answers, credit per sub-answer --------------------- #
    BenchCase("g-qa-powers", "qa",
              "Give the values of 2^5, 3^4 and 5^3, in that order, as a comma-separated list "
              "of integers and nothing else.", _found_all([32, 81, 125])),
    BenchCase("g-qa-div60", "qa",
              "For the number 60 give three integers, comma-separated and nothing else: how "
              "many positive divisors it has, the sum of all its positive divisors, and its "
              "largest prime factor.", _found_all([12, 168, 5])),
    BenchCase("g-qa-bases", "qa",
              "Write 255 in binary, octal and hexadecimal, in that order, comma-separated, "
              "lowercase, with no 0b/0o/0x prefixes and no other text.",
              _found_all(["11111111", "377", "ff"])),
    BenchCase("g-qa-gcdlcm", "qa",
              "Give the greatest common divisor and the least common multiple of 18 and 24, "
              "in that order, comma-separated, nothing else.", _found_all([6, 72])),

    # --- research: multi-part reasoning ------------------------------------- #
    BenchCase("g-research-geometric", "research",
              "Continue this sequence with the next THREE terms: 2, 4, 8, 16, ... Give only "
              "the three integers, comma-separated.", _found_all([32, 64, 128])),
    BenchCase("g-research-banana", "research",
              "For the word 'banana' give three integers, comma-separated and nothing else: "
              "its length, how many DISTINCT letters it uses, and how many times 'n' appears.",
              # 期待値に重複を置かない: membership で測る以上、[6,3,3] だと 3 が一度でも出れば
              # 2 つ分の credit が付き、部分点が出なくなる(=この suite の存在意義が消える)。
              _found_all([6, 3, 2])),
    BenchCase("g-research-primes", "research",
              "List every prime number strictly between 20 and 40, comma-separated, ascending, "
              "nothing else.", _found_all([23, 29, 31, 37])),
    BenchCase("g-research-order", "research",
              "Ann is taller than Ben. Ben is taller than Cid. List the three names from "
              "tallest to shortest, comma-separated, nothing else.", _chk_g_order),

    # --- code: partial credit falls out of the per-test-case checker --------- #
    BenchCase("g-code-divide", "code_implementation",
              "Write a Python function `divide_safe(a, b)` returning a / b, or None when b is "
              "zero. Return ONLY the function definition.",
              _func("divide_safe", [((6, 3), 2.0), ((7, 2), 3.5), ((1, 0), None),
                                    ((0, 5), 0.0)])),
    BenchCase("g-code-chunk", "code_implementation",
              "Write a Python function `chunk(lst, n)` splitting lst into consecutive lists of "
              "length n, the last one shorter if it does not divide evenly. Return ONLY the "
              "function definition.",
              _func("chunk", [(([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]]),
                              (([], 3), []), (([1, 2, 3], 3), [[1, 2, 3]]),
                              (([1], 5), [[1]])])),
    BenchCase("g-code-title", "code_implementation",
              "Write a Python function `smart_title(s)` that capitalises each word except "
              "'of', 'the' and 'and', which stay lowercase unless they are the first word. "
              "Return ONLY the function definition.",
              _func("smart_title", [(("the lord of the rings",), "The Lord of the Rings"),
                                    (("a tale of two cities",), "A Tale of Two Cities"),
                                    (("war and peace",), "War and Peace"),
                                    (("the end",), "The End")])),
    BenchCase("g-code-range", "code_implementation",
              "Write a Python function `parse_range(s)` where '1-5' -> [1,2,3,4,5], a bare "
              "'7' -> [7], and a descending range like '5-1' -> []. Return ONLY the function "
              "definition.",
              _func("parse_range", [(("1-5",), [1, 2, 3, 4, 5]), (("7",), [7]),
                                    (("3-3",), [3]), (("5-1",), [])])),

    # --- content: several constraints, credit per constraint ----------------- #
    BenchCase("g-content-b5", "content",
              "Write a line of exactly 5 words where every word begins with the letter 'b' "
              "and the line ends with an exclamation mark. Output ONLY that line.", _chk_g_b5),
    BenchCase("g-content-lines", "content",
              "Output exactly 3 lines. Every line must be 20 characters or shorter, the word "
              "'gama' must appear exactly once across all three, and no digit may appear "
              "anywhere. Output ONLY the lines.", _chk_g_lines),
    BenchCase("g-content-three-words", "content",
              "Write a short paragraph containing the words 'frog', 'pond' and 'night', each "
              "exactly once. Output ONLY the paragraph.", _chk_g_three_words),
    BenchCase("g-content-sun-moon", "content",
              "Write exactly two sentences, each ending with a period, under 100 characters "
              "in total: the first must mention the sun, the second must mention the moon. "
              "Output ONLY the sentences.", _chk_g_sun_moon),

    # --- integration: several fields, credit per field ----------------------- #
    BenchCase("g-tool-json4", "integration",
              'Output ONLY a JSON object with "name"="gama", "version"=2, "ok"=true and '
              '"tags"=["a","b"].', _chk_g_json4),
    BenchCase("g-tool-jsonlist3", "integration",
              'Output ONLY a JSON array of three objects, with "id" 1, 2 and 3 respectively.',
              _chk_g_jsonlist3),
    BenchCase("g-tool-csv2", "integration",
              "Output ONLY two CSV lines, no header, no spaces: the first is gama,toad,3 and "
              "the second is yoriai,knot,5.", _chk_g_csv),
    BenchCase("g-tool-kv4", "integration",
              "Output ONLY these four key=value pairs on one line, separated by single "
              "spaces: host=localhost port=11434 tls=off retries=2", _chk_g_kv4),
]


# --------------------------------------------------------------------------- #
# Steep suite — for models that have already saturated the others.
#
# Measured: `gemma4:e2b` scores 0.95 on the 86-case pool (wide+hard+brutal+
# default+graded), leaving one case of headroom on a 20-case sealed split. At
# that point `gama grow` cannot say anything — no mutation can clear a bar of
# one confirm case — and the ceiling, not the loop, is what stopped it.
#
# Two properties are deliberate. The cases are hard for a 7B *in its head*, and
# most of them are things a program does trivially: exact modular arithmetic,
# factorial tails, prime sums, CSV escaping, LRU eviction. A suite that is merely
# hard would only prove models are weak; this one leaves room for the STRUCTURE
# to matter, which is what the loop is searching for. Several score in fractions
# for the same reason `graded` does.
# --------------------------------------------------------------------------- #
def _chk_steep_acrostic(out: str) -> float:
    lines = [ln.strip() for ln in (out or "").splitlines() if ln.strip()]
    if not lines:
        return 0.0
    firsts = "".join(ln[0].upper() for ln in lines)
    return _fraction([len(lines) == 4, firsts == "GAMA",
                      all(len(re.findall(r"[A-Za-z']+", ln)) == 5 for ln in lines)])


def _chk_steep_every_third(out: str) -> float:
    words = re.findall(r"[A-Za-z']+", out or "")
    marked = words[2::3]
    return _fraction([len(words) == 12,
                      bool(marked) and all(w[0].lower() == "s" for w in marked),
                      len(marked) == 4])


def _chk_steep_ordered(out: str) -> float:
    o = (out or "").strip()
    low = o.lower()
    hits = [len(re.findall(r"\b%s\b" % w, low)) for w in ("toad", "pond", "moon")]
    pos = [low.find(w) for w in ("toad", "pond", "moon")]
    return _fraction([hits[0] == 1, hits[1] == 1, hits[2] == 1,
                      all(p >= 0 for p in pos) and pos == sorted(pos),
                      len(o) < 120])


def _chk_steep_lipogram3(out: str) -> float:
    lines = [ln.strip() for ln in (out or "").splitlines() if ln.strip()]
    return _fraction([len(lines) == 3,
                      "e" not in (out or "").lower(),
                      bool(lines) and all(len(re.findall(r"[A-Za-z]", ln)) >= 15
                                          for ln in lines)])


def _chk_steep_sorted_json(out: str) -> float:
    d = _g_json(out)
    if not isinstance(d, list):
        return 0.0
    names = [x.get("name") for x in d if isinstance(x, dict)]
    return _fraction([len(d) == 3, names == ["b", "c", "a"],
                      all(isinstance(x, dict) and {"name", "v"} <= set(x) for x in d)])


def _chk_steep_escaped_json(out: str) -> float:
    d = _g_json(out)
    if not isinstance(d, dict):
        return 0.0
    return _fraction([d.get("text") == 'she said "hi"', d.get("lines") == 2])


def _chk_steep_checksum(out: str) -> float:
    toks = _norm_ws(out).lower().split()
    pairs = dict(t.split("=", 1) for t in toks if "=" in t)
    return _fraction([pairs.get("a") == "3", pairs.get("b") == "4", pairs.get("sum") == "7",
                      len(toks) == 3])


STEEP_SUITE: list[BenchCase] = [
    # --- qa: exact computation a 7B cannot hold in its head, a program can ---- #
    BenchCase("steep-qa-modpow", "qa",
              "Compute 2^128 mod 1000003. Reply with ONLY the integer.", _eq_int(3026)),
    BenchCase("steep-qa-trailzeros", "qa",
              "How many trailing zeros does 250! have? Reply with ONLY the integer.",
              _eq_int(62)),
    BenchCase("steep-qa-primesum", "qa",
              "What is the sum of all prime numbers strictly below 500? Reply with ONLY the "
              "integer.", _eq_int(21536)),
    BenchCase("steep-qa-modpow2", "qa",
              "What are the last three digits of 17^19? Reply with ONLY the integer.",
              _eq_int(153)),

    # --- research: multi-step, one exact answer ------------------------------ #
    BenchCase("steep-research-anagrams", "research",
              "How many distinct arrangements are there of the letters in MISSISSIPPI? Reply "
              "with ONLY the integer.", _eq_int(34650)),
    BenchCase("steep-research-prob", "research",
              "A fair coin is flipped 5 times. What is the probability of exactly two heads? "
              "Answer with ONLY a reduced fraction a/b.", _eq_norm("5/16")),
    BenchCase("steep-research-lcm", "research",
              "What is the smallest positive integer divisible by every integer from 1 to 10? "
              "Reply with ONLY the integer.", _eq_int(2520)),
    BenchCase("steep-research-schedule", "research",
              "Four people A, B, C, D sit in seats 1 to 4. A sits somewhere before B. C sits in "
              "seat 4. A and D sit in adjacent seats. Who sits in seat 3? Answer with ONLY the "
              "letter.", _last_word("b")),

    # --- code: edge cases a confident wrong answer fails ---------------------- #
    BenchCase("steep-code-csv", "code_implementation",
              "Write a Python function `parse_csv_line(s)` that splits one CSV line into a list "
              "of fields, honouring double-quoted fields that may contain commas, and \"\" as "
              "an escaped quote inside a quoted field. Return ONLY the function definition.",
              _func("parse_csv_line", [(("a,b,c",), ["a", "b", "c"]),
                                       (('"a,b",c',), ["a,b", "c"]),
                                       (('x,"say ""hi""",z',), ["x", 'say "hi"', "z"]),
                                       (("",), [""])])),
    BenchCase("steep-code-roman", "code_implementation",
              "Write a Python function `int_to_roman(n)` converting 1..3999 to a Roman numeral. "
              "Return ONLY the function definition.",
              _func("int_to_roman", [((1994,), "MCMXCIV"), ((58,), "LVIII"), ((9,), "IX"),
                                     ((3999,), "MMMCMXCIX")])),
    BenchCase("steep-code-topo", "code_implementation",
              "Write a Python function `topo_sort(nodes, edges)` returning a topological order "
              "of nodes, breaking ties by choosing the smallest available node. edges is a list "
              "of (a, b) meaning a comes before b. Return ONLY the function definition.",
              _func("topo_sort", [((["a", "b", "c", "d"],
                                    [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]),
                                   ["a", "b", "c", "d"]),
                                  ((["x", "y"], []), ["x", "y"])])),
    BenchCase("steep-code-lru", "code_implementation",
              "Write a Python function `lru(capacity, ops)` simulating an LRU cache. ops is a "
              "list of tuples: ('put', key, value) or ('get', key). Return the list of results "
              "of the get operations, using -1 for a miss. Both put and get count as a use. "
              "Return ONLY the function definition.",
              _func("lru", [((2, [("put", "a", 1), ("put", "b", 2), ("get", "a"),
                                  ("put", "c", 3), ("get", "b"), ("get", "c")]), [1, -1, 3]),
                            ((1, [("put", "a", 1), ("put", "b", 2), ("get", "a")]), [-1])])),

    # --- content: several tight constraints at once, scored per constraint ---- #
    BenchCase("steep-content-acrostic", "content",
              "Write exactly 4 lines whose first letters spell G, A, M, A in that order, where "
              "every line has exactly 5 words. Output ONLY the 4 lines.", _chk_steep_acrostic),
    BenchCase("steep-content-every-third", "content",
              "Write a sentence of exactly 12 words in which the 3rd, 6th, 9th and 12th words "
              "each begin with the letter 's'. Output ONLY the sentence.", _chk_steep_every_third),
    BenchCase("steep-content-ordered", "content",
              "Write under 120 characters containing the words 'toad', 'pond' and 'moon' "
              "exactly once each and in that order. Output ONLY the text.", _chk_steep_ordered),
    BenchCase("steep-content-lipogram3", "content",
              "Write exactly 3 lines, each of at least 15 letters, none of which contains the "
              "letter 'e'. Output ONLY the lines.", _chk_steep_lipogram3),

    # --- integration: computed and escaped structure -------------------------- #
    BenchCase("steep-tool-computed", "integration",
              'Output ONLY a JSON object with "n"=6, "factorial" set to 6 factorial, and '
              '"squares" set to the squares of 1 through 6 as a list of integers.',
              _json_eq({"n": 6, "factorial": 720, "squares": [1, 4, 9, 16, 25, 36]})),
    BenchCase("steep-tool-sorted", "integration",
              'Sort these by "v" ascending and output ONLY the resulting JSON array: '
              '[{"name":"a","v":3},{"name":"b","v":1},{"name":"c","v":2}]',
              _chk_steep_sorted_json),
    BenchCase("steep-tool-escaped", "integration",
              'Output ONLY a JSON object with "lines"=2 and "text" set to the exact characters: '
              'she said "hi"', _chk_steep_escaped_json),
    BenchCase("steep-tool-checksum", "integration",
              "Output ONLY three key=value pairs on one line separated by single spaces: a=3, "
              "b=4, and sum set to a plus b.", _chk_steep_checksum),
]


# --------------------------------------------------------------------------- #
# QA-deep suite — deliberately class-IMBALANCED: 16 `qa` cases and nothing else.
#
# The promotion floor is one confirm case, and a gain of S/n clears it exactly
# when S >= 1, where S counts the case-equivalents a mutation actually improves.
# Since the floor 1/n and the gain S/n scale together, adding cases in classes a
# mutation never touches changes nothing at all — the only way to resolve a small
# per-class effect is more cases OF THAT CLASS. Measured: `tool:qa` on a 48B was
# worth 1.25 cases out of 40 confirm cases, i.e. barely over the bar, and the
# same lane measured 0.25 cases on a 20-case split and was refused.
#
# Half of these are exact computation, matching the existing `qa` cases. The
# other half deliberately are NOT: unambiguous short answers a program cannot
# help with. Filling the class with calculator problems would have built a suite
# that proves the tool lane right by construction; a routing decision is made per
# CLASS, so the class has to contain what the class really is.
# --------------------------------------------------------------------------- #
QADEEP_SUITE: list[BenchCase] = [
    # --- exact computation (a program does these trivially) ------------------ #
    BenchCase("qad-compute-modpow", "qa",
              "Compute 3^40 mod 100. Reply with ONLY the integer.", _eq_int(1)),
    BenchCase("qad-compute-digits", "qa",
              "How many decimal digits does 2^64 have? Reply with ONLY the integer.",
              _eq_int(20)),
    BenchCase("qad-compute-divisorsum", "qa",
              "What is the sum of all positive divisors of 496, including 496 itself? Reply "
              "with ONLY the integer.", _eq_int(992)),
    BenchCase("qad-compute-choose", "qa",
              "In how many ways can 5 items be chosen from 12 when order does not matter? "
              "Reply with ONLY the integer.", _eq_int(792)),
    BenchCase("qad-compute-count", "qa",
              "How many integers from 1 to 30 inclusive are divisible by 3 or by 5? Reply with "
              "ONLY the integer.", _eq_int(14)),
    BenchCase("qad-compute-negative", "qa",
              "Compute 7^3 - 5^4. Reply with ONLY the integer (it may be negative).",
              _eq_int(-282)),
    BenchCase("qad-compute-gcd", "qa",
              "What is the greatest common divisor of 3213 and 1386? Reply with ONLY the "
              "integer.", _eq_int(63)),
    BenchCase("qad-compute-base7", "qa",
              "Write the decimal number 255 in base 7. Reply with ONLY the digits, no prefix.",
              _eq_norm("513")),

    # --- short answers a program cannot supply ------------------------------- #
    BenchCase("qad-recall-gold", "qa",
              "What is the chemical symbol for gold? Reply with ONLY the symbol.",
              _eq_norm("Au")),
    BenchCase("qad-recall-planet", "qa",
              "Which planet is closest to the Sun? Reply with ONLY the planet name.",
              _last_word("mercury")),
    BenchCase("qad-recall-sides", "qa",
              "How many sides does a hexagon have? Reply with ONLY the integer.", _eq_int(6)),
    BenchCase("qad-recall-continent", "qa",
              "On which continent is the Sahara desert? Reply with ONLY the continent name.",
              _last_word("africa")),
    BenchCase("qad-recall-opposite", "qa",
              "What is the antonym of 'ascend'? Reply with ONLY the single word.",
              _last_word("descend")),
    BenchCase("qad-recall-water", "qa",
              "At what temperature in degrees Celsius does water boil at standard atmospheric "
              "pressure? Reply with ONLY the integer.", _eq_int(100)),
    BenchCase("qad-recall-language", "qa",
              "In which language does the word 'tsunami' originate? Reply with ONLY the "
              "language name.", _last_word("japanese")),
    BenchCase("qad-recall-shape", "qa",
              "What is the name of a three-dimensional shape with six identical square faces? "
              "Reply with ONLY the single word.", _last_word("cube")),
]


# --------------------------------------------------------------------------- #
# Research-deep suite — the same audit `qadeep` applied to `qa`, applied to
# `research`.
#
# `qa` turned out to be 16 for 16 exact computation, so a tool lane routed to the
# whole class looked like a win until non-computational cases were added and it
# fell to exactly zero. `research` had the same smell: of the 22 cases that
# existed, roughly three quarters are sequences, combinatorics, probability or
# integer arithmetic — all things a program does — and `tool:research` was the
# one mutation reproducing across splits.
#
# So: 16 more `research` cases, 8 computational (matching what is already there)
# and 8 that no program can help with — temporal ordering, referent resolution,
# odd-one-out, analogy, contradiction, deduction over categories, negation. Half
# and half for the same reason as `qadeep`: choosing new cases to suit the
# incumbent answer is how a benchmark ends up blessing it.
# --------------------------------------------------------------------------- #
RESEARCHDEEP_SUITE: list[BenchCase] = [
    # --- computational multi-step (a program does these) --------------------- #
    BenchCase("rd-compute-oddsum", "research",
              "What is the sum of the first 20 odd positive integers? Reply with ONLY the "
              "integer.", _eq_int(400)),
    BenchCase("rd-compute-palindromes", "research",
              "How many 4-digit numbers are palindromes? Reply with ONLY the integer.",
              _eq_int(90)),
    BenchCase("rd-compute-primeseq", "research",
              "What number continues this sequence: 2, 3, 5, 7, 11, 13, ? Reply with ONLY the "
              "integer.", _eq_int(17)),
    BenchCase("rd-compute-atleastone", "research",
              "A fair coin is flipped 3 times. What is the probability of at least one head? "
              "Answer with ONLY a reduced fraction a/b.", _eq_norm("7/8")),
    BenchCase("rd-compute-diagonals", "research",
              "How many diagonals does a regular octagon have? Reply with ONLY the integer.",
              _eq_int(20)),
    BenchCase("rd-compute-banana", "research",
              "How many distinct arrangements are there of the letters in BANANA? Reply with "
              "ONLY the integer.", _eq_int(60)),
    BenchCase("rd-compute-angles", "research",
              "What is the sum of the interior angles of a heptagon, in degrees? Reply with "
              "ONLY the integer.", _eq_int(900)),
    BenchCase("rd-compute-handshakes", "research",
              "Nine people each shake hands once with every other person. How many handshakes "
              "happen in total? Reply with ONLY the integer.", _eq_int(36)),

    # --- reasoning a program cannot do for you -------------------------------- #
    BenchCase("rd-reason-order", "research",
              "Ann left before Ben. Cid left after Ben. Who left second? Answer with ONLY the "
              "name.", _last_word("ben")),
    BenchCase("rd-reason-referent", "research",
              "The trophy did not fit into the suitcase because it was too small. What was too "
              "small? Answer with ONLY the single word.", _last_word("suitcase")),
    BenchCase("rd-reason-oddoneout", "research",
              "Which of these does not belong with the others: sparrow, eagle, penguin, "
              "salmon? Answer with ONLY the word.", _last_word("salmon")),
    BenchCase("rd-reason-analogy", "research",
              "Hand is to glove as foot is to what? Answer with ONLY the single word.",
              _last_word("sock")),
    BenchCase("rd-reason-contradiction", "research",
              "Consider: (1) All cats are mammals. (2) Some mammals are cats. (3) No cats are "
              "mammals. Which numbered statement contradicts statement 1? Reply with ONLY the "
              "number.", _eq_int(3)),
    BenchCase("rd-reason-category", "research",
              "Ken plays a string instrument that is neither the guitar nor the violin. His "
              "instrument is one of: cello, flute, drum, guitar. Which does he play? Answer "
              "with ONLY the instrument.", _last_word("cello")),
    BenchCase("rd-reason-negation", "research",
              "Every student who passed the exam had studied. Mika did not study. Did Mika "
              "pass? Answer with ONLY 'yes' or 'no'.", _last_word("no")),
    BenchCase("rd-reason-cause", "research",
              "The road flooded, so the bus was cancelled, so Rin walked to work. What was "
              "cancelled? Answer with ONLY the single word.", _last_word("bus")),
]


# Named suites — `gama bench --suite {default,hard,brutal,wide,graded,steep,qadeep,researchdeep}`. DEFAULT_SUITE
# stays the default so public behavior is unchanged; hard/brutal break the ceiling, wide
# adds the breadth a three-way split (`gama grow`) needs, and graded adds partial credit
# so a score can move by less than a whole case.

# --------------------------------------------------------------------------- #
# crux — 飽和した suite では構造の差が測れない、という実測への答え
# --------------------------------------------------------------------------- #
# run U の confirm 56 問の伸びしろは合計 7.48 問しかなく(46/56 が満点)、integration に至っては
# 8/8 で 0.00 だった。昇格には 1 問ぶんの価値が要るので、この上でどんな構造を試しても
# sealed が種と区別できない —— 6 走連続の NOT SEPARABLE の機構がこれ。
#
# だから「問題を増やす」のではなく**難しくする**。狙いは 48B が単体では落とすが、コードを
# 実行できれば取れる帯で、そこなら `tool` レーンや段階委譲の差が問数として現れる。
# 答えはすべて実際に計算して確かめてある(暗記でなく算出)。
CRUX_SUITE: list[BenchCase] = [
    # --- qa: 桁が頭に収まらない厳密計算 -------------------------------------- #
    BenchCase("crux-qa-modpow", "qa",
              "Compute 7^333 mod 999983. Reply with ONLY the integer.", _eq_int(286530)),
    BenchCase("crux-qa-digitsum", "qa",
              "What is the sum of the decimal digits of 2^1000? Reply with ONLY the integer.",
              _eq_int(1366)),
    BenchCase("crux-qa-nthprime", "qa",
              "What is the 10001st prime number? Reply with ONLY the integer.",
              _eq_int(104743)),
    BenchCase("crux-qa-multiples", "qa",
              "What is the sum of all natural numbers below 1000 that are multiples of 3 or 5? "
              "Reply with ONLY the integer.", _eq_int(233168)),

    # --- research: 多段だが答えは 1 つ ---------------------------------------- #
    BenchCase("crux-research-derange", "research",
              "In how many ways can 12 distinct letters be placed into 12 distinct envelopes so "
              "that NO letter goes into its own envelope? Reply with ONLY the integer.",
              _eq_int(176214841)),
    BenchCase("crux-research-collatz", "research",
              "Which starting number below 100000 produces the longest Collatz chain (n -> n/2 "
              "if even, 3n+1 if odd, ending at 1)? Reply with ONLY the starting number.",
              _eq_int(77031)),
    BenchCase("crux-research-josephus", "research",
              "41 people stand in a circle numbered 1 to 41. Counting around the circle every "
              "3rd person is removed, starting the count at person 1 (so person 3 goes first). "
              "Which position is the last one remaining? Reply with ONLY the integer.",
              _eq_int(31)),
    BenchCase("crux-research-amicable", "research",
              "The sum of the proper divisors of a is b, and of b is a, with a != b; such a and "
              "b are amicable. What is the sum of all amicable numbers under 10000? Reply with "
              "ONLY the integer.", _eq_int(31626)),

    # --- integration: 8/8 で伸びしろ 0 だったクラス。制約を重ねて厳密出力を要求 --- #
    BenchCase("crux-int-sundays", "integration",
              "How many Sundays fell on the first of the month between 1 Jan 1901 and 31 Dec "
              "2000 inclusive? Reply with ONLY the integer.", _eq_int(171)),
    BenchCase("crux-int-daycount", "integration",
              "How many days are there from 1900-01-01 to 2000-01-01 inclusive of the start and "
              "exclusive of the end? Reply with ONLY the integer.", _eq_int(36524)),
    BenchCase("crux-int-base", "integration",
              "Interpret the string zz as a base-36 number, then write that value in base 7. "
              "Reply with ONLY the base-7 digits.", _last_digits("3530")),
    BenchCase("crux-int-dijkstra", "integration",
              "A directed weighted graph has edges A->B 4, A->C 2, B->C 5, B->D 10, C->E 3, "
              "E->D 4, D->F 11. What is the length of the shortest path from A to F? Reply "
              "with ONLY the integer.", _eq_int(20)),
    BenchCase("crux-int-transform", "integration",
              "Take the sentence: The quick brown Fox jumps over the lazy Dog. Keep only the "
              "words longer than 3 characters, lowercase them, remove duplicates, sort them "
              "alphabetically, and join them with a single hyphen. Reply with ONLY the "
              "resulting string.", _last_hyphenated("brown-jumps-lazy-over-quick")),

    # --- code_implementation: 実装が要る(暗記では出ない) ---------------------- #
    BenchCase("crux-code-editdist", "code_implementation",
              "Write a Python function `edit_distance(a, b)` returning the Levenshtein distance "
              "between two strings. Return ONLY the function definition, no prose.",
              _func("edit_distance", [(("kitten", "sitting"), 3),
                                      (("intention", "execution"), 5),
                                      (("", "abc"), 3), (("same", "same"), 0)])),
    # 引数名と検査ケースの両方で「行と列の取り違え」を潰してある。正方形のケースだけだと
    # 転置した実装がそのまま通り、**正しい実装と区別できない case** になる(実測: 3x7 に
    # 障害物を置くと 60 対 116 で割れるが、10x10 では両方 121252 で割れない)。
    # 開始点を塞いだ場合の意味も、曖昧に残さず 1 ケースで固定する。
    BenchCase("crux-code-paths", "code_implementation",
              "Write a Python function `lattice_paths(rows, cols, blocked)` returning the "
              "number of monotone lattice paths on a grid of points from (0, 0) to "
              "(rows, cols), moving only down (row + 1) or right (col + 1). `blocked` is a set "
              "of (row, col) points that may not be entered, including possibly the start, in "
              "which case the answer is 0. Return ONLY the function definition, no prose.",
              _func("lattice_paths", [((10, 10, set()), 184756),
                                      ((10, 10, {(5, 5)}), 121252),
                                      ((3, 7, {(1, 3)}), 60),
                                      ((4, 4, {(0, 0)}), 0),
                                      ((1, 1, set()), 2)])),
    BenchCase("crux-code-josephus", "code_implementation",
              "Write a Python function `josephus(n, k)` returning the 1-indexed position of the "
              "survivor when every k-th person is removed from a circle of n people. Return "
              "ONLY the function definition, no prose.",
              _func("josephus", [((41, 3), 31), ((7, 3), 4), ((1, 5), 1)])),

    # --- content: すでに伸びしろのあるクラス。厳密な形式で上積み ---------------- #
    BenchCase("crux-content-triples", "content",
              "How many ordered triples of integers (a, b, c) satisfy a + b + c = 100 with "
              "0 < a < b < c? Reply with ONLY the integer.", _eq_int(784)),
]



# suite 1 行説明の**単一の出所**。CLI の help はここから作る。
# 手で書いた写しを help に置いていたせいで、qadeep / researchdeep / crux の 3 本が
# 登録済みなのに説明に出てこない状態が続いていた(選択肢のベタ書きと同じ壊れ方)。
SUITE_DOCS: dict[str, str] = {
    "default": "5 classes, may hit a ceiling",
    "hard": "harder variants of the default classes",
    "brutal": "discriminating, breaks the ceiling effect",
    "wide": "40 cases, 8 per class — breadth for splitting, same band as hard",
    "graded": "20 cases scored as a FRACTION of independently checked requirements, "
              "so a score can move by less than a whole case",
    "steep": "20 cases for models that already saturate the others — exact computation, "
             "escaping, eviction order",
    "qadeep": "16 qa cases, half computational and half not — added because `qa` had been "
              "16 for 16 arithmetic, which made a tool lane look better than it was",
    "researchdeep": "16 research cases, half of them things no program can supply — the same "
                    "audit applied to the other suspicious class",
    "crux": "17 cases aimed where a 48B fails alone but a program succeeds — for boxes where "
            "the other suites are saturated and can no longer separate structures",
}

SUITES: dict[str, list[BenchCase]] = {
    "default": DEFAULT_SUITE,
    "hard": HARD_SUITE,
    "brutal": BRUTAL_SUITE,
    "wide": WIDE_SUITE,
    "graded": GRADED_SUITE,
    "steep": STEEP_SUITE,
    "crux": CRUX_SUITE,
    "qadeep": QADEEP_SUITE,
    "researchdeep": RESEARCHDEEP_SUITE,
}


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def score_output(case: BenchCase, output: str) -> float:
    """Normalize a checker result to a float in [0, 1]; any error -> 0.0."""
    try:
        r = case.checker(output or "")
    except Exception:
        return 0.0
    if isinstance(r, bool):
        return 1.0 if r else 0.0
    try:
        return max(0.0, min(1.0, float(r)))
    except (TypeError, ValueError):
        return 0.0


def _limit_per_class(suite: list[BenchCase], limit: Optional[int]) -> list[BenchCase]:
    if not limit or limit <= 0:
        return list(suite)
    seen: dict[str, int] = {}
    out: list[BenchCase] = []
    for c in suite:
        if seen.get(c.task_type, 0) < limit:
            out.append(c)
            seen[c.task_type] = seen.get(c.task_type, 0) + 1
    return out


def _run_one(name: str, backend, case: BenchCase, tier: ModelTier, rep: int,
             unit_cost: dict) -> dict:
    error = None
    output = ""
    t0 = time.monotonic()
    try:
        if hasattr(backend, "last_usage"):
            backend.last_usage = None
        # Thread the case's external checker as `verify` so a MeshflowBackend gates its
        # cheap->expensive escalation on the SAME check the bench scores with (honest
        # measurement). Other backends accept **kwargs and ignore it.
        output = backend.complete(case.prompt, tier, task_type=case.task_type,
                                  verify=case.checker)
    except Exception as e:  # never let one backend abort the sweep
        error = f"{type(e).__name__}: {e}"[:200]
    latency = round(time.monotonic() - t0, 4)
    score = 0.0 if error else score_output(case, output)
    usage = getattr(backend, "last_usage", None) or {}
    tokens = usage.get("total_tokens") if usage else None
    uc = unit_cost.get(name, 0.0)
    cost = round((tokens / 1000.0) * uc, 6) if (tokens and uc) else None
    return {
        "backend": name, "task_type": case.task_type, "case_id": case.case_id, "rep": rep,
        "score": round(score, 4), "success": score >= 0.5, "latency_s": latency,
        "tokens": tokens, "cost": cost, "error": error, "output_preview": (output or "")[:200],
    }


def run_bench(backends: dict, suite: Optional[list[BenchCase]] = None,
              tier: ModelTier = ModelTier.LARGE, repeats: int = 1,
              limit_per_class: Optional[int] = None, unit_cost: Optional[dict] = None,
              logger: Optional[ExecutionLogger] = None, run_id: str = "bench") -> list[dict]:
    """Run every (backend x case x repeat) and return rich per-call records.

    If ``logger`` is given, each call is also appended as a LogRecord-compatible row
    so ``tehai evaluate <ledger>`` works on a bench ledger too.
    """
    suite = _limit_per_class(suite if suite is not None else DEFAULT_SUITE, limit_per_class)
    unit_cost = unit_cost or {}
    records: list[dict] = []
    # tool レーンの健康(コードが出たか・走ったか)は backends のグローバル計数にしか残らない。
    # 合計だけ持ち帰ると「どのクラスでコードが出ないか」が消え、prefill のような**クラス単位の
    # 処方**を出す側が診断を読めない。呼び出しは直列なので、1 call ごとの差分を記録に付ける。
    # (backend を並列に回す日が来たら、この差分は別の call の分を拾う。その時は計数を
    # グローバルでなく backend の返り値に載せ替えること。)
    from .backends import tool_stats
    for name, backend in backends.items():
        for case in suite:
            for rep in range(max(1, repeats)):
                before = tool_stats()
                rec = _run_one(name, backend, case, tier, rep, unit_cost)
                after = tool_stats()
                rec["tool"] = {k: after[k] - before[k] for k in after}
                records.append(rec)
                if logger is not None:
                    logger.log(_bench_logrecord(rec, run_id))
    return records


def _bench_logrecord(rec: dict, run_id: str) -> LogRecord:
    return LogRecord(
        run_id=run_id, task_id=rec["case_id"], task_type=rec["task_type"],
        selected_model=rec["backend"], review_score=rec["score"], actual_cost=rec["cost"],
        elapsed_seconds=rec["latency_s"], judge_decision="accept" if rec["success"] else "revise",
        failure_reason=rec["error"],
    )


# --------------------------------------------------------------------------- #
# Aggregation + proposal
# --------------------------------------------------------------------------- #
def _agg(rs: list[dict]) -> dict:
    n = len(rs)
    costs = [r["cost"] for r in rs if r["cost"] is not None]
    return {
        "n": n,
        "score": round(sum(r["score"] for r in rs) / n, 4),
        "success_rate": round(sum(1 for r in rs if r["success"]) / n, 4),
        "latency_s": round(sum(r["latency_s"] for r in rs) / n, 4),
        "cost": round(sum(costs) / len(costs), 6) if costs else None,
    }


def summarize(records: list[dict]) -> dict:
    """Aggregate to per-(class x backend) and per-backend-overall stats."""
    by_class: dict[str, dict[str, list]] = {}
    by_backend: dict[str, list] = {}
    for r in records:
        by_class.setdefault(r["task_type"], {}).setdefault(r["backend"], []).append(r)
        by_backend.setdefault(r["backend"], []).append(r)
    return {
        "by_class": {t: {b: _agg(rs) for b, rs in per.items()} for t, per in by_class.items()},
        "overall": {b: _agg(rs) for b, rs in by_backend.items()},
    }


def propose_routing_table(records: list[dict]) -> dict:
    """Pick the winning backend per class: highest score, then lower latency, then
    lower cost, then name (deterministic). Returns the table plus the full ranking."""
    summary = summarize(records)
    table: dict[str, str] = {}
    ranking: dict[str, list] = {}
    for task_type, per_backend in summary["by_class"].items():
        ranked = sorted(
            per_backend.items(),
            key=lambda kv: (-kv[1]["score"], kv[1]["latency_s"],
                            kv[1]["cost"] if kv[1]["cost"] is not None else 0.0, kv[0]),
        )
        table[task_type] = ranked[0][0]
        ranking[task_type] = [{"backend": b, **agg} for b, agg in ranked]
    return {"routing_table": table, "ranking": ranking, "overall": summary["overall"]}
