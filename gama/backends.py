"""Model backend seam.

The MVP planning pipeline is fully deterministic and needs **no** backend. This
module exists so that a live model can later be dropped in *without touching the
orchestrator* — the architect/reviewers accept an optional ``ModelBackend`` and
fall back to heuristics when it is ``None``.

The two live adapters mirror the pattern already proven in
``~/Projects/recurse/recurse/llm.py`` (ClaudeCliBackend / OllamaBackend). They
shell out lazily and are never invoked by the test suite.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from abc import ABC, abstractmethod

from .models import ModelTier


# --------------------------------------------------------------------------- #
# 供給されたモデルの同一性 — 「同じ相手を測り続けている」ことを証拠で言う
# --------------------------------------------------------------------------- #
# 実害(2026-08-30, run S): 走行の途中で共有 GPU の llama-server が別モデルに載せ替えられた。
# あの時は 503 が返ったので error_rate の床に引っかかったが、**200 を返しながら中身だけ
# 入れ替わる**なら何も鳴らない。世代 0 と世代 5 が別モデルの比較になっていても、台帳には
# 一続きの改善として残る。OpenAI 互換のレスポンスは毎回 `model` を返すので、追加の呼び出し
# ゼロで「要求した名前が同じ実体に解決され続けたか」を突き合わせられる。
#
# 集合の一致では見ない: 変異が新しいレーンを足せば新しいモデル名が正当に増える。見るのは
# **要求名 -> 供給名の対応が途中で変わったか**だけで、これなら偽陽性が出ない。
_SERVED: dict[str, set] = {}


def note_served(requested, served) -> None:
    """1 回の応答が名乗った実体を記録する。live backend から呼ばれる(測定の副産物)。"""
    if not requested or not served:
        return
    _SERVED.setdefault(str(requested), set()).add(str(served))


def served_conflicts() -> dict:
    """同じ要求先が複数の実体に解決された箇所。空なら「相手は変わっていない」。"""
    return {k: sorted(v) for k, v in _SERVED.items() if len(v) > 1}


def served_map() -> dict:
    """観測した 要求先 -> 実体 の対応。走行がどのファイルを測ったかの出所として台帳に残す。

    「Kimi-48B で測った」だけでは再現できない(量子化違いは別物)。応答が名乗った実体を
    そのまま控えることで、recipe の数字がどの重みのものかが後から言える。
    """
    return {k: sorted(v) for k, v in _SERVED.items()}


def reset_served() -> None:
    _SERVED.clear()


class ModelBackend(ABC):
    """Minimal completion interface. Tier lets an adapter pick a concrete model."""

    name: str = "abstract"
    available: bool = False
    # Token usage of the most recent complete() call, if the provider reports it:
    # {"prompt_tokens", "completion_tokens", "total_tokens"} or None.
    last_usage: dict | None = None
    # Whether complete(prefill=...) is honoured: the text is handed to the model as the START of
    # its own reply (a trailing assistant message on chat endpoints). Positive form on purpose —
    # every adapter takes **kwargs, so an unsupported prefill would otherwise be dropped in
    # silence and the lane measured as if it had it. ToolBackend refuses to wrap a backend that
    # doesn't declare this when a prefill is configured.
    supports_prefill: bool = False

    @abstractmethod
    def complete(self, prompt: str, tier: ModelTier, **kwargs) -> str:  # pragma: no cover
        ...


class NullBackend(ModelBackend):
    """Default. Signals 'deterministic mode' — calling it is a programming error."""

    name = "null"
    available = False

    def complete(self, prompt: str, tier: ModelTier, **kwargs) -> str:
        raise RuntimeError(
            "NullBackend: no model wired. The MVP runs deterministically; pass a "
            "real ModelBackend (claude-cli / ollama) only when you want LLM-backed "
            "decomposition or review."
        )


class EchoBackend(ModelBackend):
    """Deterministic test double: returns a stable, inspectable JSON envelope."""

    name = "echo"
    available = True

    def complete(self, prompt: str, tier: ModelTier, **kwargs) -> str:
        return json.dumps({"tier": tier.value, "echo": prompt[:200]}, ensure_ascii=False)


class ClaudeCliBackend(ModelBackend):
    """Live seam — shells out to `claude --print`. Inert unless explicitly used.

    Mirrors recurse's ClaudeCliBackend. Tiers map to model flags; here we only
    pass the prompt and let the CLI default apply, to keep the seam dependency-free.
    """

    name = "claude-cli"
    available = True

    def __init__(self, model_by_tier: dict | None = None, timeout: int = 600):
        self.model_by_tier = model_by_tier or {
            ModelTier.SMALL: "haiku",
            ModelTier.MEDIUM: "sonnet",
            ModelTier.LARGE: "opus",
        }
        self.timeout = timeout

    def complete(self, prompt: str, tier: ModelTier, **kwargs) -> str:  # pragma: no cover
        model = self.model_by_tier.get(tier, "sonnet")
        # `effort` (kwargs) is the router's reasoning-effort decision; the print CLI
        # has no flag for it today, so it's a recorded seam — map it to the API
        # thinking budget when using an SDK/API backend instead.
        self.last_effort = kwargs.get("effort")
        proc = subprocess.run(
            ["claude", "--print", "--model", model],
            input=prompt, capture_output=True, text=True, timeout=self.timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"claude cli failed: {proc.stderr.strip()}")
        return proc.stdout


class OllamaBackend(ModelBackend):
    """Live seam — ollama, reachable two ways. Inert unless explicitly used.

    ``transport="http"`` (default): POST to ``{host}/api/generate`` (local box, or any
    box whose ollama HTTP port you can reach).
    ``transport="ssh"``: run ``ssh <ssh_host> ollama run <model>`` with the prompt on
    **stdin** — a sovereign "strong floor" for a box reachable only by SSH (e.g. a Mac
    Studio with NO open HTTP port; the prompt never appears in the remote argv/process
    list). Flip the whole ollama lane local<->remote by switching ``transport`` in
    config — no routing_table change needed.
    """

    name = "ollama"
    available = True

    # The model a tier falls back to when `model_by_tier` has no entry for it. Exposed as a
    # class attribute (not an inline literal in complete()) so that tools which must predict
    # what this backend will actually load — a VRAM budgeter, a trace verifier — can read it
    # instead of duplicating the string and drifting silently when it changes here.
    DEFAULT_MODEL = "gemma4:latest"

    # On this machine Ollama answers on localhost:11434. Under some WSL2 setups it
    # is only reachable via the Windows host route (e.g. http://172.24.224.1:11434);
    # override `host` if localhost fails.
    def __init__(self, host: str = "http://localhost:11434",
                 model_by_tier: dict | None = None,
                 transport: str = "http", ssh_host: str | None = None,
                 ssh_opts: list | None = None, timeout: int = 900,
                 remote_ollama: str = "ollama"):
        self.host = host.rstrip("/")
        self.transport = transport
        self.ssh_host = ssh_host
        self.ssh_opts = ssh_opts or ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
        self.timeout = timeout
        self.remote_ollama = remote_ollama
        self.last_usage = None
        self.last_finish_reason = None
        self.model_by_tier = model_by_tier or {
            ModelTier.SMALL: "gemma4:e2b",
            ModelTier.MEDIUM: self.DEFAULT_MODEL,
            ModelTier.LARGE: self.DEFAULT_MODEL,
        }

    def _ssh_cmd(self, model: str) -> list:
        """The argv for `ssh <opts> <host> ollama run <model>` (prompt goes on stdin)."""
        return ["ssh", *self.ssh_opts, self.ssh_host, self.remote_ollama, "run", model]

    def complete(self, prompt: str, tier: ModelTier, **kwargs) -> str:  # pragma: no cover
        model = self.model_by_tier.get(tier, self.DEFAULT_MODEL)
        if self.transport == "ssh":
            if not self.ssh_host:
                raise RuntimeError("OllamaBackend(transport='ssh') requires ssh_host")
            proc = subprocess.run(self._ssh_cmd(model), input=prompt,
                                  capture_output=True, text=True, timeout=self.timeout)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"ollama over ssh ({self.ssh_host}) failed: {proc.stderr.strip()[:300]}")
            # `ollama run` は本文しか出さない(done_reason は API だけ)。前の call の理由を残さない
            self.last_finish_reason = None
            return proc.stdout
        import urllib.request

        body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        pt, ct = data.get("prompt_eval_count"), data.get("eval_count")
        # ollama は done_reason("stop" / "length")。OpenAI 形式と同じ語なのでそのまま持つ。
        self.last_finish_reason = data.get("done_reason")
        self.last_usage = (
            {"prompt_tokens": pt or 0, "completion_tokens": ct or 0,
             "total_tokens": (pt or 0) + (ct or 0)}
            if (pt is not None or ct is not None) else None
        )
        # ollama も応答に model を載せる。tag 名しか名乗らないので ssh-openai ほど強い証拠には
        # ならないが(同じ tag が別の重みに貼り替わると見えない)、記録の口は揃えておく。
        note_served(f"{self.host}/{model}", data.get("model"))
        return data.get("response", "")


class SshOpenAIBackend(ModelBackend):
    """Live seam — call an OpenAI-compatible server on a remote host, over SSH.

    The remote server (MLX ``mlx_lm.server``, LM Studio, vLLM, llama.cpp, or ollama's
    ``/v1``) binds localhost only; SSH reaches it without opening a port. Runs
    ``ssh <host> curl -s localhost:<port><path> --data-binary @-`` with the request
    JSON on **stdin** (the prompt never appears in the remote argv). A sovereign
    "strong floor" — e.g. a Mac Studio running MLX. Inert unless explicitly used.

    Trust boundary: ``path`` is shell-quoted before reaching the remote command (it has
    no legitimate reason to carry shell metacharacters). ``ssh_host``/``ssh_opts`` are
    NOT sandboxed against a malicious config author -- ``ssh_opts`` exists specifically
    to pass arbitrary OpenSSH flags (``-o ProxyJump=...`` and friends), so restricting it
    would break its own purpose. This class (like ``ClaudeCliBackend``/``CodexBackend``
    elsewhere in this module) assumes the config author IS the person running gama;
    it is not a safe boundary for configs authored by an untrusted third party.
    """

    name = "ssh-openai"
    available = True
    # The flag is about the TRANSPORT: this adapter carries a prefill as a trailing assistant
    # turn. What the server does with it (continue the text, start a new turn, ignore it) is
    # the server's business and is read back by ToolBackend from the reply's shape. The one
    # bad case, a server that drops the turn silently, measures as a plain tool lane under
    # another name and never promotes: wasted width, not a wrong result.
    supports_prefill = True

    def __init__(self, ssh_host: str | None = None, port: int = 8080,
                 path: str = "/v1/chat/completions", model_by_tier: dict | None = None,
                 ssh_opts: list | None = None, timeout: int = 900,
                 max_tokens: int | None = None, temperature: float | None = None,
                 extra_body: dict | None = None):
        self.ssh_host = ssh_host
        self.port = port
        self.path = path
        self.model_by_tier = model_by_tier or {}
        self.ssh_opts = ssh_opts or ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature  # >0 gives ensemble diversity across repeats
        # サーバ固有の sampling パラメータを config から届かせるための素通し。
        # 実害(2026-09-02): temperature 0 の greedy decoding で同じ段落を反復し続け、
        # tool レーンがコードに到達する前に token 予算を使い切っていた。llama.cpp なら
        # `repeat_penalty` がその制御だが、backend が temperature と max_tokens しか
        # 組み立てないので、pool の config からは手が届かなかった。
        # 必須フィールド(model/messages/stream)は後から上書きするので、ここで壊せない。
        self.extra_body = dict(extra_body or {})
        self.last_usage = None
        # 応答が止まった理由(OpenAI 形式の finish_reason: "stop" / "length" ...)。tool レーンの
        # 「コードが出なかった」が **max_tokens で切れた**のか散文で答えたのかは、これが無いと
        # 見分けられない(2026-09-02 実測: Kimi-48B の research は 2048 tok を思考で使い切って
        # フェンスに届かない)。直し方が正反対(枠を増やす / 先頭に道具を置く)なので分けて残す。
        self.last_finish_reason = None

    def _remote_cmd(self) -> str:
        # `path` is config-controlled; the remote host runs this whole string through a
        # shell, so a bare `'{url}'` interpolation lets a `path` containing a single quote
        # break out of the quoting into arbitrary remote shell syntax. shlex.quote() closes
        # that regardless of what `path` contains (port is already int()-coerced, so it
        # can't carry shell metacharacters).
        url = f"http://localhost:{int(self.port)}{self.path}"
        return (f"curl -s -X POST {shlex.quote(url)} "
                f"-H 'Content-Type: application/json' --data-binary @-")

    def _ssh_cmd(self) -> list:
        """argv for `ssh <opts> <host> "<remote curl>"` (JSON body goes on stdin)."""
        return ["ssh", *self.ssh_opts, self.ssh_host, self._remote_cmd()]

    def complete(self, prompt: str, tier: ModelTier, **kwargs) -> str:  # pragma: no cover
        if not self.ssh_host:
            raise RuntimeError("SshOpenAIBackend requires ssh_host")
        model = self.model_by_tier.get(tier) or self.model_by_tier.get(ModelTier.LARGE)
        messages = [{"role": "user", "content": prompt}]
        # Prefill = the opening of the reply, sent as a trailing assistant turn. What the server
        # does with it is template-dependent: llama.cpp (2026-09-02, Kimi-48B IQ2_M) starts a
        # NEW assistant turn after it rather than continuing the text, but the model then opens
        # the fence itself (0/3 → 2-3/3 code on crux research); vLLM/Anthropic-style servers
        # continue the text. The caller (ToolBackend) handles both shapes on the way back.
        if kwargs.get("prefill"):
            messages.append({"role": "assistant", "content": kwargs["prefill"]})
        payload = dict(self.extra_body)
        payload.update({"model": model, "messages": messages, "stream": False})
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        proc = subprocess.run(self._ssh_cmd(), input=json.dumps(payload),
                              capture_output=True, text=True, timeout=self.timeout)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ssh-openai ({self.ssh_host}) failed: {proc.stderr.strip()[:300]}")
        data = json.loads(proc.stdout)
        usage = data.get("usage") or {}
        if usage:
            pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
            self.last_usage = {"prompt_tokens": pt, "completion_tokens": ct,
                               "total_tokens": usage.get("total_tokens", pt + ct)}
        # 応答が名乗った実体を控える。追加の呼び出しはしない(この 1 行が同一性の証拠)。
        note_served(f"{self.ssh_host}:{int(self.port)}/{model}", data.get("model"))
        self.last_finish_reason = data["choices"][0].get("finish_reason")
        msg = data["choices"][0]["message"]
        # Reasoning models put the answer in `content`; fall back to `reasoning`
        # (or empty) so a thinking-only / truncated reply doesn't crash the call.
        return msg.get("content") or msg.get("reasoning") or ""


class ClaudeTuiBackend(ModelBackend):
    """Live seam — drives the Claude Code **interactive TUI** via claude-cli-run.py.

    This is the flat-subscription lane (covered by the Max plan, throttled by the
    rolling usage window) — deliberately NOT ``claude --print`` / Agent-SDK, which
    meters against Agent-SDK credits and bills overage. Low concurrency (tmux), so
    it is the high-quality/rate-limited lane; bulk volume belongs on ``ollama``.
    Inert unless explicitly used.
    """

    name = "claude-tui"

    DEFAULT_SCRIPT = "/home/muko1/Projects/claude-headless-via-tui/claude-cli-run.py"

    def __init__(self, script: str | None = None, model_by_tier: dict | None = None,
                 timeout: int = 600, permission_mode: str | None = None,
                 use_sentinel: bool = True):
        self.script = script or self.DEFAULT_SCRIPT
        self.model_by_tier = model_by_tier or {
            ModelTier.SMALL: "claude-haiku-4-5-20251001",
            ModelTier.MEDIUM: "claude-sonnet-4-6",
            ModelTier.LARGE: "claude-opus-4-8",
        }
        self.timeout = timeout
        self.permission_mode = permission_mode  # None -> let the script default apply
        # use_sentinel=True waits for the completion marker = the FULL answer. False
        # (--no-sentinel) returns the *first* assistant response, which can TRUNCATE the
        # model mid-answer — measured: opus returned 394 (truncated) vs 396 (completed)
        # on "17*23+5". Default to True so quality matches `claude -p`.
        self.use_sentinel = use_sentinel
        self.available = os.path.exists(self.script)
        self.last_usage = None

    def complete(self, prompt: str, tier: ModelTier, **kwargs) -> str:  # pragma: no cover
        model = self.model_by_tier.get(tier, "claude-sonnet-4-6")
        self.last_effort = kwargs.get("effort")
        cmd = ["python3", self.script, "--model", model]
        if not self.use_sentinel:
            cmd += ["--no-sentinel"]
        if self.permission_mode:
            cmd += ["--permission-mode", self.permission_mode]
        proc = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True, timeout=self.timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"claude-tui failed: {proc.stderr.strip()[:300]}")
        return proc.stdout


class CodexBackend(ModelBackend):
    """Live seam — shells out to `codex exec` (non-interactive). Inert unless used.

    Runs in the Codex/ChatGPT-subscription lane by default. ``model_by_tier`` is
    empty by default so Codex's own configured model is used (no fabricated ids);
    pass a mapping to force per-tier models.
    """

    name = "codex"
    available = True

    def __init__(self, model_by_tier: dict | None = None, timeout: int = 900):
        self.model_by_tier = model_by_tier or {}
        self.timeout = timeout
        self.last_usage = None

    def complete(self, prompt: str, tier: ModelTier, **kwargs) -> str:  # pragma: no cover
        import tempfile

        model = self.model_by_tier.get(tier)
        fd, outfile = tempfile.mkstemp(suffix=".txt", prefix="tehai-codex-")
        os.close(fd)
        try:
            cmd = ["codex", "exec", "--json", "-o", outfile,
                   "--dangerously-bypass-approvals-and-sandbox"]
            if model:
                cmd += ["-m", model]
            proc = subprocess.run(
                cmd, input=prompt, capture_output=True, text=True, timeout=self.timeout,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"codex exec failed: {proc.stderr.strip()[:300]}")
            try:
                text = open(outfile, encoding="utf-8").read()
            except OSError:
                text = ""
            return text.strip() or self._last_message_from_jsonl(proc.stdout)
        finally:
            try:
                os.unlink(outfile)
            except OSError:
                pass

    @staticmethod
    def _last_message_from_jsonl(stream: str) -> str:  # pragma: no cover
        """Best-effort: last assistant message text from a JSONL event stream."""
        last = ""
        for line in stream.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if not isinstance(ev, dict):
                continue
            if "message" in str(ev.get("type", "")).lower():
                msg = ev.get("message") or ev.get("text") or ev.get("content")
                if isinstance(msg, dict):
                    msg = msg.get("content") or msg.get("text")
                if isinstance(msg, str) and msg.strip():
                    last = msg
        return last


class GeminiBackend(ModelBackend):
    """Pluggable seam — Gemini via its OpenAI-compatible endpoint (urllib, no dep).

    Available only when an API key is present (``GEMINI_API_KEY`` by default), so it
    can be wired in later ("後付け") without affecting the other lanes. Inert otherwise.
    """

    name = "gemini"

    def __init__(self, api_key: str | None = None,
                 base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai",
                 model_by_tier: dict | None = None, timeout: int = 600):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.model_by_tier = model_by_tier or {
            ModelTier.SMALL: "gemini-2.5-flash",
            ModelTier.MEDIUM: "gemini-2.5-flash",
            ModelTier.LARGE: "gemini-2.5-pro",
        }
        self.timeout = timeout
        self.available = bool(self.api_key)
        self.last_usage = None
        self.last_finish_reason = None

    def complete(self, prompt: str, tier: ModelTier, **kwargs) -> str:  # pragma: no cover
        import urllib.request

        if not self.api_key:
            raise RuntimeError("GeminiBackend: no API key (set GEMINI_API_KEY)")
        model = self.model_by_tier.get(tier, "gemini-2.5-flash")
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        usage = data.get("usage") or {}
        if usage:
            self.last_usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
        self.last_finish_reason = data["choices"][0].get("finish_reason")
        return data["choices"][0]["message"]["content"]


class GamaBackend(ModelBackend):
    """Deterministic vendor router — the 'Conductor' (project name: **gama** / 蝦蟇).

    Coordinates a pool of model backends while keeping a local sovereignty lane. Holds
    named sub-backends and a ``routing_table``
    mapping a task_type value to a sub-backend name; ``complete()`` reads ``task_type``
    from kwargs (threaded in by the executor) and dispatches to the chosen sub-backend,
    falling back to ``default`` when the type is unmapped or absent. The table is
    *measured* by ``tehai bench`` and adopted via config (human-ratified, like
    calibrate) — never self-modified. Routing fires on measured performance, not a
    model's self-report.
    """

    name = "gama"

    def __init__(self, backends: dict[str, ModelBackend],
                 routing_table: dict[str, str] | None = None,
                 default: str | None = None):
        if not backends:
            raise ValueError("GamaBackend needs at least one sub-backend")
        self.backends = dict(backends)
        self.routing_table = dict(routing_table or {})
        self.default = default or next(iter(self.backends))
        if self.default not in self.backends:
            raise ValueError(
                f"default backend {self.default!r} not among {sorted(self.backends)}"
            )
        self.available = any(getattr(b, "available", False) for b in self.backends.values())
        self.last_usage = None
        self.last_route: tuple | None = None

    def pick(self, task_type: str | None) -> str:
        """Return the sub-backend name for a task_type (deterministic table lookup)."""
        name = self.routing_table.get(task_type, self.default) if task_type else self.default
        return name if name in self.backends else self.default

    def complete(self, prompt: str, tier: ModelTier, **kwargs) -> str:
        task_type = kwargs.get("task_type")
        name = self.pick(task_type)
        self.last_route = (task_type, name)
        backend = self.backends[name]
        out = backend.complete(prompt, tier, **kwargs)
        self.last_usage = getattr(backend, "last_usage", None)
        self.last_finish_reason = getattr(backend, "last_finish_reason", None)
        return out


def synthesize(aggregator, prompt: str, tier: ModelTier, candidates: list,
               instruction: str | None = None, **kwargs) -> str:
    """Aggregate candidate answers into one final answer via an aggregator backend
    (the classic Mixture-of-Agents synthesize step). Shared by ``EnsembleBackend`` and
    ``MeshflowBackend``'s edge-mesh so the logic lives in one place. Falls back to the
    first candidate if the aggregator errors. The caller reads ``aggregator.last_usage``."""
    listing = "\n".join(f"--- candidate {i + 1} ---\n{c[:1500]}"
                        for i, c in enumerate(candidates))
    instruction = instruction or (
        "Using the candidates, output the single best FINAL answer. Follow the "
        "original task's format instruction EXACTLY. Output only the final answer."
    )
    agg_prompt = f"Original task:\n{prompt}\n\nCandidate answers:\n{listing}\n\n{instruction}"
    try:
        return aggregator.complete(agg_prompt, tier, **kwargs)
    except Exception:
        return candidates[0] if candidates else ""


class EnsembleBackend(ModelBackend):
    """Mixture-of-Agents — run several sub-backends on the SAME prompt and combine.

    Where ``GamaBackend`` *routes* (1 task → 1 vendor), this *combines* (N models → 1
    answer) — the model-combination loop, living on the seam so the orchestrator,
    ``tehai run``, and ``tehai bench`` can drive it like any backend. Strategies:
      - ``synthesize`` (default): an aggregator backend reads all candidates and writes
        the final answer (classic MoA aggregator).
      - ``majority``: return the most common candidate (whitespace-normalized).
      - ``first``: first non-empty candidate.
    A single sub-backend may be repeated N times (homogeneous self-ensemble); pair it
    with a ``temperature``>0 backend for diversity. Members run sequentially; a member
    that errors contributes an empty candidate (the sweep never aborts).
    """

    name = "ensemble"

    def __init__(self, members, strategy: str = "synthesize", aggregator=None,
                 aggregator_prompt: str | None = None):
        if not members:
            raise ValueError("EnsembleBackend needs at least one member")
        self.members = list(members)
        self.strategy = strategy
        self.aggregator = aggregator  # for "synthesize"; defaults to members[0]
        self.aggregator_prompt = aggregator_prompt
        self.available = any(getattr(m, "available", False) for m in self.members)
        self.last_usage = None
        self.last_candidates: list | None = None

    def complete(self, prompt: str, tier: ModelTier, **kwargs) -> str:
        cands = []
        for m in self.members:
            try:
                cands.append(m.complete(prompt, tier, **kwargs))
            except Exception:
                cands.append("")
        self.last_candidates = cands
        nonempty = [c for c in cands if c and c.strip()]
        if not nonempty:
            return ""
        if self.strategy == "first":
            return nonempty[0]
        if self.strategy == "majority":
            return self._majority(nonempty)
        return self._synthesize(prompt, tier, cands, **kwargs)

    @staticmethod
    def _majority(cands: list) -> str:
        from collections import Counter

        counts = Counter(" ".join(c.split()) for c in cands)
        best_norm = counts.most_common(1)[0][0]
        for c in cands:
            if " ".join(c.split()) == best_norm:
                return c
        return cands[0]

    def _synthesize(self, prompt: str, tier: ModelTier, cands: list, **kwargs) -> str:
        agg = self.aggregator or self.members[0]
        out = synthesize(agg, prompt, tier, cands, instruction=self.aggregator_prompt, **kwargs)
        self.last_usage = getattr(agg, "last_usage", None)
        return out


# --------------------------------------------------------------------------- #
# tool レーンが実際に「道具」として働いたかの計数
# --------------------------------------------------------------------------- #
# ToolBackend はコードを取り出せなかったとき、黙ってモデルの生テキストを返す(fall back)。
# これは「効かなかった」ではなく**道具を使えていない**で、しかも例外にならないので
# error_rate にも出ず、低い得点として静かに data に混ざる。実測(2026-09-02, Kimi-48B):
# crux の research 4 問はすべて ```python が一度も出ず、生の思考文が返っていた。
# その状態の tool レーンは素のモデルより悪くなる(答えでなく途中の思考が採点される)。
# 3 状態を分ける。「コードが出てこない」と「コードは動いたが何も print しなかった」は
# 直し方が正反対(前者は prompt/モデルの問題・後者は生成コードの問題)なので、まとめて
# 「fall back」と数えると診断にならない。run_bench は直列なのでグローバルで足りる。
_TOOL: dict = {"calls": 0, "ran": 0, "no_code": 0, "empty_out": 0}


def note_tool(*, ran: bool, had_code: bool) -> None:
    _TOOL["calls"] += 1
    if ran:
        _TOOL["ran"] += 1
    elif had_code:
        _TOOL["empty_out"] += 1      # コードは取れて走ったが、出力が空だった
    else:
        _TOOL["no_code"] += 1        # そもそも ```python が出てこなかった


def tool_stats() -> dict:
    """calls / ran / no_code / empty_out。calls が 0 なら tool レーンを通っていない。"""
    return dict(_TOOL)


def reset_tool_stats() -> None:
    for k in _TOOL:
        _TOOL[k] = 0


def clear_finish_reason(backend) -> None:
    """``last_finish_reason`` を、この backend と**その中の全 backend** で None にする。

    合成 backend(GamaBackend / ToolBackend など)は内側の値を写すだけで自分では作らない。外側
    だけ消しても、内側が「今回は設定しなかった」(例外を握って素の返答を返す実装・毎回は
    書かない外部の adapter)と前の call の理由が今の call に載る。消すのは読む側(``_run_one``)
    の仕事で、読む側は木の全部を消す。子の辿り方は属性の中身で決める(dict / list / tuple の
    中の ModelBackend も含む)ので、包む側ごとに消し方を足して回る必要が無い。
    """
    seen: set = set()
    stack = [backend]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, ModelBackend):
            if hasattr(node, "last_finish_reason"):
                node.last_finish_reason = None
            stack.extend(vars(node).values())
        elif isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, (list, tuple)):
            stack.extend(node)


def _parses(code: str) -> bool:
    try:
        compile(code, "<tool>", "exec")
        return True
    except (SyntaxError, ValueError):
        return False


class ToolBackend(ModelBackend):
    """Program-aided (PAL) wrapper — the model solves by WRITING Python that prints the
    answer; we run it and return stdout. Closes 'shared blind spot' gaps a small model
    can express as code but can't do in its head (e.g. exact arithmetic). Falls back to
    the model's direct answer if no code is produced or it fails.

    Best applied selectively (math/computational classes); forcing code on a 'write
    prose' task hurts. SECURITY: runs model-generated code in a subprocess (opt-in,
    like --sandbox). Wraps any ModelBackend (including an EnsembleBackend).
    """

    name = "tool"

    _PY_FENCE = re.compile(r"```(?:python|py)?\n(.*?)```", re.DOTALL)
    _OPEN_FENCE = re.compile(r"```(?:python|py)?\n")
    # The prefill this wrapper knows how to read back. It is the opening of the fence the
    # extractor looks for, so the two must move together — which is why it lives here and
    # grow imports it instead of spelling the string again.
    PREFILL = "```python\n"

    def __init__(self, backend, timeout: int = 15, prefill: str | None = None):
        # prefill: hand the model the start of its reply (see ModelBackend.supports_prefill).
        # Why: a reasoning-tuned model asked for "ONLY the code" may still narrate its way to
        # the token limit and never open a fence (Kimi-48B IQ2_M on crux research: 0/3 at any
        # temperature or max_tokens up to 8192). Opening the fence FOR it turned that into
        # 2-3/3 replies with runnable code. Opt-in per lane, because on prompts where the model
        # already writes code it is an unmeasured change, and grow should decide from the
        # ledger, not from here.
        if prefill and not getattr(backend, "supports_prefill", False):
            raise ValueError(
                f"ToolBackend(prefill=...) over {getattr(backend, 'name', type(backend).__name__)!r}: "
                "that backend does not declare supports_prefill, so the prefill would be dropped "
                "silently and the lane measured as if it had it")
        self.backend = backend
        self.timeout = timeout
        self.prefill = prefill or None
        self.available = getattr(backend, "available", False)
        self.last_usage = None
        self.last_code = None

    def _extract(self, raw: str) -> tuple[str, bool]:
        """The program in ``raw`` and whether one was fenced at all.

        Several reply shapes reach here and all must yield the code, because the difference
        between them is the *server's* handling of a prefill, not the model's work:
          1. a closed block anywhere (no prefill, or a server that re-opened the fence);
          2. the continuation of OUR fence: body, then a closing fence, with no opening fence in
             the reply — re-attach the prefill before matching (never when the reply opens one
             itself, or the match is the empty block between the two fences — found exactly
             that: three replies full of code all scoring 0);
          3. an opened fence that never closes (length stop): what follows it is the program;
          4. the continuation of our fence that never closes: the whole reply is the program.
        The shapes are ambiguous on the surface (a bare closing fence in a continuation also
        reads as the model opening an untagged block; a reply that opens a fence mid-text under
        a prefill pairs OUR opener with ITS opener and yields the prose between them as the
        "block"), so the readings are tried in that order and the first that parses as Python
        wins. Only if none parse does the first reading stand, and the run then fails the way
        it always did. Parsing is a tiebreaker, not proof: a one-word remark parses too, which
        is why the order puts the stronger readings first.
        """
        text = raw or ""
        own_fence = "```" in text
        continuation = bool(self.prefill) and not text.lstrip().startswith("```")
        readings: list[str] = []
        blocks = self._PY_FENCE.findall(text)
        if blocks:                                       # 1. the model's own closed block
            readings.append(max(blocks, key=len))
        if continuation:
            blocks = self._PY_FENCE.findall(self.prefill + text)
            if blocks:                                   # 2. continuation of our fence, closed
                readings.append(max(blocks, key=len))
        m = self._OPEN_FENCE.search(text)
        if m:                                            # 3. the model opened, never closed
            readings.append(text[m.end():])
        if continuation:
            readings.append(text)                        # 4. continued, never closed
        if not readings:
            return text, False
        code = next((r for r in readings if _parses(r)), readings[0])
        # "Had code" is about the model: a fence it wrote itself counts even when the body is
        # not a program (that reply is code that FAILED, fixed on the generation side); prose
        # after OUR fence is the model declining to write code (no_code, fixed on the prompt
        # side). The two are fixed in opposite places, so they must not be counted together.
        return code, (own_fence or _parses(code))

    def complete(self, prompt: str, tier: ModelTier, **kwargs) -> str:
        pal = (f"{prompt}\n\nSolve by writing a short Python 3 program that computes the "
               "answer and prints ONLY the final answer with print(). Return ONLY the "
               "code in a ```python code block.")
        if self.prefill:
            kwargs = dict(kwargs, prefill=self.prefill)
        raw = self.backend.complete(pal, tier, **kwargs)
        self.last_usage = getattr(self.backend, "last_usage", None)
        # 内側のモデルが止まった理由をそのまま外へ(道具の側は理由を作らない)
        self.last_finish_reason = getattr(self.backend, "last_finish_reason", None)
        code, had_code = self._extract(raw)
        self.last_code = code
        try:
            # Run in a throwaway directory: a model asked for CSV happily writes
            # `output.csv` next to whatever the caller was working on, and a benchmark that
            # litters the repo it is being run from is its own kind of side effect. (Found
            # exactly that file committed-adjacent after a bench sweep.)
            import tempfile

            with tempfile.TemporaryDirectory(prefix="gama-tool-") as sandbox:
                proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                      text=True, timeout=self.timeout, cwd=sandbox)
            out = proc.stdout.strip()
            if out:
                note_tool(ran=True, had_code=had_code)
                return out
        except Exception:
            pass
        # ここに来たら道具は働いていない。返すのは素の返答だが、**その事実を数える**
        # (黙って低い点になるだけだと、道具が壊れているのかモデルが弱いのか区別できない)。
        # コードが在ったかどうかも一緒に残す —— 直し方が正反対なので混ぜない。
        note_tool(ran=False, had_code=had_code)
        return raw  # fall back to the model's direct answer


_BACKENDS = {
    "null": NullBackend,
    "echo": EchoBackend,
    "claude-cli": ClaudeCliBackend,
    "claude-tui": ClaudeTuiBackend,
    "codex": CodexBackend,
    "gemini": GeminiBackend,
    "ollama": OllamaBackend,
    "ssh-openai": SshOpenAIBackend,
}


def get_backend(name: str = "null", **kwargs) -> ModelBackend:
    """Factory. Defaults to the deterministic NullBackend.

    Extra kwargs are forwarded to the adapter constructor (e.g.
    get_backend("ollama", host="http://172.24.224.1:11434")).
    """
    try:
        cls = _BACKENDS[name]
    except KeyError:
        raise ValueError(f"unknown backend {name!r}; choose from {sorted(_BACKENDS)}")
    return cls(**kwargs) if kwargs else cls()
