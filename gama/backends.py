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
import subprocess
import sys
from abc import ABC, abstractmethod

from .models import ModelTier


class ModelBackend(ABC):
    """Minimal completion interface. Tier lets an adapter pick a concrete model."""

    name: str = "abstract"
    available: bool = False
    # Token usage of the most recent complete() call, if the provider reports it:
    # {"prompt_tokens", "completion_tokens", "total_tokens"} or None.
    last_usage: dict | None = None

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
        self.model_by_tier = model_by_tier or {
            ModelTier.SMALL: "gemma4:e2b",
            ModelTier.MEDIUM: "gemma4:latest",
            ModelTier.LARGE: "gemma4:latest",
        }

    def _ssh_cmd(self, model: str) -> list:
        """The argv for `ssh <opts> <host> ollama run <model>` (prompt goes on stdin)."""
        return ["ssh", *self.ssh_opts, self.ssh_host, self.remote_ollama, "run", model]

    def complete(self, prompt: str, tier: ModelTier, **kwargs) -> str:  # pragma: no cover
        model = self.model_by_tier.get(tier, "gemma4:latest")
        if self.transport == "ssh":
            if not self.ssh_host:
                raise RuntimeError("OllamaBackend(transport='ssh') requires ssh_host")
            proc = subprocess.run(self._ssh_cmd(model), input=prompt,
                                  capture_output=True, text=True, timeout=self.timeout)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"ollama over ssh ({self.ssh_host}) failed: {proc.stderr.strip()[:300]}")
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
        self.last_usage = (
            {"prompt_tokens": pt or 0, "completion_tokens": ct or 0,
             "total_tokens": (pt or 0) + (ct or 0)}
            if (pt is not None or ct is not None) else None
        )
        return data.get("response", "")


class SshOpenAIBackend(ModelBackend):
    """Live seam — call an OpenAI-compatible server on a remote host, over SSH.

    The remote server (MLX ``mlx_lm.server``, LM Studio, vLLM, llama.cpp, or ollama's
    ``/v1``) binds localhost only; SSH reaches it without opening a port. Runs
    ``ssh <host> curl -s localhost:<port><path> --data-binary @-`` with the request
    JSON on **stdin** (the prompt never appears in the remote argv). A sovereign
    "strong floor" — e.g. a Mac Studio running MLX. Inert unless explicitly used.
    """

    name = "ssh-openai"
    available = True

    def __init__(self, ssh_host: str | None = None, port: int = 8080,
                 path: str = "/v1/chat/completions", model_by_tier: dict | None = None,
                 ssh_opts: list | None = None, timeout: int = 900,
                 max_tokens: int | None = None, temperature: float | None = None):
        self.ssh_host = ssh_host
        self.port = port
        self.path = path
        self.model_by_tier = model_by_tier or {}
        self.ssh_opts = ssh_opts or ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature  # >0 gives ensemble diversity across repeats
        self.last_usage = None

    def _remote_cmd(self) -> str:
        url = f"http://localhost:{int(self.port)}{self.path}"
        return (f"curl -s -X POST '{url}' "
                f"-H 'Content-Type: application/json' --data-binary @-")

    def _ssh_cmd(self) -> list:
        """argv for `ssh <opts> <host> "<remote curl>"` (JSON body goes on stdin)."""
        return ["ssh", *self.ssh_opts, self.ssh_host, self._remote_cmd()]

    def complete(self, prompt: str, tier: ModelTier, **kwargs) -> str:  # pragma: no cover
        if not self.ssh_host:
            raise RuntimeError("SshOpenAIBackend requires ssh_host")
        model = self.model_by_tier.get(tier) or self.model_by_tier.get(ModelTier.LARGE)
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "stream": False}
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
        return out


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
        listing = "\n".join(f"--- candidate {i + 1} ---\n{c[:1500]}"
                            for i, c in enumerate(cands))
        instruction = self.aggregator_prompt or (
            "Using the candidates, output the single best FINAL answer. Follow the "
            "original task's format instruction EXACTLY. Output only the final answer."
        )
        agg_prompt = f"Original task:\n{prompt}\n\nCandidate answers:\n{listing}\n\n{instruction}"
        try:
            out = agg.complete(agg_prompt, tier, **kwargs)
        except Exception:
            out = cands[0]
        self.last_usage = getattr(agg, "last_usage", None)
        return out


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

    def __init__(self, backend, timeout: int = 15):
        self.backend = backend
        self.timeout = timeout
        self.available = getattr(backend, "available", False)
        self.last_usage = None
        self.last_code = None

    def complete(self, prompt: str, tier: ModelTier, **kwargs) -> str:
        pal = (f"{prompt}\n\nSolve by writing a short Python 3 program that computes the "
               "answer and prints ONLY the final answer with print(). Return ONLY the "
               "code in a ```python code block.")
        raw = self.backend.complete(pal, tier, **kwargs)
        self.last_usage = getattr(self.backend, "last_usage", None)
        blocks = self._PY_FENCE.findall(raw or "")
        code = max(blocks, key=len) if blocks else (raw or "")
        self.last_code = code
        try:
            proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                  text=True, timeout=self.timeout)
            out = proc.stdout.strip()
            if out:
                return out
        except Exception:
            pass
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
