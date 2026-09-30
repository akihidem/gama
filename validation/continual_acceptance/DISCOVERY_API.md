# Existing frozen RSI APIs

This is a parent-authored interface summary of the actual source at d6a0447.
Existing modules remain frozen, and independent behavior checks exercise them.
It reduces repeated whole-module prompt context; it does not replace testing.


## Python and imports

Use sys.executable without resolving symlinks (production is the pinned 3.12
venv). Existing package __init__ eagerly imports application modules. A scoring
script must put candidate cwd first before importing any gama module. Workers
can import controller siblings through a private synthetic package rooted at
Path(__file__).parent, as rsi_bridge._sibling does, to avoid candidate __init__.

## Guard and durable helpers

`gama.rsi_guard.run_guarded(command: list[str], *, cwd, timeout, artifact_dir,
input_text="", cancel=None, max_output_bytes=1048576) -> ProcessResult`.
Result has returncode, stdout, stderr, elapsed_s. Raises rsi_process.ProcessError
on deadline, cancellation, output/encoding or containment failure. artifact_dir
must be fresh/empty, external to candidate repo. Every invocation needs a unique
directory. The Linux guardian contains descendants including setsid/double-fork,
and drains for at most ~4.5 seconds after owner death/cancel. Input must be str.
There is no `env` parameter. Credentials are inherited, never serialized.

rsi_mission `_read(Path)->dict`, `_write(Path,text)`, `_put(Path,value)` implement
atomic replacement plus fsync including parent directory. `_owner(root)` uses a
stable owner.lock inode and PID/starttime/boot_id receipt; `_ownership(root)` gives
live/stale/none. `_control(root, bool)` persists/removes STOP under separate lock.
`_cancellation(root)` yields a threading.Event watching STOP. These are private
helpers with current semantics; reuse is optional, never edit their source.

## Guarded model transport reusable helpers

`gama.rsi_bridge._load_backend(normalized_bridge_config)` verifies the explicit
astra_loop package location and creates `LiveBackend(config["backend"])`.
`_identities(backend, evidence_dir)` writes identities.json once and enforces
actual Astra/OpenAI and Claude/Anthropic models; evidence_dir already exists.
`_complete(backend, role, prompt, evidence_dir, budget_bytes)->str` invokes
LiveBackend.complete once in current cwd, writes role-output.txt and
role-usage.json, and uses fresh evidence_dir/role for adapter-native logs.
It enforces bounded text; it does not itself impose a deadline. The whole worker
must be contained by run_guarded. A role cannot be repeated in that evidence dir.
Redirect noisy adapter stdout to stderr around calls to keep worker JSON clean.

`_loads(text_or_bytes)` rejects duplicate keys, NaN/infinite floats and invalid
UTF-8. `_prompt(instructions, marker, payload, limit)` constructs instructions +
newline + marker + newline + compact JSON and rejects too many UTF-8 bytes.
New discovery markers are DISCOVERY_JSON and DISCOVERY_REVIEW_JSON. Use their
different goal schema rather than the existing source patch worker.
LiveBackend.identity(role) returns provider,model,family,resolved_model. Roles
are builder/reviewer, not planner. Native complete may return text, tuple or
completion object; _complete records usage from all supported forms/meta.json.
No tools or model fallback, and one call per role per scout.
