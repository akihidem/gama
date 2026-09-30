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

## Existing mission CLI

`python -B -m gama.rsi_mission run|status|stop|resume --mission /abs/file.json`.
Mission format: mission_id, absolute repo/state_dir/rsi_config/bridge_config,
search_ceiling, confirm_ceiling, rounds_per_cycle (1 or 2), batch_size=2,
max_reservations_per_cycle=4. State_dir must be external/disjoint from repo.
Core is under state_dir/rsi; mission checkpoint is state_dir/state.json.
Artifacts must be external to repo and disjoint from state_dir/rsi.

`gama.rsi_runtime.load_inputs(Path)->dict` gives mission, rsi_config,
bridge_config, digest. It validates inputs, overrides agents with the fixed
bridge commands using this interpreter, appends all three config paths to
evaluation_files, and fingerprints mission/config/controller bytes and Python.
`load_bridge_config(Path)->dict` validates the bridge schema.

Bridge config: absolute astra_loop_root, external artifact_root, positive timeout
(510 production), backend dict with backend_python, timeout_seconds,
bedrock_max_tokens, bedrock_effort, max_context_bytes, dotclaude_scripts,
bedrock_script, codex_transport. Preserve adapter-specific backend options.
RSI timeout must be >= bridge timeout + 5.

Mission run opens a cycle if the preceding one closed. resume only continues
the existing cycle or finalization and clears its STOP/blockage; it does not
open another closed cycle. Ordinary run leaves a blocked mission blocked.
The coordinator must choose run/resume deliberately and buy a fresh pair before
any possibly paid invocation. With rounds_per_cycle=1 each invocation reserves
at most one new pair, even when retrying a pending failed attempt. Prior lost
attempts are still counted, so the inner cycle can have four reservations.
No invocation loops through a failed pair: uncheckpointed failure blocks.

Before proposing, the mission measures a seed and stops the core at its seed
event. If already perfect, it finalizes with zero source proposals. At search
and confirmation ceilings it finalizes automatically. Finalization is irreversible
and reuses checkpointed measurements on resume. Saturated is only a goal status.
Status fields: phase, cycle_open, reserved_baseline, cycle_reservations,
last_outcome, champion, ref, patch, error, counts:{cycles,proposals}, ownership.
Initial champion is null. Return codes: 0 normal, 2 runtime error, 3 overlap,
4 stopped/blocked, 5 changed frozen inputs. It never adopts or pushes the caller.
