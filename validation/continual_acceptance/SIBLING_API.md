Supervisor dependency APIs; original contracts remain authoritative

No sibling edits or invented helpers. Preserve sys.executable spelling/controller
location. Evaluate candidate cwd; keep state/artifacts external.

Accepted tasks also exposes frozen private helpers that may be reused here:
`_configuration(raw)->dict` validates the exact campaign schema, normalizes
defaults, checks absolute/disjoint paths and the currently checked-out feature
branch with bounded read-only Git. Use it for execution, not status/stop when
configuration or checkout is broken. `_read(absolute_path, limit=16MiB)->bytes`
requires a bounded regular non-symlink file; `_loads(bytes)->value` rejects
duplicate keys and nonfinite JSON; `_dump(value)->bytes` is sorted indented JSON;
`_goal(raw)->dict` validates only descriptor schema/paths/test syntax without
reading current mutation targets. This last distinction preserves old history.

`continual_tasks.validate_goal(raw: dict, repo: Path) -> dict`
validates {id,title,goal,allowed_paths,tests}, with search/confirm/sealed unittest
source. Targets are exact tracked regular gama/*.py except rsi*, continual*,
__init__, __main__, cli. Tests/config stay protected; sealed is compiled, not run.

`freeze_goal(config: dict, goal: dict, directory: Path, prior_goals: list) -> Path`
returns absolute mission_path. prior_goals contains absolute accepted descriptor
paths, not IDs. Writes goal/scripts/regressions/controls/inventory; preserves
mandatory checks plus cumulative regressions. workers=batch_size=2,
rounds_per_cycle=1, ceilings=1, min_gain=0, search_repeats=1, confirm_repeats=3
(also sealed; no sealed_repeats). Freeze once; reuse the path after publication.

Evaluator CLI: `python -I -B ABS_continual_tasks.py score --test ABS_SCRIPT`
or `regressions --goals ABS_MANIFEST`. score emits finite
{score,tests_run,failures,errors}; assertions are measurements. Regressions verify
bytes and require all passing. Zero tests/load errors refuse.

`continual_discover.discover(config, *, directory: Path, seen: list,
cursor: int, cancel) -> dict`
returns {goals:[validated descriptors],cursor:int,rejected:[reasons]}.
seen contains descriptors, not IDs; cursor is nonnegative. Caller already bought
two reservations. Uses guarded scouts/review and unique evidence under directory,
baseline search only. Persist cursor/seen/rejections; [] is legitimate.
cancel is Event-like (or None).

`continual_publish.publish(config, *, goal, mission_path: Path, journal: dict,
save, cancel) -> dict`
requires journal["base_commit"] equal expected released HEAD and the original
mission_path with matching finalized/sealed-improved evidence. Validates source,
checks/regressions, journals release SHA, fast-forwards the clean branch, normally
pushes and verifies remote SHA. save(snapshot) durably saves the ENTIRE journal
before returning. Preserve publisher-owned fields unchanged for recovery.
Success returns phase="published", commit/release_commit, source_commit,
remote_commit, branch, remote, ref, regression_descriptor and regression_files.
Use release_commit as next base, regression_descriptor for later goals.
Reconcile the same journal/path; no new inference or early history advancement.

`rsi_runtime.load_inputs(Path) -> dict`
returns mission, rsi_config, bridge_config, digest. Replaces agents with astra-a/b:
[sys.executable,"-I","-B",ABS_rsi_bridge.py,"--config",ABS_bridge].
Appends mission/RSI/bridge config paths to evaluation_files; retain tasks' files.
Identity covers controls/bytes/controller paths/Python.
`load_bridge_config(Path) -> dict` validates bridge config.

Mission CLI: `python -B -m gama.rsi_mission ACTION --mission ABS_JSON`.
run opens a closed cycle; resume continues existing cycle/finalization and clears
its STOP/blockage. rounds_per_cycle=1 means at most two NEW proposals per invocation,
including pending retry. Core: mission state_dir/rsi; initial champion may be absent.
Saturated is not sealed improvement. Status/exit codes: SUPERVISOR_API.

`rsi_guard.run_guarded(command, *, cwd, timeout, artifact_dir, input_text="",
cancel=None, max_output_bytes=1048576) -> ProcessResult`
has no env argument. Result: returncode,stdout,stderr,elapsed_s; ProcessError on
deadline/cancellation/containment failure. Use fresh external artifacts each time.
process.json starts EMPTY before guardian Popen. Only the guardian writes exact
{returncode:int|null,error:str|null} after _drain. Cancellation/deadline errors can
still mean successful drain; "children did not exit after kill"/"cleanup failed"
mean containment failure. Missing/incomplete records are not clearance.
Use the enclosing guard's receipt, not an inner worker lock. No guardian-PID field
or public wait-for-drain helper exists. Startup intent + receipt reconciliation,
blocking on ambiguity, is sufficient.

Existing `rsi_mission` helpers: `_put(Path,value)` atomically writes JSON with
file/parent fsync; `_owner(root)` is a context manager holding stable owner.lock
and writing owner.json identity on entry, {} on exit; `_ownership(root)` returns
live/stale/none. `_control(root,stop:bool)` manages independent persistent STOP;
`_cancellation(root)` is a context manager yielding an Event watching that file.
`_identity(pid)` returns {pid,starttime,boot_id} or None for dead/reused processes.
Validate corrupt existing checkpoints before entering `_owner` (which writes),
and revalidate after ownership. `_wait_core` is only an inner lock check, not
proof that enclosing guardians/providers drained. Do not substitute it for the
enclosing action receipt.
