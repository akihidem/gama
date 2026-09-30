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

## RSIConfig exact fields and evidence

Required: goal, allowed_paths, agents, checks, search_command, confirm_command.
Optional: sealed_command, workers=2, batch_size=2, timeout=600,
evaluation_timeout=180, search_repeats=1, confirm_repeats=3, min_gain=0,
seed=0, papers=[], evaluation_files=[].
Unknown keys reject. NO sealed_repeats: sealed also uses confirm_repeats.
All evaluator file paths are absolute, unique, regular and hashed. Core freezes
all gama/rsi*.py controller sources; new continual modules must be added explicitly
to evaluation_files and must never be editable goal targets.

`run_rsi(config_or_dict, *, repo, state_dir, rounds=1, resume=False,
finalize=False, on_event=None)`. rounds is additional work per call.
Core config/repo/interpreter/controller/evaluator bytes cannot change on resume.
Parent lookup, duplicate tree rejection before evaluation, fresh serialized
confirmation, mandatory checks, sealed finalization and receipts are existing.

Core state.json keys: run_id, phase, base (commit), archive (list), champion (ID),
contract, contract_hash, next_round, reserved_proposals, pending, recent_failures.
Contract contains schema=1, repo, python=[major,minor], controller fingerprint,
normalized config, evaluation_files mapping absolute path -> sha256.
Archive entries have id, commit, tree, search, confirm and retained ref.
Seed initialization may have no champion and an empty archive.
Finalized state also has sealed_base, sealed_champion, sealed_verdict.
Measurements have score, samples, results; each result retains evaluator
details. Do not invent acceptance from missing evidence.

Core result.json has phase, base, champion (entry), ref, archive_size,
rounds_completed, reserved_proposals, sealed_verdict, sealed, patch, latest_patch,
state_dir. sealed is null before finalized, otherwise {base:measurement,
champion:measurement}. Valid verdicts: improved, regressed, not_separable.
An improved score requires min(after.samples) > max(before.samples) + min_gain.
The core finalization result must agree with state and exact candidate commit.

Per-proposal outcome under attempts/<attempt>/<candidate>/result.json has
id, round, agent, attempt, artifact_dir, parent, status, commit, tree, changed,
checks, search; rejected/duplicate variants omit irrelevant fields. A viable
outcome has a nonempty checks list and returncode=0 for each mandatory command.
Each check receipt has command, returncode, stdout, stderr, elapsed_s. Serial
confirmation also emits events and evaluator receipts. Never rewrite core
state/result/evidence or infer a successful check from absent/empty receipts.

## Existing Workspaces

`Workspaces(repo, external_root)` manages detached worktrees directly below its
owned root. Methods: create(name,parent)->Path; read_sources(path,allowed_paths)
->dict; apply_patch(path, unified_diff, allowed_paths)->list of changed paths;
commit(path,parent,message)->commit; keep(commit,name)->refs/gama-rsi/<name>;
diff(base,head)->text; assert_clean(path,expected_commit=None); remove(path);
recover()->list removed owned interrupted worktrees.

create needs a unique valid simple name. It journals creation with a nonce Git
lock; recover preflights every receipt and refuses ambiguous/manual locks.
Do not call generic prune/unlock or recover someone else's root.
apply_patch validates a normal text diff, stages only its explicitly allowed
paths and freezes approval/tree in its receipt. commit requires that exact
approval and message, uses commit-tree/update-ref, and is idempotent.
For operator regression additions, allowed paths can be exact new validation/
files; RSIConfig's separate source restrictions apply only to model mutations.
Normal Workspaces.remove uses the owned detached receipt and refuses unrelated
trees. keep refuses overwriting a different existing ref.

Git helper calls are bounded internally but not the campaign's cancel boundary.
For public branch/network commands use run_guarded with unique evidence,
explicit argv and SHA/refspec, no shell/force/reset. Ensure parent guarded
publication/STOP handling covers owned workspace operations as well.
