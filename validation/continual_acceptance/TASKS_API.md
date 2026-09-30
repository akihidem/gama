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
