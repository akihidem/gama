# Bounded AWS RSI operations

The checkout is `/home/akhd/work/gama-rsi` on `codex/gama-parallel-rsi`.
Use its Python 3.12 interpreter at `/home/akhd/work/gama-rsi/.venv/bin/python` for every mission command, test and scorer.

The [first real model cycle](rsi_first_live.md) completed on 2026-09-30 with two
proposals and a verified JSON parser improvement. The user timer is enabled for
09:00 and 21:00 Asia/Tokyo. This mission has reached both score ceilings and is
`saturated`: later ticks check its completed state without further model calls.
A different improvement goal uses a new mission with separate state and evidence.

The mission is `examples/rsi_aws_mission.json`; metadata lives in `/home/akhd/work/gama-rsi-runs/json-extraction`.
Its core checkpoint is exclusively under `json-extraction/rsi/`. Do not prepopulate that directory or delete `owner.lock` to bypass ownership.
Proposal evidence lives beneath `/home/akhd/work/gama-rsi-runs/json-extraction-models`, with separate builder and reviewer records.
Mission worker requests, events, results and guarded process logs live in `json-extraction/workers/`; exported patches live in `json-extraction/exports/`.

Each cycle uses `astra-a` and `astra-b`, two proposals per batch, at most two completed rounds and four durable reservations including retries.
The ready seed is inspected before any proposal. Search and confirmation ceilings are inspected again after each committed round.
Counts include reserved proposals even when a process dies before a model responds. Recovery preserves the original cycle baseline.
A normal completed cycle enters `waiting`; the next ordinary `run` starts a new bounded cycle.

After independent integration PASS, the parent operator runs one real verification cycle:

```sh
cd /home/akhd/work/gama-rsi
/home/akhd/work/gama-rsi/.venv/bin/python --version
/home/akhd/work/gama-rsi/.venv/bin/python -B -m gama.rsi_mission run --mission /home/akhd/work/gama-rsi/examples/rsi_aws_mission.json
/home/akhd/work/gama-rsi/.venv/bin/python -B -m gama.rsi_mission status --mission /home/akhd/work/gama-rsi/examples/rsi_aws_mission.json
```

Inspect `phase`, `cycle_open`, `cycle_reservations`, `counts`, `champion`, `ref`, `patch` and the model evidence before enabling the timer.
The runner exports the committed champion's diff from the original seed. It leaves the caller's branch, tracked edits and untracked files intact.

To cancel active work and prevent later scheduled ticks from buying proposals:

```sh
/home/akhd/work/gama-rsi/.venv/bin/python -B -m gama.rsi_mission stop --mission /home/akhd/work/gama-rsi/examples/rsi_aws_mission.json
/home/akhd/work/gama-rsi/.venv/bin/python -B -m gama.rsi_mission status --mission /home/akhd/work/gama-rsi/examples/rsi_aws_mission.json
```

STOP is a separate persistent marker. It cancels the active guarded worker and survives process restarts and concurrent state saves.
`status` distinguishes a live kernel owner from stale PID/starttime metadata; a stale active phase does not mean work is still running.

After inspecting a stop, interruption or blocked cycle, explicitly continue:

```sh
/home/akhd/work/gama-rsi/.venv/bin/python -B -m gama.rsi_mission resume --mission /home/akhd/work/gama-rsi/examples/rsi_aws_mission.json
```

`resume` clears STOP/blocking and continues an open cycle within its remaining reservations. All rejected generation/evaluation proposals block further ordinary ticks.
A closed cycle's `resume` buys no proposals. Duplicates and viable nonwinners alone do not block operation.
Saturation requires both ceilings and finalization; finalized search never reopens, including on later `run` or `resume` commands.
Checkpointed sealed measurements are reused during interrupted finalization. An external measurement interrupted before its checkpoint may execute again.

Frozen inputs include complete mission/config contents, their byte hashes, controller contents and interpreter identity.
Input changes reject continuation before inference or finalization, including changes made after saturation.
Use a new mission and separate state/artifact locations for an intentionally different experiment; preserve the old records.

Only after integration PASS and the first manual verification does the parent operator install and activate the user units:

```sh
install -d -m 700 ~/.config/systemd/user
install -m 644 deploy/gama-rsi.service deploy/gama-rsi.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now gama-rsi.timer
systemctl --user list-timers gama-rsi.timer
```

The timer uses exactly 09:00 and 21:00 Asia/Tokyo. `Persistent=false` skips missed ticks.
The user manager must remain available for unattended operation. Builders and acceptance checks do not install units or call live models.
The oneshot has a ten-hour wall limit, a ten-second stop limit and `KillMode=control-group`; individual core workers also have guarded execution deadlines.
Use the mission `stop` command to cancel active work. `systemctl --user stop gama-rsi.timer` only prevents future timer activations.

Interrupted worktree creation carries a unique marker in both its durable receipt
and Git's lock reason. Recovery validates that ownership before releasing the
lock, including when checkout has already written its index. Locks on completed
worktrees remain protected. Seed checks can resume before a champion exists, and
the bridge's deadline includes waiting for input to finish.

Exit codes: 0 success, 2 invalid input/runtime failure, 3 overlapping owner, 4 stopped/blocked ordinary tick, 5 changed frozen inputs.
Source adoption and GitHub publishing remain separate explicit operator actions; the mission never switches branches, adopts a patch or pushes.
