The continual supervisor runs bounded improvement ticks on AWS and keeps choosing
goals after each mission finishes. Its normal between-tick phase is `scheduled`, even
with an empty queue. A mission can be `saturated`; the campaign never uses that
as its top-level phase. The procedures below do not assert any live installation,
inference, GitHub push, or completed integration checks.

`examples/continual_aws.json` supplies these AWS settings:

| Setting | Value |
| --- | --- |
| Campaign | `gama-continual-aws` |
| Repository | `/home/akhd/work/gama-rsi` |
| State directory | `/home/akhd/work/gama-rsi-runs/continuous` |
| Branch / remote | `codex/gama-parallel-rsi` / `origin` |
| Bridge | `/home/akhd/work/gama-rsi/examples/rsi_bridge_aws.json` |
| Fixed workers / `max_proposals_per_tick` | `2` / `4` |
| Cadence | `09:00` and `21:00`, `Asia/Tokyo` |
| Evaluation timeout / goal work limit | `180` seconds / `max_goal_cycles=3` |

Configure absolute, canonical paths before the first run, with state external and
disjoint from the repository and model artifacts. Use the existing bridge format
and the already checked-out, clean feature branch. Keep the Python 3.12 venv
executable spelling `/home/akhd/work/gama-rsi/.venv/bin/python`, including its symlink.
Config bytes, initial descriptors, controller files, bridge and interpreter are
frozen in campaign identity; released source HEAD is tracked separately. Unknown
controls, invalid numeric values and changes to the fixed cadence/limits are
rejected. There is no production clock, quota, STOP or verification override.
Controller sources, `gama/rsi*.py`, existing tests and historical validation
remain protected from proposed source changes.

From the repository, invoke the public CLI with an absolute config path:

```sh
cd /home/akhd/work/gama-rsi
/home/akhd/work/gama-rsi/.venv/bin/python -B -m gama.continual status --config /home/akhd/work/gama-rsi/examples/continual_aws.json
```

Replace `status` with the desired action; each invocation prints one JSON status.

| Action | Effect |
| --- | --- |
| `run` | Perform a bounded tick in the current cadence slot, including eligible recovery. A persisted blocked/stopped state requires `resume`. |
| `status` | Read status without initializing a campaign or calling the provider. |
| `stop` | Persist independent `STOP` and request cancellation. It returns before active work necessarily finishes draining. |
| `resume` | Validate inputs and recovery evidence, clear `STOP`, and immediately execute a bounded tick. Spending and goal work remain unchanged until new work is charged. |

Exit codes are `0` for a successful bounded tick or status/stop request, `2` for
runtime failure, `3` for overlap, `4` for stopped/blocked execution, and `5` for
invalid or changed controls. Inspect `phase` even after exit `0`: status/stop can
successfully report a blocked or stopped campaign.

JSON includes `phase`, `active_goal`, `queue`, `history`, `slot`, `reserved`,
`next_run`, `ownership`, `error`, `expected_head`, `publication`, `source_commit`
and, when applicable, `publication_journal`. Goal summaries include `work`,
`mission_path`, base and evidence paths. Ownership is `live`, `stale`, `none` or
`unknown`. Detailed prompts, test bodies and core archives stay in evidence files.
`slot` and `reserved` describe the last recorded ledger slot; after a clock
boundary, status can still show that slot until a tick records the new one.
`next_run` is a timezone-aware ISO cadence boundary; STOP still prevents work.

Initial goals run in this order: `meshflow-finite-scores`,
`abmcts-failure-signal`, then `tool-exit-status`. The completed historical JSON
mission is left untouched. Optional `bootstrap_history` is informational; the
AWS example starts it empty. Each goal freezes once on the initial HEAD or latest
verified release HEAD, retaining its original mission path, descriptor identity,
base and prior accepted regressions. Recovery reuses these artifacts.

Mandatory production checks run the unchanged full unittest suite (636 tests)
and all 288 unchanged JSON regression cases against `gama/_json.py`. From
`/home/akhd/work/gama-rsi`, run the exact commands in `examples/continual_aws.json`:

```sh
/home/akhd/work/gama-rsi/.venv/bin/python -B -m unittest discover -s tests -t . -q
/home/akhd/work/gama-rsi/.venv/bin/python -B validation/json_extraction.py --source gama/_json.py
```

`--source` is required: the prior invocation without it failed in argparse
before any JSON cases ran and remains a failure in its original record. Retain
the fresh output from both commands; confirm `Ran 636 tests`, JSON
`passed == total == 288` and `source_unchanged: true`, and zero exit codes.

Previously published goal descriptors become mandatory regressions in later
missions. Unmet new goal tests measure the objective and are not added to the
seed's mandatory passing checks.

A finalized, sealed improvement proceeds to verified publication, then history
records `published` and the next goal can start within the remaining slot budget.
Other sealed verdicts are recorded as `regressed` or `not_separable`; a goal that
reaches its paid work limit without completion becomes `exhausted`. Rejected,
stalled, already-met and provider-failing goals therefore have bounded work.
Ordinary provider failures retain evidence and consume work; they do not
permanently pin unrelated queued goals. Integrity/publication faults can block.

Each paid source attempt or scouting call reserves two units durably before
dispatch. The global limit is four reservation units per cadence slot across all
goals, discovery, failures and retries. It counts funded attempts, including
failed or unused proposals. A paid source dispatch also consumes one goal cycle.
Delegated source work uses one round of two proposals (`rounds_per_cycle=1`).
Interruptions can conservatively consume unused reservations; ambiguous charges
are never refunded. Any retry that might call the provider buys a fresh pair.
There is no provider/model fallback or uncharged retry. Probing status, proven
finalization-only recovery and publication reconciliation do not buy reservations.

| Work in one slot | Reservations | Resulting capacity |
| --- | --- | --- |
| Goal A attempt + A retry | `2 + 2 = 4` | Goal B waits for a later slot. |
| Goal A attempt + goal B attempt | `2 + 2 = 4` | Slot fully spent. |
| Scouting + a discovered goal attempt | `2 + 2 = 4` | Slot fully spent. |
| Scouting with no valid goal | `2` | Slot closes even with two units unused. |

Manual `run` performs one bounded tick using the current 09/21 JST slot's ledger.
Before 09:00 it uses the preceding day's 21:00 slot. Between 09:00 and 21:00 it
uses today's 09:00 slot; from 21:00 it uses today's 21:00 slot. Repeated calls in
a closed/spent slot buy nothing, though pending recovery can still reconcile.
Overlap and calls refused by STOP buy nothing. `resume`, process restarts and
goal changes preserve spending. Further paid work waits for a genuine new slot.

An empty queue triggers budgeted scouting using the durable source cursor and
all registered goal descriptors, preserving completed/exhausted IDs/fingerprints.
Validated goals enter the queue and can use remaining quota. No-result or failed
discovery closes the slot, retains evidence and advances the cursor for a later
cadence; the campaign returns to `scheduled` even when no goals were found.

Publication uses the original mission and the full durable
`goals/<key>/publication.json` journal. Only the exact verified sealed improvement
is released with its regression files. `source_commit` identifies the source
archive winner; `release_commit` identifies the release including regressions.
The publisher must report `published`, persist a complete journal and verify
`remote_commit == release_commit` on the configured branch/remote before the
supervisor advances `expected_head` or selects another goal. A local source
commit alone is insufficient. After a crash, reconciliation reuses that journal
and original mission, including when the release already reached the remote.
Never re-freeze an adopted/completed goal: added validation files change inventory.

For maintenance, use `stop`, then observe status and drain evidence before
touching state or Git. STOP persists across timer ticks and restarts until
explicit `resume`. SIGTERM/SIGINT cancellation also persists STOP when handled.
`status` and `stop` bypass provider validation, provided the campaign JSON is
readable and its repo/state paths remain valid. With a corrupt checkpoint they
report unavailable evidence; `stop` still creates STOP. Fallback zero counters
on unavailable state do not authorize fresh spending.

The state directory holds atomic, fsynced `state.json`, persistent `owner.lock`,
`owner.json`, independent `STOP`, immutable goal artifacts and action evidence.
Consult `actions/<id>/result.json`, optional `output.json`/`error.txt`, and
`actions/<id>/guard/process.json`. Owner takeover requires the enclosing guard's
post-drain completion receipt for the saved startup intent. An unlocked owner
or core lock, a PID exit, or elapsed time alone does not prove descendants drained.
Missing/incomplete receipts or containment failure block recovery. Run/resume
validate checkpoint/journals before rewriting `owner.json`; corrupt records
are refused without resetting spending.

| Condition | Recovery action |
| --- | --- |
| Ordinary provider failure | Restore availability of the configured provider and inspect action evidence. Automatic retries remain bounded by fresh pairs and goal work; a `scheduled` campaign can continue on its next eligible tick. |
| Frozen controls changed | Restore the exact frozen config, bridge, initial descriptors, controller and venv/interpreter artifacts. `resume` does not adopt a new identity. |
| Missing/corrupt checkpoint or interrupted freeze | Preserve the entire state/evidence set. Recover consistent original checkpoint and frozen artifacts with every known reservation intact; `resume` cannot reconstruct missing proof. |
| Guard completion unproven | Inspect the saved action and guardian evidence, establish descendant containment and recover authentic completion evidence. Keep STOP while proof is absent; never fabricate a receipt. |
| Dirty checkout, changed branch/HEAD or divergence | Preserve operator changes and reconcile the configured feature branch with `expected_head` and the publication journal. Only journaled releases can advance the campaign; arbitrary merges/HEAD changes cannot be adopted by retrying. |
| Git ref lock after interruption | Preserve ambiguous locks. Operator repair requires proving no live writer owns the lock; the controller does not automatically unlink an unowned lock. |

Back up consistent state and publication evidence after drain before repairs.
Restoring an older checkpoint must not discard later charges or releases. If
evidence cannot establish a consistent recovery, keep the campaign stopped.
Do not delete `owner.lock`, zero the ledger, remove state to start over, or switch
state directories to evade spending. There is no force/reset repair option;
preserve the feature branch and use normal verified publication, never force-push.
After repairing a blocked/stopped campaign, invoke `resume` with the same config
and inspect the returned phase and evidence.

After integration PASS, package and commit all approved changes, including the
README navigation link, before the first campaign run so the checkout is clean.
On AWS, pause the existing schedule and inspect the old service:

```sh
systemctl --user stop gama-rsi.timer
systemctl --user status gama-rsi.service
```

Stopping the timer does not stop an active service. Let the old service and any
manually started mission finish/drain before replacing its unit. For an existing
continual campaign use persistent `stop` first. When ready for real provider
calls and verified pushes to `origin`, install the new example under the existing
service name and retain the installed timer unchanged at 09:00/21:00 Asia/Tokyo:

```sh
install -d /home/akhd/.config/systemd/user
install -m 0644 /home/akhd/work/gama-rsi/deploy/gama-continual.service /home/akhd/.config/systemd/user/gama-rsi.service
systemctl --user daemon-reload
systemctl --user enable --now gama-rsi.timer
systemctl --user list-timers --all gama-rsi.timer
```

The timer continues to target `gama-rsi.service`. That oneshot uses the exact venv
executable and continual `run` command, `TimeoutStartSec=infinity`,
`TimeoutStopSec=20s`, `KillMode=control-group`, and `Restart=no`. Its
`SuccessExitStatus=3 4` means a successful systemd result can indicate overlap or
stopped/blocked execution. Inspect CLI JSON and
`journalctl --user -u gama-rsi.service`. Enabling the timer permits live work.
An existing STOP survives reload/restart; explicit `resume` clears it and may
immediately run the current slot.

Integration evidence must separately cover the unchanged seven campaign cases,
accepted sibling checks, legacy mission/operations checks, JSON regressions and
the full original unittest suite. Record actual installation, inference and
verified remote release results in the deployment record.
