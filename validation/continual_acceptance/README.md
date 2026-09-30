These behavioral acceptance inputs were authored independently before the
continual controller implementation. They use temporary Git repositories, local
bare remotes, real RSI worktrees/evaluation, and deliberately simulated model
adapters. They do not call actual Astra or Claude.

Run from the implementation checkout with Python 3.12:

```bash
.venv/bin/python -B validation/continual_acceptance/check_tasks.py
.venv/bin/python -B validation/continual_acceptance/check_discovery.py
.venv/bin/python -B validation/continual_acceptance/check_publish.py
.venv/bin/python -B validation/continual_acceptance/check_campaign.py
```

The tasks checks bind external scoring scripts to candidate source and freeze
all regression inputs. Discovery covers two parallel scouts, strict replies,
baseline validation, preserved evidence, source rotation and descendant cleanup.
Publication uses real finalized core results and exercises exact regression
retention, branch/remote protection and interrupted publication. Campaign checks
compose real source improvement across goals with global quotas, cadence,
STOP, retry and checkpoint handling.

The initial goals in `examples/continual_goals/` are public executable behavior
contracts for three independently reproduced bugs. Each has search, confirmation
and finalization cases. They are not a hidden generalization benchmark.

`CORE_API.md` is the interface summary supplied to implementers. The historical
`frozen_files.json` and the Astra run's acceptance wrapper identify the input
bytes used for this implementation review; later intentionally adopted source
improvements can differ from that historical baseline.
