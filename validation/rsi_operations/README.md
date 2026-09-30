# RSI operations acceptance

These offline fixtures are frozen design inputs for the Astra Loop implementation
of the live RSI bridge and mission runner. They use real temporary Git repositories,
real child processes, and an explicit fake adapter; they never invoke a model or
manage the user's services.

Run the component and recovery scripts from the repository root with Python 3.12
and `-B`. For example:

```sh
.venv/bin/python -B validation/rsi_operations/check_mission.py
```

The mission check defaults to the real bridge plus the fake external adapter.
`--component` supplies a bridge interface fixture only in the temporary repository,
so the bridge and mission implementations can proceed in parallel.

The supplied Astra Loop plans verified each fixture's SHA-256 before execution.
The original scripts and their checks remain unchanged. The final operating
composition passed all twelve planned checks, including the 636-test suite.

`check_integrity.py` is a historical construction gate against commit
`d4497d36d17182ad4c8e832c3ae6814b3aaea486`. Its byte comparison predates the
authorized workspace recovery fix and the measured parser improvement, so it is
not a current-HEAD regression command. Later plans froze every non-target input
at their own input commit instead; the original historical script is preserved.

The recovery fixtures cover unfinished and completed-index Git creation, manual
locks, interrupted seed initialization, and input deadlines. They use temporary
repositories and fake model adapters. The separate
[`json_extraction.py`](../json_extraction.py) validates the real parser with the
same 288 generated cases prepared before the first model proposals:

```sh
.venv/bin/python -B validation/json_extraction.py --source gama/_json.py
```
