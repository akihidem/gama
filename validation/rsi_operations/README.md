# RSI operations acceptance

These offline fixtures are frozen design inputs for the Astra Loop implementation
of the live RSI bridge and mission runner. They use real temporary Git repositories,
real child processes, and an explicit fake adapter; they never invoke a model or
manage the user's services.

After implementation, run the `check_*.py` scripts from the repository root with
Python 3.12 and `-B`. For example:

```sh
.venv/bin/python -B validation/rsi_operations/check_mission.py
```

The mission check defaults to the real bridge plus the fake external adapter.
`--component` supplies a bridge interface fixture only in the temporary repository,
so the bridge and mission implementations can proceed in parallel.

The supplied Astra Loop plan verifies the SHA-256 of every fixture before executing
it. `check_integrity.py` also compares the existing engine, tests, JSON extractor,
and fixed evaluator against commit `d4497d36d17182ad4c8e832c3ae6814b3aaea486`.
The implementation must not edit these acceptance files.
