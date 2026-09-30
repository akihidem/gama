# Existing core reporting and ownership APIs

Exact source excerpts, not new APIs. Existing controller files remain immutable.
`rsi_runtime.load_inputs(Path(original_mission))` returns `mission`, normalized
`rsi_config`, `bridge_config`, and frozen-input `digest`. Use the effective
`rsi_config`, including its fixed bridge agents; raw rsi.json has placeholders.
`mission` includes absolute `repo`, `state_dir`, `rsi_config`, `bridge_config`.
The core lives at `Path(mission["state_dir"])/"rsi"`.

The mission CLI maps finalized core state to saturated, then exports the mission
patch; it dispatches no core worker and does not regenerate core/result.json.
The public run_rsi(..., resume=True, finalize=True) below never executes source
rounds; a finalized checkpoint skips scoring and regenerates its actual report.
Keep original config, interpreter spelling, mission path, owner locks, STOP,
frozen inputs and enclosing guardian receipt. Do not synthesize a report or
reset accounting. Coordinate with the mission owner lock and let run_rsi take
its own core lock; pre-holding that same flock would deadlock.

## gama/rsi.py:337 _run_lock

```python
@contextmanager
def _run_lock(directory: Path):
    # A kernel lock is released after SIGKILL too; a stale pid file is not a lock.
    try:
        import fcntl
    except ImportError as exc:
        raise RSIError("RSI run locking requires Linux/WSL or macOS") from exc
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a+") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RSIError(f"another RSI coordinator owns {directory}") from exc
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
```

## gama/rsi.py:688 run_rsi

```python
def run_rsi(config: RSIConfig | dict, *, repo, state_dir, rounds: int = 1,
            resume: bool = False, finalize: bool = False,
            on_event: Optional[Callable[[dict], None]] = None) -> dict:
    """Run bounded source-evolution batches, or finalize an existing run's holdout.

    ``rounds`` is additional work per invocation, not a changed frozen budget.
    ``resume`` can extend a ready run. ``finalize`` opens the sealed evaluator and
    permanently closes that run to search. The caller's checkout is never replaced.
    """
    config = RSIConfig.from_dict(asdict(config) if isinstance(config, RSIConfig) else config)
    _positive(rounds, "rounds", integer=True)
    repo, directory = Path(repo).resolve(), Path(state_dir).resolve()
    if directory == repo or repo in directory.parents:
        raise ValueError("state_dir must be outside the repository checkout")
    if finalize and config.sealed_command is None:
        raise ValueError("finalize requires a sealed_command")
    contract = {"schema": SCHEMA_VERSION, "config": asdict(config),
                "repo": str(repo), "controller": controller_fingerprint(),
                "python": [sys.version_info.major, sys.version_info.minor],
                "evaluation_files": _file_digests(config.evaluation_files)}
    with _run_lock(directory):
        checkpoint = directory / "state.json"
        if not resume:
            if checkpoint.exists():
                raise RSIError(f"checkpoint already exists in {directory}; use --resume")
            if any(p.name != "run.lock" for p in directory.iterdir()):
                raise RSIError("a new run needs an empty state directory")
            if finalize:
                raise RSIError("initialize and run search before --resume --finalize")
        session = _Session(config, repo, directory, on_event)
        try:
            _validate_evaluation_scope(config, session.workspace.repo)
            if resume:
                if not checkpoint.is_file():
                    raise RSIError(f"no checkpoint in {directory}")
                session.state = json.loads(checkpoint.read_text(encoding="utf-8"))
                if session.state.get("contract_hash") != _digest(contract):
                    raise RSIError("cannot resume with changed config, repository, controller "
                                   "or Python version; use the frozen inputs or a new run")
                phase = session.state.get("phase")
                if phase in ("finalizing", "finalized") and not finalize:
                    raise RSIError("the sealed evaluator has been opened; this run cannot search "
                                   "again (use --finalize to finish/report it)")
                recovered = [str(path) for path in session.workspace.recover()]
                fragment = _repair_events(directory)
                session.emit("resumed", next_round=session.state["next_round"],
                             interrupted_attempt=session.state.get("pending"),
                             recovered_event_fragment=fragment, recovered_worktrees=recovered)
            else:
                session.state = {
                    "contract_hash": _digest(contract), "contract": contract,
                    "run_id": uuid.uuid4().hex, "base": session.workspace.head(),
                    "phase": "initializing", "next_round": 0, "reserved_proposals": 0,
                    "archive": [], "pending": None,
                }
                session.workspace.keep(session.state["base"], f"{session.state['run_id']}/seed")
                session.save()
            if session.state["phase"] == "initializing":
                session.initialize()
            if finalize:
                if session.state["phase"] != "finalized":
                    session.finalize()
            else:
                for _ in range(rounds):
                    if not session.round():
                        break
            return session.result()
        finally:
            session.close()
```

## gama/rsi_mission.py:92 _core

```python
def _core(root: Path) -> dict | None:
    try:
        return _read(root / "rsi" / "state.json")
    except FileNotFoundError:
        return None
```

## gama/rsi_mission.py:207 _owner

```python
@contextlib.contextmanager
def _owner(root: Path):
    # Never replace this inode: independent ticks must contend on the same lock.
    with (root / "owner.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise _Overlap("mission already has a kernel owner") from exc
        identity = _identity(os.getpid())
        if identity is None:
            raise RuntimeError("cannot identify the mission owner")
        _put(root / "owner.json", identity)
        try:
            yield
        finally:
            _put(root / "owner.json", {})
```

## gama/rsi_mission.py:235 _wait_core

```python
def _wait_core(root: Path) -> None:
    # A killed outer owner releases its lock before its guardian finishes draining.
    try:
        lock = (root / "rsi" / "run.lock").open("rb")
    except FileNotFoundError:
        return
    with lock:
        deadline = time.monotonic() + 5
        while True:
            if (root / "STOP").exists():
                raise _Stopped("STOP requested")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("previous core worker has not released its lock")
                time.sleep(0.05)
```
