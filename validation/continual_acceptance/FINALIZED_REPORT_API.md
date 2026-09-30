# Existing core reporting and ownership APIs

These are exact source excerpts. Existing controller files remain immutable.
The public run_rsi(..., resume=True, finalize=True) never executes source rounds;
a finalized checkpoint skips scoring and regenerates its report. The mission
CLI short-circuits finalized core states and does not regenerate that report.
Keep the original effective config, interpreter spelling, mission path, owner
locks, STOP, frozen inputs and enclosing guardian receipt. Never synthesize a
report or reset accounting; use the existing core implementation.

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

## gama/rsi_mission.py:255 _cancellation

```python
@contextlib.contextmanager
def _cancellation(root: Path):
    cancel, done = threading.Event(), threading.Event()

    def watch():
        while not done.is_set():
            if (root / "STOP").exists():
                cancel.set()
                return
            done.wait(0.05)

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    try:
        yield cancel
    finally:
        done.set()
        thread.join(timeout=0.2)
```

## gama/rsi_mission.py:447 _drive

```python
def _drive(root: Path, path: Path, state: dict, inputs: dict, action: str) -> int:
    if action == "resume":
        _control(root, False)
    if (root / "STOP").exists():
        raise _Stopped("STOP requested")
    if action == "run" and state["phase"] == "blocked":
        return 4
    _wait_core(root)
    core = _core(root)
    _reconcile(state, core)
    _settle(root, state, core)
    if action == "resume":
        state["phase"], state["error"] = "waiting", None
    elif state["phase"] == "blocked":
        _put(root / "state.json", state)
        return 4
    mission = inputs["mission"]
    terminal = core is not None and core.get("phase") in ("finalizing", "finalized")
    if not state["cycle_open"] and action == "run" and not terminal:
        state.update(cycle_open=True, reserved_baseline=_count(core, "reserved_proposals"),
                     cycle_reservations=0, round_baseline=_count(core, "next_round"),
                     round_limit=mission["rounds_per_cycle"], cycle_healthy=False, inflight=None,
                     cycle_champion=state["champion"]["commit"] if state["champion"] else None,
                     last_outcome=None, error=None)
        state["counts"]["cycles"] += 1
    state["phase"] = "active" if state["cycle_open"] or terminal else "waiting"
    _put(root / "state.json", state)
    with _cancellation(root) as cancel:
        while True:
            inputs = _inputs(path, state["digest"])
            core = _core(root)
            _reconcile(state, core)
            if cancel.is_set() or (root / "STOP").exists():
                raise _Stopped("STOP requested")
            if state["phase"] == "blocked":
                _put(root / "state.json", state)
                return 0
            if core is not None and core.get("phase") == "finalized":
                state.update(phase="saturated", cycle_open=False, last_outcome="saturated", inflight=None)
                kind = None
            else:
                kind = _step(state, core, mission)
            if kind is None:
                if state["phase"] != "saturated":
                    state["last_outcome"] = state["last_outcome"] or ("exhausted" if state["cycle_open"] else None)
                    state.update(phase="waiting", cycle_open=False, inflight=None)
                _export(root, state, mission)
                _put(root / "state.json", state)
                return 0
            failure = _dispatch(root, path, state, inputs, kind, cancel)
            core = _core(root)
            _reconcile(state, core)
            completed = _settle(root, state, core)
            _put(root / "state.json", state)
            if cancel.is_set() or (root / "STOP").exists():
                raise _Stopped("STOP requested")
            _inputs(path, state["digest"])
            # A committed checkpoint wins even if worker stdout/export failed.
            if not completed:
                raise RuntimeError(failure or "core worker returned without checkpoint progress")
```

## gama/rsi_mission.py:553 _view

```python
def _view(root: Path) -> dict:
    state = _state(root)
    _reconcile(state, _core(root))
    result = {key: state[key] for key in _PUBLIC}
    result["ownership"] = _ownership(root)
    if (root / "STOP").exists():
        result["phase"] = "stopped"
    return result
```

## gama/rsi_mission.py:563 main

```python
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "status", "stop", "resume"))
    parser.add_argument("--mission", type=Path, required=True)
    args = parser.parse_args(argv)
    root, error, code = None, None, 0
    try:
        path = args.mission.resolve()
        root = _root(path)
        if args.action != "status":
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if args.action == "stop":
            _control(root, True)
        elif args.action != "status":
            code = _execute(root, path, args.action)
    except _Overlap as exc:
        code, error = 3, str(exc)
    except _Changed as exc:
        code, error = 5, str(exc)
    except (Exception, KeyboardInterrupt) as exc:
        code, error = 2, f"{type(exc).__name__}: {exc}"
    result = {key: _empty()[key] for key in _PUBLIC}
    result["ownership"] = "none"
    if root is not None:
        try:
            result = _view(root)
        except Exception as exc:
            error = error or f"{type(exc).__name__}: {exc}"
            if code == 0 and args.action != "stop":
                code = 2
            if (root / "STOP").exists():
                result["phase"] = "stopped"
    if error:
        result["error"] = error
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return code
```
