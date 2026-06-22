"""Append-only JSONL ledger for benchmark runs (vendored, stdlib-only)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class LogRecord:
    run_id: str
    task_id: str
    task_type: str
    selected_model: str
    review_score: Optional[float] = None
    actual_cost: Optional[float] = None
    elapsed_seconds: Optional[float] = None
    judge_decision: Optional[str] = None
    failure_reason: Optional[str] = None
    ts: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExecutionLogger:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: LogRecord) -> LogRecord:
        if record.ts is None:
            record.ts = _now_iso()
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        return record

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out
