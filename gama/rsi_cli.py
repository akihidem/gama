"""Command-line entry point for persistent source evolution."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .rsi import run_rsi


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _progress(event: dict) -> None:
    kind = event["event"]
    if kind == "seed":
        text = f"seed search={event['search']:.4f}, confirm={event['confirm']:.4f}"
    elif kind == "round_start":
        text = f"round {event['round'] + 1}: {len(event['jobs'])} proposals, {event['workers']} workers"
    elif kind == "candidate":
        text = f"{event['id']} ({event['agent']}): {event['status']}"
        if "search" in event:
            text += f", search={event['search']['score']:.4f}"
        if event.get("error"):
            text += f": {event['error'][:500]}"
    elif kind == "confirmation":
        text = f"{event['candidate']}: {'promoted' if event['accepted'] else 'not promoted'}; {event['reason']}"
    elif kind == "round_complete":
        text = f"round {event['round'] + 1} saved: champion={event['champion']}, archive={event['archive_size']}"
    elif kind in {"confirmation_failed", "cleanup_failed", "stop", "finalized", "resumed"}:
        text = json.dumps(event, ensure_ascii=False)
    else:
        return
    sys.stderr.write(f"[gama rsi] {text}\n")


def cmd_rsi(args: argparse.Namespace) -> int:
    try:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        result = run_rsi(
            config, repo=args.repo, state_dir=args.state_dir, rounds=args.rounds,
            resume=args.resume, finalize=args.finalize, on_event=_progress,
        )
    except KeyboardInterrupt:
        sys.stderr.write("[gama rsi] interrupted; completed rounds are checkpointed. "
                         "Use the same config and --resume.\n")
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        sys.stderr.write(f"[gama rsi] {type(exc).__name__}: {exc}\n")
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "rsi", help="evolve source in parallel Git worktrees with fixed external evaluation",
    )
    parser.add_argument("--config", required=True, help="RSI JSON configuration")
    parser.add_argument("--repo", default=".", help="Git checkout whose committed HEAD is the seed")
    parser.add_argument("--state-dir", required=True,
                        help="persistent run directory outside the repository checkout")
    parser.add_argument("--rounds", type=_positive_int, default=1,
                        help="additional bounded search batches in this invocation (default: 1)")
    parser.add_argument("--resume", action="store_true",
                        help="continue from the last completed round with identical inputs")
    parser.add_argument("--finalize", action="store_true",
                        help="with --resume, open the sealed evaluator and permanently end search")
    parser.set_defaults(func=cmd_rsi)
