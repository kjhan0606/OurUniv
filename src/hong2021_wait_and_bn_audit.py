#!/usr/bin/env python
"""Wait for a Hong training tmux job and run the frozen BN audits."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def completed_epoch(history_path: Path) -> int:
    if not history_path.is_file():
        return 0
    history = json.loads(history_path.read_text())
    return int(history[-1]["epoch"]) if history else 0


def tmux_alive(session: str) -> bool:
    result = subprocess.run(
        ["tmux", "list-sessions", "-F", "#{session_name}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    return session in result.stdout.splitlines()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-tmux", required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--expected-epoch", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    history_path = args.training_output / "history.json"
    while True:
        epoch = completed_epoch(history_path)
        print(
            f"[wait] completed_epoch={epoch}/{args.expected_epoch} "
            f"training_tmux_alive={tmux_alive(args.training_tmux)}",
            flush=True,
        )
        if epoch >= args.expected_epoch:
            break
        if not tmux_alive(args.training_tmux):
            raise SystemExit(
                "training tmux ended before the expected epoch; BN audit aborted"
            )
        time.sleep(args.poll_seconds)

    args.audit_output.mkdir(parents=True, exist_ok=True)
    audit_script = Path(__file__).with_name("hong2021_bn_audit.py")
    checkpoints = (
        ("minimum_validation", args.training_output / "minimum_validation_loss.pt"),
        ("last_epoch", args.training_output / "last_epoch.pt"),
    )
    reports = []
    for label, checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise SystemExit(f"missing completed checkpoint: {checkpoint}")
        destination = args.audit_output / f"{label}.json"
        command = [
            sys.executable,
            str(audit_script),
            "--train",
            str(args.train),
            "--validation",
            str(args.validation),
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(destination),
            "--batch",
            str(args.batch),
            "--workers",
            str(args.workers),
            "--device",
            args.device,
        ]
        print(f"[audit:start] {label} checkpoint={checkpoint}", flush=True)
        subprocess.run(command, check=True)
        report = json.loads(destination.read_text())
        reports.append(
            {
                "label": label,
                "checkpoint_epoch": report["checkpoint"]["epoch"],
                "diagnosis": report["diagnosis"],
            }
        )
        print(
            f"[audit:complete] {label} diagnosis="
            f"{json.dumps(report['diagnosis'], sort_keys=True)}",
            flush=True,
        )
    summary = {
        "schema": "hong2021-posttrain-bn-audit-v1",
        "training_output": str(args.training_output),
        "completed_epoch": completed_epoch(history_path),
        "reports": reports,
    }
    (args.audit_output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
