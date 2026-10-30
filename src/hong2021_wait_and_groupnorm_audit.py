#!/usr/bin/env python
"""Wait for the GroupNorm pilot, then run its full frozen-weight evaluation."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from hong2021_wait_and_bn_audit import completed_epoch, tmux_alive


def run(command: list[str]) -> None:
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def unique_checkpoints(training_output: Path) -> list[tuple[str, Path]]:
    candidates = (
        ("minimum_validation", training_output / "minimum_validation_loss.pt"),
        ("minimum_training", training_output / "minimum_training_loss.pt"),
        ("last_epoch", training_output / "last_epoch.pt"),
    )
    selected: list[tuple[str, Path]] = []
    for label, path in candidates:
        if not path.is_file():
            raise SystemExit(f"missing completed checkpoint: {path}")
        if any(os.path.samefile(path, prior) for _, prior in selected):
            print(f"[skip] {label} is the same file as an earlier candidate", flush=True)
            continue
        selected.append((label, path))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-tmux", required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--evaluation-output", type=Path, required=True)
    parser.add_argument("--expected-epoch", type=int, default=20)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    history_path = args.training_output / "history.json"
    while tmux_alive(args.training_tmux):
        print(
            f"[wait] completed_epoch={completed_epoch(history_path)}/"
            f"{args.expected_epoch} training_tmux_alive=True",
            flush=True,
        )
        time.sleep(args.poll_seconds)
    epoch = completed_epoch(history_path)
    print(f"[wait] completed_epoch={epoch} training_tmux_alive=False", flush=True)
    if epoch < args.expected_epoch:
        raise SystemExit("training ended before the expected epoch")

    checkpoints = unique_checkpoints(args.training_output)
    args.evaluation_output.mkdir(parents=True, exist_ok=True)
    audit_output = args.evaluation_output / "inference_audit"
    audit_output.mkdir(parents=True, exist_ok=True)
    audit_script = Path(__file__).with_name("hong2021_groupnorm_audit.py")
    for label, checkpoint in checkpoints:
        run(
            [
                sys.executable,
                str(audit_script),
                "--train",
                str(args.train),
                "--validation",
                str(args.validation),
                "--checkpoint",
                str(checkpoint),
                "--out",
                str(audit_output / f"{label}.json"),
                "--batch",
                str(args.batch),
                "--workers",
                str(args.workers),
                "--device",
                args.device,
            ]
        )

    candidate_output = args.evaluation_output / "candidates"
    evaluate_command = [
        sys.executable,
        str(Path(__file__).with_name("hong2021_evaluate.py")),
        "--validation",
        str(args.validation),
    ]
    for label, checkpoint in checkpoints:
        evaluate_command.extend(("--checkpoint", f"{label}={checkpoint}"))
    evaluate_command.extend(
        (
            "--out",
            str(candidate_output),
            "--history",
            str(history_path),
            "--batch",
            str(args.batch),
            "--workers",
            str(args.workers),
            "--device",
            args.device,
        )
    )
    run(evaluate_command)
    metrics = json.loads((candidate_output / "metrics.json").read_text())
    selected = metrics["selection"]["selected_label"]

    spectral_output = args.evaluation_output / "spectral"
    run(
        [
            sys.executable,
            str(Path(__file__).with_name("hong2021_spectral_diagnostics.py")),
            "--data",
            str(args.validation),
            "--predictions",
            str(candidate_output / "predictions.h5"),
            "--label",
            selected,
            "--out",
            str(spectral_output),
        ]
    )
    run(
        [
            sys.executable,
            str(Path(__file__).with_name("hong2021_paper_visual.py")),
            "--data",
            str(args.validation),
            "--predictions",
            str(candidate_output / "predictions.h5"),
            "--label",
            selected,
            "--out",
            str(args.evaluation_output / "paper_visual.png"),
        ]
    )
    density_stratified = args.evaluation_output / "density_stratified.json"
    run(
        [
            sys.executable,
            str(
                Path(__file__).with_name(
                    "hong2021_density_stratified_metrics.py"
                )
            ),
            "--data",
            str(args.validation),
            "--predictions",
            str(candidate_output / "predictions.h5"),
            "--label",
            selected,
            "--out",
            str(density_stratified),
        ]
    )
    summary = {
        "schema": "hong2021-groupnorm-posttrain-evaluation-v1",
        "completed_epoch": epoch,
        "unique_checkpoint_labels": [label for label, _ in checkpoints],
        "selected_label": selected,
        "selected_epoch": metrics["selection"]["selected_epoch"],
        "candidate_metrics": str(candidate_output / "metrics.json"),
        "spectral_metrics": str(spectral_output / "spectral_metrics.json"),
        "inference_audits": str(audit_output),
        "density_stratified_metrics": str(density_stratified),
    }
    (args.evaluation_output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
