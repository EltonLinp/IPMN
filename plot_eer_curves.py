from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


PATTERN = re.compile(
    r"\[Val\]\s+Epoch\s+(?P<epoch>\d+)\s+EER\s+final=(?P<final>\d+\.\d+)"
    r"\s+audio=(?P<audio>\d+\.\d+)\s+video=(?P<video>\d+\.\d+)"
    r"(?:\s+sync=(?P<sync>\d+\.\d+))?"
)


def parse_log(path: Path) -> dict[str, list[float]]:
    metrics = {"epoch": [], "final": [], "audio": [], "video": [], "sync": []}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = PATTERN.search(line)
        if not match:
            continue
        metrics["epoch"].append(int(match["epoch"]))
        metrics["final"].append(float(match["final"]))
        metrics["audio"].append(float(match["audio"]))
        metrics["video"].append(float(match["video"]))
        sync_val = match.group("sync")
        metrics["sync"].append(float(sync_val) if sync_val is not None else float("nan"))
    if not metrics["epoch"]:
        raise ValueError(f"No validation entries found in {path}")
    return metrics


def plot_metrics(metrics: dict[str, list[float]], output: Path, title: str | None = None) -> None:
    plt.figure(figsize=(8, 5))
    epochs = metrics["epoch"]
    plt.plot(epochs, metrics["final"], label="Final", linewidth=2)
    plt.plot(epochs, metrics["audio"], label="Audio")
    plt.plot(epochs, metrics["video"], label="Video")
    if any(not (val != val) for val in metrics["sync"]):  # check for non-NaN
        plt.plot(epochs, metrics["sync"], label="Sync")
    plt.xlabel("Epoch")
    plt.ylabel("EER")
    plt.ylim(bottom=0)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    if title:
        plt.title(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot validation EER curves from training log.")
    parser.add_argument("--log", type=Path, required=True, help="Path to training log containing [Val] lines.")
    parser.add_argument("--output", type=Path, default=Path("eer_curves.png"), help="Output figure path.")
    parser.add_argument("--title", type=str, default=None, help="Optional plot title.")
    args = parser.parse_args()

    metrics = parse_log(args.log)
    plot_metrics(metrics, args.output, args.title)
    best_final = min(zip(metrics["final"], metrics["epoch"]), key=lambda item: item[0])
    print(f"Saved plot to {args.output} (best final EER {best_final[0]:.4f} at epoch {best_final[1]}).")


if __name__ == "__main__":
    main()
