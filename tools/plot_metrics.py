"""Plot training curves from metrics produced by tools/finetune.py."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def load_metrics(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_loss(records: List[Dict], out_dir: Path) -> None:
    epochs = [r["epoch"] for r in records]
    train_loss = [r.get("train_loss") for r in records]
    val_loss = [r.get("val_loss") for r in records]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="train_loss")
    if any(val is not None for val in val_loss):
        plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "loss.png")
    plt.close()


def _collect_metric_series(records: List[Dict], container_key: str, metric_key: str) -> Dict[str, List[float]]:
    names = set()
    for record in records:
        metrics = record.get("metrics", {})
        container = metrics.get(container_key, {})
        names.update(container.keys())

    series: Dict[str, List[float]] = {name: [] for name in sorted(names)}
    for record in records:
        metrics = record.get("metrics", {})
        container = metrics.get(container_key, {})
        for name in series:
            metric_values = container.get(name, {})
            if isinstance(metric_values, dict):
                series[name].append(metric_values.get(metric_key))
            else:
                series[name].append(metric_values)
    return series


def plot_metric_groups(records: List[Dict], out_dir: Path, container_key: str, title_prefix: str) -> None:
    for metric_key in ("iou", "f1", "accuracy"):
        series = _collect_metric_series(records, container_key, metric_key)
        if not series:
            continue

        plt.figure(figsize=(10, 6))
        for name, values in series.items():
            plt.plot([r["epoch"] for r in records], values, label=name)
        plt.xlabel("Epoch")
        plt.ylabel(metric_key.upper())
        plt.title(f"{title_prefix} — {metric_key.upper()}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / f"{container_key}_{metric_key}.png")
        plt.close()


def plot_overall(records: List[Dict], out_dir: Path) -> None:
    metric_keys = ("mIoU", "mF1", "mAcc")
    epochs = [r["epoch"] for r in records]
    values: Dict[str, List[float]] = {name: [] for name in metric_keys}

    for record in records:
        metrics = record.get("metrics", {})
        for key in metric_keys:
            values[key].append(metrics.get(key))

    plt.figure(figsize=(8, 5))
    for label, vals in values.items():
        plt.plot(epochs, vals, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Overall validation metrics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "overall_metrics.png")
    plt.close()


def plot_all(metrics_path: Path, output_dir: Path) -> None:
    records = load_metrics(metrics_path)
    if not records:
        raise ValueError("No metrics found in the provided file")

    out_dir = ensure_output_dir(output_dir)
    plot_loss(records, out_dir)
    plot_metric_groups(records, out_dir, "class_metrics", "Per-class metrics")
    plot_metric_groups(records, out_dir, "group_metrics", "Grouped metrics")
    plot_overall(records, out_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot curves from metrics.jsonl")
    parser.add_argument("--metrics-file", required=True, type=Path, help="Path to metrics.jsonl")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store generated plots (defaults to metrics file folder / plots)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_path: Path = args.metrics_file
    output_dir = args.output_dir or metrics_path.parent / "plots"
    plot_all(metrics_path, output_dir)


if __name__ == "__main__":
    main()
