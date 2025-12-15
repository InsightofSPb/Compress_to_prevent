import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot delta_p95 time series for a facade")
    parser.add_argument("--timeseries-features", required=True, type=Path, help="Path to timeseries_features.parquet")
    parser.add_argument("--facade-id", required=True, type=str, help="Facade identifier to plot")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for debug plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.timeseries_features)
    subset = df[df["facade_id"].astype(str) == str(args.facade_id)].copy()

    if subset.empty:
        raise SystemExit(f"No rows found for facade_id={args.facade_id}")

    subset.sort_values("year_next", inplace=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = args.out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(subset["year_next"], subset["delta_p95"], marker="o")
    plt.xlabel("year_next")
    plt.ylabel("delta_p95")
    plt.title(f"Facade {args.facade_id}: delta_p95 over time")
    plt.grid(True)

    out_path = debug_dir / f"facade_{args.facade_id}_delta_p95.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved debug plot to {out_path}")


if __name__ == "__main__":
    main()
