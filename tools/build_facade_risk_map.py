import argparse
from pathlib import Path
import os
import pandas as pd
import numpy as np
import cv2


def parse_args():
    ap = argparse.ArgumentParser("Build facade-level risk heatmaps from patch risk scores.")
    ap.add_argument("--patch-changes", required=True, type=Path, help="patch_changes.csv or .parquet")
    ap.add_argument("--patch-manifest", required=True, type=Path, help="patch_manifest.csv from patches_ref")
    ap.add_argument("--temporal-manifest", required=True, type=Path, help="CSV with facade_id,year,image_path (full facade image)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--risk-column", default="risk_total", type=str)
    ap.add_argument("--time-agg", choices=("latest", "max"), default="latest",
                    help="How to aggregate patch risk over time steps.")
    ap.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha (0..1)")
    ap.add_argument("--normalize", choices=("per_facade", "global", "none"), default="per_facade")
    return ap.parse_args()


def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    changes = read_any(args.patch_changes)
    mani = pd.read_csv(args.patch_manifest)
    tman = pd.read_csv(args.temporal_manifest)

    # 1) pick patch risk per (facade_id, patch_id)
    need_cols = {"facade_id", "patch_id", args.risk_column}
    missing = need_cols - set(changes.columns)
    if missing:
        raise ValueError(f"patch_changes missing columns: {missing}")

    if args.time_agg == "latest":
        # latest by (year_next, step_idx) if present, else by step_idx
        sort_cols = [c for c in ["year_next", "step_idx"] if c in changes.columns]
        if not sort_cols:
            sort_cols = ["patch_id"]  # fallback (stable but arbitrary)
        changes_sorted = changes.sort_values(["facade_id", "patch_id"] + sort_cols)
        risk_df = changes_sorted.groupby(["facade_id", "patch_id"], as_index=False).tail(1)
    else:
        risk_df = changes.groupby(["facade_id", "patch_id"], as_index=False)[args.risk_column].max()

    risk_df = risk_df[["facade_id", "patch_id", args.risk_column]].rename(columns={args.risk_column: "risk"})

    # 2) get patch coords in ref-frame (unique per facade_id,patch_id)
    # coords are identical across years; take first row per (facade,patch)
    coord_cols = ["patch_x0", "patch_y0", "patch_x1", "patch_y1", "ref_year"]
    missing = set(coord_cols) - set(mani.columns)
    if missing:
        raise ValueError(f"patch_manifest missing columns: {missing}")

    coords = mani.sort_values(["facade_id", "patch_id", "year"]).groupby(["facade_id", "patch_id"], as_index=False).head(1)
    coords = coords[["facade_id", "patch_id"] + coord_cols]

    df = coords.merge(risk_df, on=["facade_id", "patch_id"], how="inner")
    if df.empty:
        raise ValueError("No overlap between patch_manifest and patch_changes on (facade_id, patch_id).")

    # 3) map facade_id -> ref image path
    if "image_path" not in tman.columns:
        raise ValueError("temporal_manifest must contain image_path (full facade image path).")
    if "year" not in tman.columns:
        raise ValueError("temporal_manifest must contain year column.")

    # choose ref image per facade using ref_year from coords
    ref_year_by_facade = coords.groupby("facade_id")["ref_year"].max().to_dict()

    out_rows = []
    global_max = float(df["risk"].max()) if len(df) else 1.0
    global_max = global_max if global_max > 0 else 1.0

    for facade_id, g in df.groupby("facade_id"):
        ref_year = int(ref_year_by_facade.get(facade_id, g["ref_year"].iloc[0]))
        # lookup full facade image path
        sel = tman[(tman["facade_id"].astype(str) == str(facade_id)) & (tman["year"].astype(int) == ref_year)]
        if sel.empty:
            # fallback: any year row for that facade
            sel = tman[tman["facade_id"].astype(str) == str(facade_id)]
        if sel.empty:
            print(f"[warn] no image in temporal manifest for {facade_id}; skipping")
            continue
        img_path = Path(sel.iloc[0]["image_path"])
        if not img_path.exists():
            print(f"[warn] missing image file {img_path}; skipping {facade_id}")
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[warn] failed to read {img_path}; skipping {facade_id}")
            continue
        H, W = img.shape[:2]

        acc = np.zeros((H, W), dtype=np.float32)
        cnt = np.zeros((H, W), dtype=np.float32)

        for _, r in g.iterrows():
            x0, y0, x1, y1 = int(r.patch_x0), int(r.patch_y0), int(r.patch_x1), int(r.patch_y1)
            x0 = max(0, min(W, x0)); x1 = max(0, min(W, x1))
            y0 = max(0, min(H, y0)); y1 = max(0, min(H, y1))
            if x1 <= x0 or y1 <= y0:
                continue
            acc[y0:y1, x0:x1] += float(r.risk)
            cnt[y0:y1, x0:x1] += 1.0

        risk_map = np.zeros((H, W), dtype=np.float32)
        m = cnt > 0
        risk_map[m] = acc[m] / cnt[m]

        if args.normalize == "per_facade":
            denom = float(risk_map.max()) if float(risk_map.max()) > 0 else 1.0
        elif args.normalize == "global":
            denom = global_max
        else:
            denom = 1.0

        norm = np.clip(risk_map / denom, 0.0, 1.0)
        heat_u8 = (norm * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 1.0 - args.alpha, heat_color, args.alpha, 0)

        out_facade_dir = args.out_dir / str(facade_id)
        out_facade_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(out_facade_dir / f"risk_heatmap_{args.time_agg}.png"), heat_color)
        cv2.imwrite(str(out_facade_dir / f"risk_overlay_{args.time_agg}.png"), overlay)

        out_rows.append({
            "facade_id": facade_id,
            "ref_year": ref_year,
            "n_patches": int(len(g)),
            "mean_risk": float(g["risk"].mean()),
            "max_risk": float(g["risk"].max()),
            "image_path": str(img_path),
            "out_dir": str(out_facade_dir),
        })

    pd.DataFrame(out_rows).sort_values(["max_risk"], ascending=False).to_csv(args.out_dir / f"facade_risk_summary_{args.time_agg}.csv", index=False)
    print(f"[OK] Wrote {len(out_rows)} facade maps to {args.out_dir}")
    print(f"[OK] Summary: {args.out_dir / f'facade_risk_summary_{args.time_agg}.csv'}")


if __name__ == "__main__":
    main()
