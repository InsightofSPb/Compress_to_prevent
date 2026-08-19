"""Dataset-independent command line runner for genuine stock OVS LPOSS."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import yaml
from omegaconf import OmegaConf

from models import build_model
from .ontology import load_ontology
from .stock_lposs import StockOVSEngine, UpstreamGraphPropagator, metadata_dict, preflight
from .vocabulary import RuntimeClass, build_prototypes, heritage_runtime_vocabulary

UPSTREAM_REPOSITORY = "https://github.com/vladan-stojnic/LPOSS"
# Inspected source is the LPOSS snapshot from which this repository was forked. Network was
# unavailable in the implementation environment; see the focused documentation.
UPSTREAM_REFERENCE_COMMIT = "repository-vendored-snapshot"


class LPOSSStockAdapter:
    def __init__(self, model): self.model = model
    def clip_dense(self, image):
        if hasattr(self.model, "get_clip_features"):
            return self.model.get_clip_features(image)[0]
        return self.model(image, return_feat=True)[1]
    def dino_dense(self, image):
        feats, (h, w) = self.model.get_dino_features(image)
        return feats.reshape(feats.shape[0], feats.shape[1], h, w)
    def encode_text(self, prompts):
        clip = getattr(self.model, "clip_backbone", self.model)
        tokens = torch.cat([clip.decode_head.tokenizer(p) for p in prompts]).to(next(clip.parameters()).device)
        return clip.backbone.encode_text(tokens)


def _args():
    p = argparse.ArgumentParser(description="Stock open-vocabulary MaskCLIP/LPOSS inference")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image")
    src.add_argument("--image-dir")
    src.add_argument("--manifest")
    p.add_argument("--model-config", required=True)
    p.add_argument("--vocabulary", default=str(Path(__file__).parent / "configs/heritage_vocab.yaml"))
    p.add_argument("--extra-concepts", help="YAML list of {name,prompts,semantic_id:null}")
    p.add_argument("--ornament-contrast", default=str(Path(__file__).parent / "configs/ornament_contrast_v1.yaml"))
    p.add_argument("--mode", choices=("maskclip_raw", "lposs", "lposs_plus"), required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--inference", choices=("whole", "slide"), default="whole")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--save-scores", action="store_true")
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--ornament-threshold", type=float)
    p.add_argument("--ledger-dir")
    return p.parse_args()


def _hash(path): return sha256(Path(path).read_bytes()).hexdigest()


def _inputs(args):
    if args.image: return [(Path(args.image).stem, Path(args.image))]
    if args.image_dir:
        return [(p.stem, p) for p in sorted(Path(args.image_dir).iterdir()) if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}]
    rows = []
    for line in Path(args.manifest).read_text().splitlines():
        item = json.loads(line); path = Path(item.get("image_path") or item["image"])
        rows.append((str(item.get("sample_id") or item.get("image_id") or path.stem), path))
    return rows


def main():
    args = _args()
    if args.inference == "slide":
        raise NotImplementedError("stock CLI slide mode requires the upstream window graph adapter; no independent-tile fallback is permitted")
    preflight(args.mode, args.device)
    cfg = OmegaConf.load(args.model_config)
    if cfg.model.type != "LPOSS": raise ValueError("--model-config must configure the stock LPOSS model")
    ontology = load_ontology(args.vocabulary)
    classes = list(heritage_runtime_vocabulary(ontology))
    if args.extra_concepts:
        for item in yaml.safe_load(Path(args.extra_concepts).read_text()):
            classes.append(RuntimeClass(item["name"], tuple(item["prompts"]), semantic_id=item.get("semantic_id")))
    contrast = yaml.safe_load(Path(args.ornament_contrast).read_text())
    classes.append(RuntimeClass(contrast["name"], tuple(contrast["prompts"]), semantic_id=None))
    model_cfg = cfg.model
    if args.mode == "maskclip_raw":
        # Building LPOSS would load DINO even though raw mode must not execute that branch.
        model_cfg = OmegaConf.load(Path(args.model_config).parent / f"{cfg.model.clip_backbone}.yaml").model
    model = build_model(model_cfg, class_names=[c.name for c in classes]).to(args.device).eval()
    adapter = LPOSSStockAdapter(model)
    prototypes = build_prototypes(classes, adapter.encode_text, device=args.device, ontology_hash=ontology.hash)
    params = {key: OmegaConf.select(cfg, key) for key in ("alpha", "gamma", "k", "sigma", "pix_dist_pow", "tau", "r")}
    params.update({"vit_arch": cfg.model.vit_arch, "vit_patch_size": cfg.model.vit_patch_size,
                   "enc_type_feats": cfg.model.enc_type_feats, "inference": args.inference})
    engine = StockOVSEngine(adapter, UpstreamGraphPropagator(), device=args.device,
                            cosine_scale=1.0, graph_parameters=params)
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    records = []
    negative = len(classes) - 1
    for sample_id, path in _inputs(args):
        started = time.perf_counter()
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        image = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(args.device)
        if torch.cuda.is_available() and torch.device(args.device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(torch.device(args.device))
        result = engine.run(image, prototypes, mode=args.mode, ornament_negative_index=negative,
                            ornament_threshold=args.ornament_threshold)
        base = outdir / sample_id; base.mkdir(exist_ok=True)
        artifacts = {}
        if result.main_mask is not None:
            target = base / "main_semantic.png"; Image.fromarray(result.main_mask[0].byte().cpu().numpy()).save(target); artifacts["main_mask"] = _hash(target)
        if result.ornament_probability is not None:
            target = base / "ornament_probability.npy"; np.save(target, result.ornament_probability[0, 0].cpu().numpy()); artifacts["ornament_probability"] = _hash(target)
        if result.ornament_mask is not None:
            target = base / "ornament_mask.png"; Image.fromarray(result.ornament_mask[0, 0].mul(255).cpu().numpy()).save(target); artifacts["ornament_mask"] = _hash(target)
        if args.save_scores:
            target = base / "scores.pt"; torch.save({"seed_scores": result.seed_scores.cpu(), "propagated_scores": result.propagated_scores.cpu(), "extra_scores": {k:v.cpu() for k,v in result.extra_scores.items()}}, target); artifacts["scores"] = _hash(target)
        records.append({"sample_id": sample_id, "image_path": str(path), "image_sha256": _hash(path),
            "original_size": list(rgb.shape[:2]), "output_grid": list(result.propagated_scores.shape[-2:]),
            "channel_names": list(prototypes.channel_names), "semantic_ids": list(prototypes.semantic_ids),
            "vocabulary_hash": prototypes.vocabulary_hash, "ontology_hash": ontology.hash,
            "ornament_contrast": contrast, "ornament_threshold": args.ornament_threshold,
            "execution": metadata_dict(result), "graph_parameters": params, "artifacts": artifacts,
            "elapsed_seconds": time.perf_counter()-started,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(torch.device(args.device)) if torch.cuda.is_available() and torch.device(args.device).type == "cuda" else None})
    manifest = {"schema_version": 1, "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_reference_commit": UPSTREAM_REFERENCE_COMMIT, "model_config": str(args.model_config),
        "model_config_sha256": _hash(args.model_config), "device": args.device,
        "requested_mode": args.mode, "records": records}
    (outdir / "run_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    if args.ledger_dir:
        from research_ledger import Ledger, NewEvent
        ledger = Ledger(args.ledger_dir, f"stock-lposs-{int(time.time())}")
        ledger.append(NewEvent("run.started", {"command": "ovs_heritage.infer_ovs", "mode": args.mode}))
        ledger.append(NewEvent("config.snapshot", {"manifest": manifest}))
        ledger.append(NewEvent("run.completed", {"status": "completed", "run_manifest": str(outdir / 'run_manifest.json')}))


if __name__ == "__main__": main()
