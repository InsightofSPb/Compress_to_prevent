"""Stock open-vocabulary MaskCLIP/LPOSS inference.

The graph equations follow the MIT-licensed LPOSS implementation by Stojnic et al.
Components are injectable so the execution contract can be tested without weights/CUDA.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import importlib.util
from typing import Protocol

import torch
import torch.nn.functional as F

from .projection import MAIN_SEMANTIC_IDS
from .scoring import RawCosineScorer
from .vocabulary import PrototypeSet

MODES = ("maskclip_raw", "lposs", "lposs_plus")


class StockFeatureModel(Protocol):
    def clip_dense(self, image: torch.Tensor) -> torch.Tensor: ...
    def dino_dense(self, image: torch.Tensor) -> torch.Tensor: ...


class Propagator(Protocol):
    def patch(self, dino: torch.Tensor, seeds: torch.Tensor, *, image: torch.Tensor,
              locations: torch.Tensor | None, parameters: dict) -> tuple[torch.Tensor, int]: ...
    def pixel(self, image: torch.Tensor, scores: torch.Tensor, *, parameters: dict) -> torch.Tensor: ...


@dataclass(frozen=True)
class ExecutionMetadata:
    requested_mode: str
    resolved_mode: str
    dino_executed: bool
    patch_propagation_executed: bool
    pixel_refinement_executed: bool
    cosine_scale: float
    requested_k: int | None
    effective_k: int | None


@dataclass
class StockOutput:
    seed_scores: torch.Tensor
    propagated_scores: torch.Tensor
    main_scores: torch.Tensor | None
    main_mask: torch.Tensor | None
    ornament_score: torch.Tensor | None
    ornament_probability: torch.Tensor | None
    ornament_mask: torch.Tensor | None
    extra_scores: dict[str, torch.Tensor]
    metadata: ExecutionMetadata


def preflight(mode: str, device: torch.device | str) -> None:
    """Fail closed before feature extraction for graph modes."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "maskclip_raw":
        return
    device = torch.device(device)
    missing = []
    if device.type != "cuda" or not torch.cuda.is_available():
        missing.append("an available CUDA device")
    for module, label in (("faiss", "FAISS with GPU support"), ("cupy", "CuPy"),
                          ("cupyx.scipy.sparse.linalg", "cupyx sparse solver")):
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError):
            found = False
        if not found:
            missing.append(label)
    if importlib.util.find_spec("faiss") is not None:
        import faiss
        if not hasattr(faiss, "StandardGpuResources"):
            missing.append("FAISS GPU bindings")
    if missing:
        raise RuntimeError(
            f"{mode} requires genuine GPU graph propagation; missing: {', '.join(missing)}. "
            "Install matching CUDA CuPy and faiss-gpu packages, or explicitly request maskclip_raw."
        )


class StockOVSEngine:
    """Runs exactly the stages named by ``mode``; prototypes are call-time data."""
    def __init__(self, model: StockFeatureModel, propagator: Propagator | None,
                 *, device: str | torch.device, cosine_scale: float = 1.0,
                 graph_parameters: dict | None = None, enforce_preflight: bool = True):
        self.model, self.propagator = model, propagator
        self.device = torch.device(device)
        self.scorer = RawCosineScorer(scale=cosine_scale)
        self.graph_parameters = dict(graph_parameters or {})
        self.enforce_preflight = enforce_preflight

    @torch.no_grad()
    def run(self, image: torch.Tensor, prototypes: PrototypeSet, *, mode: str,
            ornament_negative_index: int | None = None, ornament_threshold: float | None = None,
            locations: torch.Tensor | None = None) -> StockOutput:
        if self.enforce_preflight:
            preflight(mode, self.device)
        elif mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        image = image.to(self.device)
        original_size = image.shape[-2:]
        clip = self.model.clip_dense(image)
        seeds = self.scorer(clip, prototypes.prototypes)
        scores, dino_done, patch_done, pixel_done, effective_k = seeds, False, False, False, None
        if mode != "maskclip_raw":
            if self.propagator is None:
                raise RuntimeError("LPOSS mode requested but no graph propagator is configured")
            dino = self.model.dino_dense(image)
            dino_done = True
            scores, effective_k = self.propagator.patch(
                dino, seeds, image=image, locations=locations, parameters=self.graph_parameters)
            patch_done = True
            if mode == "lposs_plus":
                scores = self.propagator.pixel(image, scores, parameters=self.graph_parameters)
                pixel_done = True
        scores = F.interpolate(scores, original_size, mode="bilinear", align_corners=False)
        seeds_full = F.interpolate(seeds, original_size, mode="bilinear", align_corners=False)
        names, ids = prototypes.channel_names, prototypes.semantic_ids
        main_indices = [ids.index(i) for i in MAIN_SEMANTIC_IDS if i in ids]
        complete = len(main_indices) == len(MAIN_SEMANTIC_IDS)
        main_scores = scores[:, main_indices] if complete else None
        main_mask = None
        if complete:
            lookup = torch.tensor(MAIN_SEMANTIC_IDS, device=scores.device)
            main_mask = lookup[main_scores.argmax(1)]
        ornament_index = ids.index(8) if 8 in ids else None
        ornament_score = None
        if ornament_index is not None and ornament_negative_index is not None:
            ornament_score = scores[:, ornament_index:ornament_index + 1] - scores[:, ornament_negative_index:ornament_negative_index + 1]
        ornament_probability = torch.sigmoid(ornament_score) if ornament_score is not None else None
        ornament_mask = None
        if ornament_probability is not None and ornament_threshold is not None:
            ornament_mask = (ornament_probability >= ornament_threshold).to(torch.uint8)
        excluded = set(main_indices)
        if ornament_index is not None: excluded.add(ornament_index)
        if ornament_negative_index is not None: excluded.add(ornament_negative_index)
        extras = {name: scores[:, i:i + 1] for i, (name, sid) in enumerate(zip(names, ids))
                  if sid is None and i not in excluded}
        meta = ExecutionMetadata(mode, mode, dino_done, patch_done, pixel_done,
                                 self.scorer.scale, self.graph_parameters.get("k"), effective_k)
        return StockOutput(seeds_full, scores, main_scores, main_mask, ornament_score,
                           ornament_probability, ornament_mask, extras, meta)


class UpstreamGraphPropagator:
    """Thin device-safe adapter around the forked upstream LPOSS graph routines."""
    def patch(self, dino, seeds, *, image, locations, parameters):
        from segmentation.evaluation.lposs_eval import get_lposs_laplacian, perform_lp
        b, _, h, w = dino.shape
        if b != 1:
            raise ValueError("stock whole-image graph propagation currently requires batch size 1")
        feats = F.normalize(dino.permute(0, 2, 3, 1).reshape(h * w, -1), dim=-1)
        patch_seeds = F.interpolate(seeds, (h, w), mode="bilinear", align_corners=False)
        flat = patch_seeds.permute(0, 2, 3, 1).reshape(h * w, -1)
        requested = int(parameters["k"])
        effective = min(requested, h * w)
        loc = image.new_zeros((1, 4)) if locations is None else locations
        L = get_lposs_laplacian(feats, loc, [(h, w)], k=effective,
            sigma=parameters["sigma"], pix_dist_pow=parameters["pix_dist_pow"],
            gamma=parameters["gamma"], alpha=parameters["alpha"],
            patch_size=parameters["vit_patch_size"])
        out = perform_lp(L, flat, device=image.device)
        return out.reshape(h, w, -1).permute(2, 0, 1).unsqueeze(0), effective

    def pixel(self, image, scores, *, parameters):
        from segmentation.evaluation.lposs_eval import get_lposs_plus_laplacian, perform_lp
        scores = F.interpolate(scores, image.shape[-2:], mode="bilinear", align_corners=False)
        flat = scores[0].permute(1, 2, 0).reshape(-1, scores.shape[1])
        L = get_lposs_plus_laplacian(image, flat, tau=parameters["tau"],
                                     neigh=parameters["r"] // 2, alpha=parameters["alpha"])
        out = perform_lp(L, flat, device=image.device)
        return out.reshape(*image.shape[-2:], -1).permute(2, 0, 1).unsqueeze(0)


def metadata_dict(output: StockOutput) -> dict:
    return asdict(output.metadata)
