"""Execution core for stock open-vocabulary MaskCLIP/LPOSS inference.

Graph behavior follows LPOSS commit e489a7445528922ddfe4e39631ef2fe34827c873.
The small protocols keep CPU contract tests independent of pretrained weights.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import importlib.util
import math
from typing import Protocol

import torch
import torch.nn.functional as F

from .projection import MAIN_SEMANTIC_IDS
from .scoring import RawCosineScorer
from .vocabulary import PrototypeSet

MODES = ("maskclip_raw", "lposs", "lposs_plus")
UPSTREAM_COMMIT = "e489a7445528922ddfe4e39631ef2fe34827c873"
IMPLEMENTATION_ID = "stock-maskclip-lposs-p1a-v1"


class FeatureModel(Protocol):
    def clip_dense(self, image: torch.Tensor) -> torch.Tensor: ...
    def dino_dense(self, image: torch.Tensor) -> torch.Tensor: ...


class GraphPropagator(Protocol):
    def patch_nodes(self, dino_nodes: torch.Tensor, seed_nodes: torch.Tensor, *,
                    locations: torch.Tensor, height_width: list[tuple[int, int]],
                    parameters: dict, device_index: int) -> tuple[torch.Tensor, int]: ...
    def pixel(self, image: torch.Tensor, scores: torch.Tensor, *, parameters: dict,
              device_index: int) -> torch.Tensor: ...


@dataclass(frozen=True)
class DeviceInfo:
    requested_device: str
    resolved_device: str
    logical_index: int | None
    visible_device_name: str | None


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
    inference: str
    window_count: int
    graph_nodes: int | None
    estimated_dense_graph_bytes: int | None
    estimated_pixel_graph_edges: int | None
    estimated_pixel_graph_bytes: int | None
    crop_size: tuple[int, int] | None
    stride: tuple[int, int] | None


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


def resolve_device(requested: str) -> DeviceInfo:
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"invalid torch device {requested!r}: {exc}") from exc
    if device.type == "cpu":
        return DeviceInfo(requested, "cpu", None, None)
    if device.type != "cuda":
        raise ValueError("stock inference supports only cpu and explicitly indexed cuda devices")
    if device.index is None:
        raise ValueError("CUDA device must include a logical index, for example cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError(f"requested cuda:{device.index}, but CUDA is unavailable")
    count = torch.cuda.device_count()
    if device.index < 0 or device.index >= count:
        raise RuntimeError(f"requested cuda:{device.index}, but only {count} logical CUDA device(s) are visible")
    return DeviceInfo(requested, str(device), device.index, torch.cuda.get_device_name(device.index))


def graph_preflight(mode: str, device: DeviceInfo) -> dict[str, str | None]:
    """Import graph dependencies only for graph modes, before model construction."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if mode == "maskclip_raw":
        return {"cupy": None, "faiss": None}
    if device.logical_index is None:
        raise RuntimeError(f"{mode} requires an explicitly indexed CUDA device")
    missing = []
    for name in ("cupy", "cupyx.scipy.sparse.linalg", "faiss"):
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except (ImportError, ModuleNotFoundError):
            missing.append(name)
    if missing:
        raise RuntimeError(f"{mode} requires CUDA CuPy/cupyx and faiss-gpu; missing: {', '.join(missing)}")
    cp = importlib.import_module("cupy")
    faiss = importlib.import_module("faiss")
    if not hasattr(faiss, "StandardGpuResources"):
        raise RuntimeError("faiss is installed without GPU bindings; install a matching faiss-gpu build")
    try:
        with cp.cuda.Device(device.logical_index):
            cp.cuda.runtime.getDevice()
    except Exception as exc:
        raise RuntimeError(f"CuPy cannot select logical CUDA device {device.logical_index}: {exc}") from exc
    return {"cupy": getattr(cp, "__version__", "unknown"),
            "cupy_cuda_runtime": str(cp.cuda.runtime.runtimeGetVersion()),
            "faiss": getattr(faiss, "__version__", "unknown")}


def _available_graph_bytes(device: DeviceInfo, parameters: dict) -> tuple[int, int]:
    reserve = int(parameters.get("gpu_memory_reserve_bytes", 1024**3))
    if "available_gpu_bytes" in parameters:  # deterministic contract-test override
        free = int(parameters["available_gpu_bytes"])
    else:
        free, _ = torch.cuda.mem_get_info(device.logical_index)
    return free, max(0, free - reserve)


def patch_graph_preflight(nodes: int, channels: int, element_size: int,
                          device: DeviceInfo, parameters: dict) -> int:
    """Conservative peak for affinity, distances, masks, Laplacian and solver work."""
    dense = nodes * nodes
    estimated = dense * (element_size * 6 + 2) + nodes * channels * element_size * 4
    free, usable = _available_graph_bytes(device, parameters)
    max_nodes = int(parameters.get("max_graph_nodes", 250_000))
    max_bytes = int(parameters.get("max_dense_graph_bytes", 8 * 1024**3))
    if nodes > max_nodes or estimated > max_bytes or estimated > usable:
        raise RuntimeError("patch graph preflight rejected: "
            f"nodes={nodes}, edges={dense}, estimated_bytes={estimated}, available_bytes={free}, "
            f"usable_after_reserve_bytes={usable}; reduce image/window count or select a larger GPU; "
            "stock inference will not resize automatically")
    return estimated


def pixel_graph_preflight(height: int, width: int, channels: int, element_size: int,
                          device: DeviceInfo, parameters: dict) -> tuple[int, int]:
    nodes = height * width
    radius = int(parameters["r"]) // 2
    neighbours = (2 * radius + 1) ** 2 - 1
    edges = nodes * neighbours  # safe upper bound; boundaries only reduce it
    # COO row/column + value, two sparse matrices, CSR work, labels and solver vectors.
    estimated = edges * (8 * 2 + element_size) * 3 + (nodes + 1) * 8 * 3
    estimated += nodes * channels * element_size * 5
    free, usable = _available_graph_bytes(device, parameters)
    limits = (int(parameters.get("max_pixel_nodes", 2_000_000)),
              int(parameters.get("max_pixel_edges", 250_000_000)),
              int(parameters.get("max_pixel_graph_bytes", 8 * 1024**3)))
    if nodes > limits[0] or edges > limits[1] or estimated > limits[2] or estimated > usable:
        raise RuntimeError("pixel graph preflight rejected: "
            f"nodes={nodes}, edges={edges}, estimated_bytes={estimated}, available_bytes={free}, "
            f"usable_after_reserve_bytes={usable}; reduce input size explicitly or select a larger GPU; "
            "stock inference will not resize automatically")
    return edges, estimated


def _windows(h: int, w: int, crop: tuple[int, int], stride: tuple[int, int]):
    ch, cw = crop
    sh, sw = stride
    if min(ch, cw, sh, sw) <= 0 or sh > ch or sw > cw:
        raise ValueError("crop and stride must be positive and stride must not exceed crop")
    hg = max(h - ch + sh - 1, 0) // sh + 1
    wg = max(w - cw + sw - 1, 0) // sw + 1
    result = []
    for hi in range(hg):
        for wi in range(wg):
            y2, x2 = min(hi * sh + ch, h), min(wi * sw + cw, w)
            y1, x1 = max(y2 - ch, 0), max(x2 - cw, 0)
            result.append((y1, y2, x1, x2))
    return result


class StockOVSEngine:
    def __init__(self, model: FeatureModel, propagator: GraphPropagator | None, *,
                 device: DeviceInfo, cosine_scale: float = 1.0,
                 graph_parameters: dict | None = None):
        self.model, self.propagator, self.device = model, propagator, device
        self.scorer = RawCosineScorer(scale=cosine_scale)
        self.parameters = dict(graph_parameters or {})

    @torch.no_grad()
    def run(self, image: torch.Tensor, prototypes: PrototypeSet, *, mode: str,
            inference: str = "whole", crop_size: tuple[int, int] | None = None,
            stride: tuple[int, int] | None = None, ornament_negative_index: int | None = None,
            ornament_threshold: float | None = None) -> StockOutput:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        if inference not in ("whole", "slide"):
            raise ValueError("inference must be whole or slide")
        if ornament_threshold is not None and (not math.isfinite(ornament_threshold) or
                                                not 0 <= ornament_threshold <= 1):
            raise ValueError("ornament_threshold must be finite and within [0, 1]")
        target = torch.device(self.device.resolved_device)
        image = image.to(target)
        original = image.shape[-2:]
        windows = [(0, original[0], 0, original[1])]
        if inference == "slide":
            if crop_size is None or stride is None:
                raise ValueError("slide inference requires crop_size and stride")
            windows = _windows(*original, crop_size, stride)
        crops = torch.cat([image[..., y1:y2, x1:x2] for y1, y2, x1, x2 in windows])
        clip = self.model.clip_dense(crops)
        window_seed_maps = self.scorer(clip, prototypes.prototypes)
        dino_done = patch_done = pixel_done = False
        effective_k = graph_nodes = estimated_graph_bytes = None
        pixel_edges = estimated_pixel_bytes = None
        if mode == "maskclip_raw":
            window_outputs = [window_seed_maps[i:i + 1] for i in range(len(windows))]
        else:
            if self.propagator is None or self.device.logical_index is None:
                raise RuntimeError("graph mode requires a configured GPU propagator")
            dino = self.model.dino_dense(crops)
            dino_done = True
            height_width = [tuple(item.shape[-2:]) for item in dino]
            seeds = [F.interpolate(window_seed_maps[i:i + 1], size=height_width[i],
                                   mode="bilinear", align_corners=False)[0]
                     for i in range(len(windows))]
            dino_nodes = torch.cat([item.permute(1, 2, 0).reshape(-1, item.shape[0]) for item in dino])
            seed_nodes = torch.cat([item.permute(1, 2, 0).reshape(-1, item.shape[0]) for item in seeds])
            graph_nodes = dino_nodes.shape[0]
            estimated_graph_bytes = patch_graph_preflight(
                graph_nodes, seed_nodes.shape[1], dino_nodes.element_size(), self.device, self.parameters)
            locations = torch.tensor(windows, device=target, dtype=dino_nodes.dtype)
            context = torch.cuda.device(self.device.logical_index)
            with context:
                propagated, effective_k = self.propagator.patch_nodes(
                    dino_nodes, seed_nodes, locations=locations, height_width=height_width,
                    parameters=self.parameters, device_index=self.device.logical_index)
            patch_done = True
            window_outputs, offset = [], 0
            for h, w in height_width:
                count = h * w
                window_outputs.append(propagated[offset:offset + count].reshape(h, w, -1)
                                      .permute(2, 0, 1).unsqueeze(0))
                offset += count
        seeds_full = self._stitch(window_seed_maps, windows, original)
        scores = self._stitch(window_outputs, windows, original)
        if mode == "lposs_plus":
            pixel_edges, estimated_pixel_bytes = pixel_graph_preflight(
                original[0], original[1], scores.shape[1], scores.element_size(),
                self.device, self.parameters)
            with torch.cuda.device(self.device.logical_index):
                scores = self.propagator.pixel(image, scores, parameters=self.parameters,
                                               device_index=self.device.logical_index)
            pixel_done = True
        self._validate_scores(seeds_full, scores, original, len(prototypes.channel_names))
        result = self._project(scores, seeds_full, prototypes, ornament_negative_index,
                               ornament_threshold)
        result.metadata = ExecutionMetadata(
            mode, mode, dino_done, patch_done, pixel_done, self.scorer.scale,
            self.parameters.get("k"), effective_k, inference, len(windows), graph_nodes,
            estimated_graph_bytes,
            pixel_edges, estimated_pixel_bytes,
            crop_size if inference == "slide" else None,
            stride if inference == "slide" else None)
        return result

    @staticmethod
    def _stitch(maps, windows, size):
        if isinstance(maps, torch.Tensor):
            maps = [maps[i:i + 1] for i in range(maps.shape[0])]
        channels, device, dtype = maps[0].shape[1], maps[0].device, maps[0].dtype
        output = torch.zeros(1, channels, *size, device=device, dtype=dtype)
        count = torch.zeros(1, 1, *size, device=device, dtype=dtype)
        for score, (y1, y2, x1, x2) in zip(maps, windows):
            score = F.interpolate(score, (y2 - y1, x2 - x1), mode="bilinear", align_corners=False)
            output[..., y1:y2, x1:x2] += score
            count[..., y1:y2, x1:x2] += 1
        if torch.any(count == 0):
            raise RuntimeError("sliding windows do not cover the complete image")
        return output / count

    @staticmethod
    def _validate_scores(seed, propagated, size, channels):
        for name, tensor in (("seed_scores", seed), ("propagated_scores", propagated)):
            if tensor.shape != (1, channels, *size):
                raise ValueError(f"{name} has invalid shape {tuple(tensor.shape)}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains non-finite values")

    @staticmethod
    def _project(scores, seeds, prototypes, negative, threshold):
        names, ids = prototypes.channel_names, prototypes.semantic_ids
        indices = [ids.index(i) for i in MAIN_SEMANTIC_IDS if i in ids]
        complete = len(indices) == len(MAIN_SEMANTIC_IDS)
        main_scores = scores[:, indices] if complete else None
        main_mask = None
        if complete:
            lookup = torch.tensor(MAIN_SEMANTIC_IDS, device=scores.device, dtype=torch.uint8)
            main_mask = lookup[main_scores.argmax(1)]
        ornament = ids.index(8) if 8 in ids else None
        contrast = (scores[:, ornament:ornament + 1] - scores[:, negative:negative + 1]
                    if ornament is not None and negative is not None else None)
        probability = torch.sigmoid(contrast) if contrast is not None else None
        binary = ((probability >= threshold).to(torch.uint8)
                  if probability is not None and threshold is not None else None)
        excluded = set(indices) | ({ornament} if ornament is not None else set())
        if negative is not None:
            excluded.add(negative)
        extras = {name: scores[:, i:i + 1] for i, (name, sid) in enumerate(zip(names, ids))
                  if sid is None and i not in excluded}
        placeholder = ExecutionMetadata("", "", False, False, False, 1, None, None,
                                        "whole", 1, None, None, None, None, None, None)
        return StockOutput(seeds, scores, main_scores, main_mask, contrast, probability,
                           binary, extras, placeholder)


class UpstreamGraphPropagator:
    def patch_nodes(self, dino_nodes, seed_nodes, *, locations, height_width,
                    parameters, device_index):
        from segmentation.evaluation.lposs_eval import get_lposs_laplacian, perform_lp
        requested = int(parameters["k"])
        effective = min(requested, dino_nodes.shape[0])
        L = get_lposs_laplacian(
            F.normalize(dino_nodes, dim=-1), locations, height_width,
            sigma=parameters["sigma"], pix_dist_pow=parameters["pix_dist_pow"],
            k=effective, gamma=parameters["gamma"], alpha=parameters["alpha"],
            patch_size=parameters["vit_patch_size"], device_index=device_index)
        return perform_lp(L, seed_nodes, device=seed_nodes.device,
                          device_index=device_index), effective

    def pixel(self, image, scores, *, parameters, device_index):
        from segmentation.evaluation.lposs_eval import get_lposs_plus_laplacian, perform_lp
        flat = scores[0].permute(1, 2, 0).reshape(-1, scores.shape[1])
        L = get_lposs_plus_laplacian(image, flat, tau=parameters["tau"],
                                     neigh=parameters["r"] // 2,
                                     alpha=parameters["alpha"], device_index=device_index)
        out = perform_lp(L, flat, device=flat.device, device_index=device_index)
        return out.reshape(*image.shape[-2:], -1).permute(2, 0, 1).unsqueeze(0)


def metadata_dict(output: StockOutput) -> dict:
    return asdict(output.metadata)
