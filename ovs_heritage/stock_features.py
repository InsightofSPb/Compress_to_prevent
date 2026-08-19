"""Vocabulary-independent frozen CLIP and DINO dense feature model."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class StockFeatureModel(nn.Module):
    """Load image/text CLIP once; runtime classes never enter module state."""
    def __init__(self, *, clip_model: str, clip_pretrained: str,
                 patch_size: int, image_size: int = 224):
        super().__init__()
        from open_clip import create_model_from_pretrained, get_tokenizer

        model, _ = create_model_from_pretrained(clip_model, pretrained=clip_pretrained)
        model.eval()
        self.clip = model
        self.tokenizer = get_tokenizer(clip_model)
        self.patch_size = patch_size
        self.image_size = image_size
        self.register_buffer("clip_mean", torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("clip_std", torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(1, 3, 1, 1), persistent=False)
        self._original_position = self.clip.visual.positional_embedding.detach().clone()
        self._hook = {}
        self.clip.visual.transformer.resblocks[-2].register_forward_hook(
            lambda _module, _inputs, output: self._hook.__setitem__("v", output))
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.dino_encoder = None
        self.dino_patch_size = None
        self.dino_feature_type = None
        self._dino_hook = {}

    @torch.no_grad()
    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        tokens = torch.cat([self.tokenizer(prompt) for prompt in prompts])
        tokens = tokens.to(next(self.clip.parameters()).device)
        return self.clip.encode_text(tokens)

    @torch.no_grad()
    def clip_dense(self, image: torch.Tensor) -> torch.Tensor:
        pad_h = (-image.shape[-2]) % self.patch_size
        pad_w = (-image.shape[-1]) % self.patch_size
        image = F.pad(image, (0, pad_w, 0, pad_h))
        image = (image - self.clip_mean) / self.clip_std
        b, _, h, w = image.shape
        grid = (h // self.patch_size, w // self.patch_size)
        if min(grid) <= 0:
            raise ValueError("image/window is smaller than the CLIP patch size")
        positional = self._resize_position(grid)
        visual = self.clip.visual
        saved = visual.positional_embedding.data
        try:
            visual.positional_embedding.data = positional
            _ = self.clip(image)
            value = self._extract_value(self._hook["v"], visual.transformer.resblocks[-1])
            value = visual.ln_post(value.permute(1, 0, 2)).permute(1, 0, 2)[:, 1:]
            value = value.reshape(b, *grid, -1).permute(0, 3, 1, 2).contiguous()
            projection = visual.proj
            return F.conv2d(value, projection.t()[:, :, None, None])
        finally:
            visual.positional_embedding.data = saved

    def configure_dino(self, *, repository: str, model: str, patch_size: int,
                       feature_type: str, source: str = "github", weights: str | None = None):
        kwargs = {"source": source}
        if weights is not None:
            kwargs["weights"] = weights
        self.dino_encoder = torch.hub.load(repository, model, **kwargs)
        self.dino_encoder.eval()
        for parameter in self.dino_encoder.parameters():
            parameter.requires_grad = False
        self.dino_patch_size, self.dino_feature_type = patch_size, feature_type
        self.dino_encoder.blocks[-1].attn.qkv.register_forward_hook(
            lambda _module, _inputs, output: self._dino_hook.__setitem__("qkv", output))
        return self

    @torch.no_grad()
    def dino_dense(self, image: torch.Tensor) -> torch.Tensor:
        if self.dino_encoder is None:
            raise RuntimeError("DINO was not configured for this feature model")
        patch = self.dino_patch_size
        pad_h, pad_w = (-image.shape[-2]) % patch, (-image.shape[-1]) % patch
        image = F.pad(image, (0, pad_w, 0, pad_h))
        mean = image.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = image.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        batch = (image - mean) / std
        h, w = batch.shape[-2] // patch, batch.shape[-1] // patch
        _ = self.dino_encoder(batch)
        qkv = self._dino_hook["qkv"]
        heads = self.dino_encoder.blocks[-1].attn.num_heads
        qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, heads, -1).permute(2, 0, 3, 1, 4)
        selected = {"q": 0, "k": 1, "v": 2}[self.dino_feature_type]
        features = qkv[selected][:, :, 1:, :].permute(0, 2, 1, 3).flatten(-2)
        return F.normalize(features, dim=-1).reshape(batch.shape[0], h, w, -1).permute(0, 3, 1, 2)

    def _resize_position(self, grid):
        position = self._original_position.to(next(self.clip.parameters()).device)
        old = self.image_size // self.patch_size
        cls, spatial = position[:1], position[-old * old:]
        spatial = spatial.reshape(1, old, old, -1).permute(0, 3, 1, 2)
        spatial = F.interpolate(spatial, grid, mode="bicubic", align_corners=False)
        return torch.cat((cls, spatial.flatten(2).transpose(1, 2)[0]))

    @staticmethod
    def _extract_value(x, block):
        y = F.linear(block.ln_1(x), block.attn.in_proj_weight, block.attn.in_proj_bias)
        b, n, c = y.shape
        y = y.view(b, n, 3, c // 3).permute(2, 0, 1, 3).reshape(3 * b, n, c // 3)
        value = F.linear(y, block.attn.out_proj.weight, block.attn.out_proj.bias).tensor_split(3)[2]
        value = value + x
        return value + block.mlp(block.ln_2(value))


def module_state_fingerprint(module: nn.Module) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    """Functional test/provenance helper for vocabulary-invariant model state."""
    result = []
    for key, value in module.state_dict().items():
        raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        digest = sha256(raw).hexdigest()
        result.append((key, tuple(value.shape), digest))
    return tuple(result)


def optional_weight_hash(identifier: str | None) -> str | None:
    if not identifier:
        return None
    path = Path(identifier).expanduser()
    if not path.is_file():
        return None
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
