from .base import BackendRegistry
from .clip_backend import CLIPBackend
from .dinov2_backend import DINOv2Backend
from .florence2_backend import Florence2Backend
from .lposs_backend import LPOSSBackend
from .siglip2_backend import SigLIP2Backend


def default_registry() -> BackendRegistry:
    reg = BackendRegistry()
    reg.register(LPOSSBackend)
    reg.register(DINOv2Backend)
    reg.register(CLIPBackend)
    reg.register(Florence2Backend)
    reg.register(SigLIP2Backend)
    return reg


__all__ = [
    "default_registry",
    "BackendRegistry",
    "LPOSSBackend",
    "DINOv2Backend",
    "CLIPBackend",
    "Florence2Backend",
    "SigLIP2Backend",
]
