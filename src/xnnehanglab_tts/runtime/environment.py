from collections.abc import Callable
import importlib
from typing import Any

from .models import EnvironmentState


def _default_torch_loader() -> Any:
    return importlib.import_module("torch")


def inspect_environment(torch_loader: Callable[[], Any] | None = None) -> EnvironmentState:
    loader = torch_loader or _default_torch_loader
    try:
        torch = loader()
    except ImportError as error:
        return EnvironmentState(
            mode="cpu",
            torch_available=False,
            cuda_available=False,
            issues=[f"torch import failed: {error}"],
        )

    cuda_available = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    return EnvironmentState(
        mode="gpu" if cuda_available else "cpu",
        torch_available=True,
        torch_version=str(getattr(torch, "__version__", "unknown")),
        cuda_available=cuda_available,
    )
