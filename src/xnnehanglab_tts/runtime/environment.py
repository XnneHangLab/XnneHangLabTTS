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
    except Exception as error:
        return EnvironmentState(
            mode="cpu",
            torch_available=False,
            cuda_available=False,
            issues=[f"torch import failed: {error}"],
        )

    issues: list[str] = []
    cuda_available = False
    try:
        cuda_available = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    except Exception as error:  # pragma: no cover - defensive for runtime probe failures
        issues.append(f"torch cuda probe failed: {error}")

    return EnvironmentState(
        mode="gpu" if cuda_available else "cpu",
        torch_available=True,
        torch_version=str(getattr(torch, "__version__", "unknown")),
        cuda_available=cuda_available,
        issues=issues,
    )
