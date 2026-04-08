import os
import sys
from collections.abc import Callable
from typing import Protocol

from .models import DownloadStep, DownloadTargetSpec
from .tty import FakeTtyStderr

EmitEvent = Callable[[dict], None]
SnapshotDownload = Callable[..., str]


class DownloadProviderAdapter(Protocol):
    def download(self, *, target: DownloadTargetSpec, step: DownloadStep | None = None) -> str | None:
        ...


class ModelscopeDownloadAdapter:
    def __init__(
        self,
        emit: EmitEvent,
        target_id: str,
        snapshot_download: SnapshotDownload | None = None,
    ) -> None:
        self._emit = emit
        self._target_id = target_id
        self._snapshot_download = snapshot_download

    def download(self, *, target: DownloadTargetSpec, step: DownloadStep | None = None) -> str:
        kwargs = self._build_kwargs(target=target, step=step)
        downloader = self._snapshot_download or self._make_snapshot_download()
        return downloader(**kwargs)

    def _make_snapshot_download(self) -> SnapshotDownload:
        def _downloader(**kwargs) -> str:
            import logging

            os.environ["MODELSCOPE_LOG_LEVEL"] = str(logging.WARNING)
            from modelscope import snapshot_download

            logger = logging.getLogger("modelscope")
            logger.setLevel(logging.WARNING)
            for handler in logger.handlers:
                handler.setLevel(logging.WARNING)

            orig_stderr = sys.stderr
            sys.stderr = FakeTtyStderr(orig_stderr)
            try:
                return snapshot_download(**kwargs)
            finally:
                sys.stderr = orig_stderr

        return _downloader

    @staticmethod
    def _build_kwargs(
        *,
        target: DownloadTargetSpec,
        step: DownloadStep | None,
    ) -> dict[str, str | list[str] | None]:
        repo_id = target.repo_id if step is None else step.repo_id
        local_dir = target.local_dir if step is None else step.local_dir
        allow_file_pattern = target.allow_file_pattern if step is None else step.allow_file_pattern
        return {
            "model_id": repo_id,
            "local_dir": str(local_dir),
            "allow_file_pattern": allow_file_pattern or None,
        }
