import os
import re
import sys
from collections.abc import Callable
from typing import Protocol

from .models import DownloadStep, DownloadTargetSpec

EmitEvent = Callable[[dict], None]
SnapshotDownload = Callable[..., str]

# Matches modelscope tqdm lines of the form:
#   Downloading [some/file.bin]:  42%|████      | 75.0M/180M [00:30<00:40, 1.02MB/s]
_RE_TQDM_DOWNLOADING = re.compile(
    r"Downloading \[(.+?)\]:\s*(\d+)%"
    r"(?:[^\|]*\|[^\|]*\|\s*([\d.]+\w+)/([\d.]+\w+))?"
)


class DownloadProviderAdapter(Protocol):
    def download(self, *, target: DownloadTargetSpec, step: DownloadStep | None = None) -> str | None:
        ...


class _TqdmSpy:
    """Wrap sys.stderr: forward everything to the real stderr so the console
    sees all output, and parse tqdm download lines to emit structured
    file-progress events.  No fd-level redirection — no pipes, no threads.
    """

    def __init__(self, emit: EmitEvent, target_id: str, real_stderr) -> None:
        self._emit = emit
        self._target_id = target_id
        self._real = real_stderr
        self._last_percent: dict[str, int] = {}
        self._buf = ""

    # ---- file-like interface ------------------------------------------------

    def write(self, text: str) -> int:
        self._real.write(text)
        try:
            self._real.flush()
        except Exception:
            pass
        self._buf += text
        parts = re.split(r"[\r\n]", self._buf)
        self._buf = parts[-1]
        for line in parts[:-1]:
            self._parse(line.strip())
        return len(text)

    def flush(self) -> None:
        try:
            self._real.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        # Lie to tqdm so it stays in dynamic/interactive mode (uses \r).
        return True

    @property
    def encoding(self) -> str:
        return getattr(self._real, "encoding", "utf-8")

    @property
    def errors(self) -> str:
        return getattr(self._real, "errors", "replace")

    # ---- tqdm line parser ---------------------------------------------------

    def _parse(self, line: str) -> None:
        if not line:
            return
        m = _RE_TQDM_DOWNLOADING.search(line)
        if not m:
            return
        desc = m.group(1)
        percent = int(m.group(2))
        if self._last_percent.get(desc) == percent:
            return
        self._last_percent[desc] = percent
        event: dict = {
            "event": "download.file_progress",
            "target": self._target_id,
            "desc": desc,
            "percent": percent,
        }
        if m.group(3) and m.group(4):
            event["downloaded"] = m.group(3)
            event["total"] = m.group(4)
        self._emit(event)


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
            sys.stderr = _TqdmSpy(self._emit, self._target_id, orig_stderr)
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
