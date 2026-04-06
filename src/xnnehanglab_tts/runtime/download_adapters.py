import os
import re
import sys
import threading
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

# Matches any tqdm-like bar (e.g. "Processing 25 items: 32%|###...")
# Used to silently drop non-download tqdm output instead of forwarding it to
# stderr where it would appear as garbled text.
_RE_TQDM_ANY = re.compile(r"\d+%\|")

# Matches informational ModelScope logger lines that would otherwise flood the
# launcher console during downloads.
_RE_MODELSCOPE_LOG_NOISE = re.compile(r"\bmodelscope\b\s*-\s*(?:INFO|DEBUG)\s*-")


class DownloadProviderAdapter(Protocol):
    def download(self, *, target: DownloadTargetSpec, step: DownloadStep | None = None) -> str | None:
        ...


class _TqdmCapture:
    """Capture modelscope tqdm/log output and re-emit structured file progress."""

    def __init__(self, emit: EmitEvent, target_id: str) -> None:
        self._emit = emit
        self._target_id = target_id
        self._last_percent: dict[str, int] = {}
        self._saved_fd1: int | None = None
        self._saved_fd2: int | None = None
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        r, w = os.pipe()
        self._r = r
        self._w = w
        self._thread = threading.Thread(target=self._reader, daemon=True)

    def __enter__(self) -> "_TqdmCapture":
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        self._saved_fd1 = os.dup(1)
        self._saved_fd2 = os.dup(2)
        os.dup2(self._w, 1)
        os.dup2(self._w, 2)
        sys.stdout = open(self._saved_fd1, "w", buffering=1, closefd=False)
        sys.stderr = open(2, "w", buffering=1, closefd=False)
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        try:
            sys.stderr.flush()
        except Exception:
            pass
        if self._saved_fd1 is not None:
            os.dup2(self._saved_fd1, 1)
        if self._saved_fd2 is not None:
            os.dup2(self._saved_fd2, 2)
        try:
            os.close(self._w)
        except OSError:
            pass
        self._thread.join(timeout=5)
        if self._saved_fd1 is not None:
            os.close(self._saved_fd1)
            self._saved_fd1 = None
        if self._saved_fd2 is not None:
            os.close(self._saved_fd2)
            self._saved_fd2 = None
        tmp_out, tmp_err = sys.stdout, sys.stderr
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        try:
            tmp_out.close()
        except Exception:
            pass
        try:
            tmp_err.close()
        except Exception:
            pass

    def _reader(self) -> None:
        buf = ""
        while True:
            try:
                chunk = os.read(self._r, 4096)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk.decode("utf-8", errors="replace")
            parts = re.split(r"[\r\n]", buf)
            buf = parts[-1]
            for line in parts[:-1]:
                self._handle(line.strip())
        if buf.strip():
            self._handle(buf.strip())
        try:
            os.close(self._r)
        except OSError:
            pass

    def _handle(self, line: str) -> None:
        if not line:
            return
        if line.lstrip().startswith("{"):
            try:
                if self._saved_fd1 is not None:
                    os.write(self._saved_fd1, (line + "\n").encode())
            except OSError:
                pass
            return
        m = _RE_TQDM_DOWNLOADING.search(line)
        if m:
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
            return
        if _RE_TQDM_ANY.search(line):
            return
        if _RE_MODELSCOPE_LOG_NOISE.search(line):
            return
        try:
            if self._saved_fd2 is not None:
                os.write(self._saved_fd2, (line + "\n").encode())
        except OSError:
            pass


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
            with _TqdmCapture(self._emit, self._target_id):
                return snapshot_download(**kwargs)

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
            "cache_dir": str(target.cache_dir),
            "local_dir": str(local_dir),
            "allow_file_pattern": allow_file_pattern or None,
        }

