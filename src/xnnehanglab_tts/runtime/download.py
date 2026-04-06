import os
import re
import sys
import threading
from collections.abc import Callable

from .models import ResourceState
from .verify import verify_target

SnapshotDownload = Callable[..., str]
EmitEvent = Callable[[dict], None]

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


class _TqdmCapture:
    """Context manager: redirect OS file descriptors 1 (stdout) and 2 (stderr)
    to a pipe so that ALL output — including C-level tqdm writes that bypass
    Python's ``sys.stdout`` / ``sys.stderr`` — is captured.

    Python-level ``sys.stdout`` is re-pointed to the saved Tauri IPC file
    descriptor so that ``emit_event()`` (which uses ``print()``) bypasses the
    pipe and reaches the Tauri backend directly. Python-level ``sys.stderr``
    stays attached to the redirected fd 2 so regular ``tqdm`` writes still flow
    through the capture pipe.

    The reader thread parses captured lines:

    * ``{``-prefixed → forwarded to Tauri IPC fd (safety net)
    * ``Downloading [...]`` tqdm → structured ``download.file_progress`` events
    * Other ``N%|`` tqdm bars → silently dropped
    * Everything else → forwarded to Tauri stderr fd
    """

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

    # ── context manager ────────────────────────────────────────────────

    def __enter__(self) -> "_TqdmCapture":
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        # Save Tauri's original file descriptors
        self._saved_fd1 = os.dup(1)
        self._saved_fd2 = os.dup(2)
        # Redirect OS-level fd 1 & 2 → capture pipe
        os.dup2(self._w, 1)
        os.dup2(self._w, 2)
        # Keep stdout on the original IPC fd so emit_event() bypasses the pipe,
        # but keep stderr attached to fd 2 so tqdm/logging writes are captured.
        sys.stdout = open(self._saved_fd1, "w", buffering=1, closefd=False)
        sys.stderr = open(2, "w", buffering=1, closefd=False)
        # Start reader now that fds are redirected
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        # Flush any buffered Python output
        try:
            sys.stdout.flush()
        except Exception:
            pass
        try:
            sys.stderr.flush()
        except Exception:
            pass
        # Restore OS-level fds (closes their pipe-write-end copies)
        if self._saved_fd1 is not None:
            os.dup2(self._saved_fd1, 1)
        if self._saved_fd2 is not None:
            os.dup2(self._saved_fd2, 2)
        # Close original pipe write end → reader gets EOF
        try:
            os.close(self._w)
        except OSError:
            pass
        # Wait for reader thread to drain
        self._thread.join(timeout=5)
        # Reader is done — safe to close saved fds
        if self._saved_fd1 is not None:
            os.close(self._saved_fd1)
            self._saved_fd1 = None
        if self._saved_fd2 is not None:
            os.close(self._saved_fd2)
            self._saved_fd2 = None
        # Close temporary Python wrappers and restore originals
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

    # ── reader thread ──────────────────────────────────────────────────

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
        # Safety net: JSON IPC that ended up in the pipe
        if line.lstrip().startswith("{"):
            try:
                if self._saved_fd1 is not None:
                    os.write(self._saved_fd1, (line + "\n").encode())
            except OSError:
                pass
            return
        # Download tqdm → structured progress event
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
        # Drop any other tqdm bar silently
        if _RE_TQDM_ANY.search(line):
            return
        if _RE_MODELSCOPE_LOG_NOISE.search(line):
            return
        # Forward genuine output to Tauri stderr
        try:
            if self._saved_fd2 is not None:
                os.write(self._saved_fd2, (line + "\n").encode())
        except OSError:
            pass


def _make_modelscope_downloader(emit: EmitEvent, target_id: str) -> SnapshotDownload:
    """Return a SnapshotDownload callable that captures tqdm output as emit events."""

    def _downloader(**kwargs) -> str:
        import logging

        os.environ["MODELSCOPE_LOG_LEVEL"] = str(logging.WARNING)
        from modelscope import snapshot_download

        logger = logging.getLogger("modelscope")
        logger.setLevel(logging.WARNING)
        for handler in logger.handlers:
            handler.setLevel(logging.WARNING)
        with _TqdmCapture(emit, target_id):
            return snapshot_download(**kwargs)

    return _downloader


def download_target_bundle(
    target,
    emit: EmitEvent,
    snapshot_download: SnapshotDownload | None = None,
) -> ResourceState:
    downloader = snapshot_download or _make_modelscope_downloader(emit, target.target_id)

    steps = target.download_steps
    if steps:
        return _download_compound(target, steps, emit, downloader)
    return _download_single(target, emit, downloader)


def _download_single(target, emit: EmitEvent, downloader: SnapshotDownload) -> ResourceState:
    total = 3

    emit(
        {
            "event": "download.started",
            "target": target.target_id,
            "status": "preparing",
            "message": f"开始准备下载 {target.label}",
            "progressCurrent": 0,
            "progressTotal": total,
            "progressUnit": "stage",
        }
    )
    emit(
        {
            "event": "download.progress",
            "target": target.target_id,
            "status": "downloading",
            "message": f"正在从 ModelScope 下载 {target.repo_id}",
            "progressCurrent": 1,
            "progressTotal": total,
            "progressUnit": "stage",
        }
    )

    downloader(
        model_id=target.repo_id,
        cache_dir=str(target.cache_dir),
        local_dir=str(target.local_dir),
        allow_file_pattern=target.allow_file_pattern or None,
    )

    return _verify_and_complete(target, emit, current=2, total=total)


def _download_compound(target, steps, emit: EmitEvent, downloader: SnapshotDownload) -> ResourceState:
    total = len(steps) + 2  # started + N step downloads + verifying/done

    emit(
        {
            "event": "download.started",
            "target": target.target_id,
            "status": "preparing",
            "message": f"开始准备下载 {target.label}（共 {len(steps)} 个子资源）",
            "progressCurrent": 0,
            "progressTotal": total,
            "progressUnit": "stage",
        }
    )

    for idx, step in enumerate(steps, start=1):
        repo_name = step.repo_id.split("/")[-1]
        emit(
            {
                "event": "download.progress",
                "target": target.target_id,
                "status": "downloading",
                "message": f"下载 {repo_name}（{idx}/{len(steps)}）",
                "progressCurrent": idx,
                "progressTotal": total,
                "progressUnit": "stage",
            }
        )
        kwargs: dict = {
            "model_id": step.repo_id,
            "cache_dir": str(target.cache_dir),
            "local_dir": str(step.local_dir),
        }
        if step.allow_file_pattern:
            kwargs["allow_file_pattern"] = step.allow_file_pattern
        downloader(**kwargs)

    return _verify_and_complete(target, emit, current=len(steps) + 1, total=total)


def _verify_and_complete(target, emit: EmitEvent, current: int, total: int) -> ResourceState:
    emit(
        {
            "event": "download.verifying",
            "target": target.target_id,
            "status": "verifying",
            "message": f"开始校验 {target.label}",
            "progressCurrent": current,
            "progressTotal": total,
            "progressUnit": "stage",
        }
    )

    result = verify_target(target)
    if result.status != "ready":
        missing = ",".join(result.missing_paths) if result.missing_paths else "<none>"
        raise RuntimeError(
            f"{target.target_id} verify failed: status={result.status}; missing_paths={missing}"
        )

    emit(
        {
            "event": "download.completed",
            "target": target.target_id,
            "status": "completed",
            "message": f"{target.label} 下载完成",
            "progressCurrent": total,
            "progressTotal": total,
            "progressUnit": "stage",
        }
    )
    return result
