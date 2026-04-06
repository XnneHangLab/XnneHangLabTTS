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


class _TqdmCapture:
    """Context manager: redirect sys.stderr to a pipe, parse modelscope tqdm lines
    into structured ``download.file_progress`` emit events.

    tqdm detects ``isatty() == False`` and switches from \\r overwrites to \\n
    newlines, which makes line-splitting trivial. Duplicate percent values for the
    same file are deduplicated before emitting.
    """

    def __init__(self, emit: EmitEvent, target_id: str) -> None:
        self._emit = emit
        self._target_id = target_id
        self._last_percent: dict[str, int] = {}
        r, w = os.pipe()
        self._r = r
        self._w: int | None = w
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    # ── fake file interface ────────────────────────────────────────────────────

    def write(self, data: str) -> int:
        if self._w is not None:
            try:
                os.write(self._w, data.encode("utf-8", errors="replace"))
            except OSError:
                pass
        return len(data)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False

    # ── context manager ────────────────────────────────────────────────────────

    def __enter__(self) -> "_TqdmCapture":
        self._old_stderr = sys.stderr
        sys.stderr = self  # type: ignore[assignment]
        return self

    def __exit__(self, *_) -> None:
        sys.stderr = self._old_stderr
        if self._w is not None:
            try:
                os.close(self._w)
            except OSError:
                pass
            self._w = None
        self._thread.join(timeout=5)

    # ── reader thread ──────────────────────────────────────────────────────────

    def _reader(self) -> None:
        buf = ""
        while True:
            try:
                chunk = os.read(self._r, 512)
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


def _make_modelscope_downloader(emit: EmitEvent, target_id: str) -> SnapshotDownload:
    """Return a SnapshotDownload callable that captures tqdm output as emit events."""

    def _downloader(**kwargs) -> str:
        import logging

        from modelscope import snapshot_download

        logging.getLogger("modelscope").setLevel(logging.WARNING)
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
