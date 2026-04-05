from collections.abc import Callable

from .models import ResourceState
from .verify import verify_target

SnapshotDownload = Callable[..., str]
EmitEvent = Callable[[dict], None]


def _modelscope_snapshot_download(**kwargs) -> str:
    from modelscope import snapshot_download

    return snapshot_download(**kwargs)


def download_target_bundle(
    target,
    emit: EmitEvent,
    snapshot_download: SnapshotDownload | None = None,
) -> ResourceState:
    downloader = snapshot_download or _modelscope_snapshot_download

    emit(
        {
            "event": "download.started",
            "target": target.target_id,
            "status": "preparing",
            "message": f"开始准备下载 {target.label}",
            "progressCurrent": 0,
            "progressTotal": 3,
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
            "progressTotal": 3,
            "progressUnit": "stage",
        }
    )

    downloader(
        model_id=target.repo_id,
        cache_dir=str(target.cache_dir),
        local_dir=str(target.local_dir),
        allow_file_pattern=target.allow_file_pattern,
    )

    emit(
        {
            "event": "download.verifying",
            "target": target.target_id,
            "status": "verifying",
            "message": f"开始校验 {target.label}",
            "progressCurrent": 2,
            "progressTotal": 3,
            "progressUnit": "stage",
        }
    )

    result = verify_target(target)
    if result.status != "ready":
        raise RuntimeError(f"{target.target_id} verify failed: {result.status}")

    emit(
        {
            "event": "download.completed",
            "target": target.target_id,
            "status": "completed",
            "message": f"{target.label} 下载完成",
            "progressCurrent": 3,
            "progressTotal": 3,
            "progressUnit": "stage",
        }
    )
    return result
