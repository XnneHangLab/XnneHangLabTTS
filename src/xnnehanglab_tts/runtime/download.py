from collections.abc import Callable

from .models import ResourceState
from .verify import verify_target

SnapshotDownload = Callable[..., str]
EmitEvent = Callable[[dict], None]


def _modelscope_snapshot_download(**kwargs) -> str:
    import logging
    import os

    from modelscope import snapshot_download

    # modelscope 的 tqdm 进度条用 \r 原地覆写，无法被日志系统干净捕获。
    # 前端进度完全依赖 emit_event 的结构化事件，此处压掉 modelscope 自身输出。
    logging.getLogger("modelscope").setLevel(logging.WARNING)
    os.environ["TQDM_DISABLE"] = "1"

    return snapshot_download(**kwargs)


def download_target_bundle(
    target,
    emit: EmitEvent,
    snapshot_download: SnapshotDownload | None = None,
) -> ResourceState:
    downloader = snapshot_download or _modelscope_snapshot_download

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
