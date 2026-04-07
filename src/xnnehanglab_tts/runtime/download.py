from collections.abc import Callable, Mapping
from typing import Protocol

from .download_adapters import (
    DownloadProviderAdapter,
    ModelscopeDownloadAdapter,
    SnapshotDownload,
)
from .models import ResourceState
from .verify import verify_target

EmitEvent = Callable[[dict], None]

class DownloadVerifier(Protocol):
    def verify(self, target) -> ResourceState:
        ...


class PathDownloadVerifier:
    def verify(self, target) -> ResourceState:
        return verify_target(target)


def download_target_bundle(
    target,
    emit: EmitEvent,
    provider_adapters: Mapping[str, DownloadProviderAdapter] | None = None,
    verifiers: Mapping[str, DownloadVerifier] | None = None,
    snapshot_download: SnapshotDownload | None = None,
) -> ResourceState:
    providers = _build_provider_adapters(
        emit=emit,
        target_id=target.target_id,
        provider_adapters=provider_adapters,
        snapshot_download=snapshot_download,
    )
    verifier_registry = _build_verifiers(verifiers)

    steps = target.download_steps
    if steps:
        return _download_compound(target, steps, emit, providers, verifier_registry)
    return _download_single(target, emit, providers, verifier_registry)


def _build_provider_adapters(
    *,
    emit: EmitEvent,
    target_id: str,
    provider_adapters: Mapping[str, DownloadProviderAdapter] | None,
    snapshot_download: SnapshotDownload | None,
) -> dict[str, DownloadProviderAdapter]:
    providers: dict[str, DownloadProviderAdapter] = {
        "modelscope": ModelscopeDownloadAdapter(
            emit=emit,
            target_id=target_id,
            snapshot_download=snapshot_download,
        )
    }
    if provider_adapters:
        providers.update(provider_adapters)
    return providers


def _build_verifiers(
    verifiers: Mapping[str, DownloadVerifier] | None,
) -> dict[str, DownloadVerifier]:
    registry: dict[str, DownloadVerifier] = {"paths": PathDownloadVerifier()}
    if verifiers:
        registry.update(verifiers)
    return registry


def _select_provider_name(target, step=None) -> str:
    if step is not None and step.provider:
        return step.provider
    return target.provider


def _select_provider_adapter(
    target,
    providers: Mapping[str, DownloadProviderAdapter],
    step=None,
) -> DownloadProviderAdapter:
    provider_name = _select_provider_name(target, step)
    try:
        return providers[provider_name]
    except KeyError as error:
        raise RuntimeError(
            f"unsupported download provider: {provider_name} (target={target.target_id})"
        ) from error


def _select_verifier(
    target,
    verifiers: Mapping[str, DownloadVerifier],
) -> DownloadVerifier:
    try:
        return verifiers[target.verifier]
    except KeyError as error:
        raise RuntimeError(
            f"unsupported verifier: {target.verifier} (target={target.target_id})"
        ) from error


def _build_single_download_message(target) -> str:
    if target.provider == "modelscope":
        return f"正在从 ModelScope 下载 {target.repo_id}"
    return f"正在通过 {target.provider} 下载 {target.label}"


def _download_single(
    target,
    emit: EmitEvent,
    providers: Mapping[str, DownloadProviderAdapter],
    verifiers: Mapping[str, DownloadVerifier],
) -> ResourceState:
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
            "message": _build_single_download_message(target),
            "progressCurrent": 1,
            "progressTotal": total,
            "progressUnit": "stage",
        }
    )

    adapter = _select_provider_adapter(target, providers)
    adapter.download(target=target)

    return _verify_and_complete(
        target,
        emit,
        verifier=_select_verifier(target, verifiers),
        current=2,
        total=total,
    )


def _download_compound(
    target,
    steps,
    emit: EmitEvent,
    providers: Mapping[str, DownloadProviderAdapter],
    verifiers: Mapping[str, DownloadVerifier],
) -> ResourceState:
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
        emit(
            {
                "event": "download.file_progress",
                "target": target.target_id,
                "status": "downloading",
                "desc": repo_name,
                "percent": 0,
            }
        )
        adapter = _select_provider_adapter(target, providers, step)
        adapter.download(target=target, step=step)
        emit(
            {
                "event": "download.file_progress",
                "target": target.target_id,
                "status": "downloading",
                "desc": repo_name,
                "percent": 100,
            }
        )

    return _verify_and_complete(
        target,
        emit,
        verifier=_select_verifier(target, verifiers),
        current=len(steps) + 1,
        total=total,
    )


def _verify_and_complete(
    target,
    emit: EmitEvent,
    verifier: DownloadVerifier,
    current: int,
    total: int,
) -> ResourceState:
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

    result = verifier.verify(target)
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
