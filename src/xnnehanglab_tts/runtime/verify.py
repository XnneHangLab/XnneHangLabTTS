from .models import ResourceState


def verify_target(target) -> ResourceState:
    if not target.resource_root.exists():
        return ResourceState(
            key=target.target_id,
            label=target.label,
            status="missing",
            path=str(target.resource_root),
            missing_paths=target.required_paths,
        )

    missing_paths: list[str] = []
    for relative_path in target.required_paths:
        if not (target.resource_root / relative_path).exists():
            missing_paths.append(relative_path)

    if not missing_paths:
        status = "ready"
    elif len(missing_paths) == len(target.required_paths):
        status = "missing"
    else:
        status = "partial"

    return ResourceState(
        key=target.target_id,
        label=target.label,
        status=status,
        path=str(target.resource_root),
        missing_paths=missing_paths,
    )
