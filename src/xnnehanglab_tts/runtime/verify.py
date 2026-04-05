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
    required_file_paths = list(getattr(target, "required_file_paths", []))
    required_dir_paths = list(getattr(target, "required_dir_paths", []))
    expected_paths = (
        [*required_file_paths, *required_dir_paths]
        if (required_file_paths or required_dir_paths)
        else list(target.required_paths)
    )

    if required_file_paths or required_dir_paths:
        for relative_path in required_file_paths:
            if not (target.resource_root / relative_path).is_file():
                missing_paths.append(relative_path)
        for relative_path in required_dir_paths:
            if not (target.resource_root / relative_path).is_dir():
                missing_paths.append(relative_path)
    else:
        for relative_path in target.required_paths:
            if not (target.resource_root / relative_path).exists():
                missing_paths.append(relative_path)

    if not missing_paths:
        status = "ready"
    elif len(missing_paths) == len(expected_paths):
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
