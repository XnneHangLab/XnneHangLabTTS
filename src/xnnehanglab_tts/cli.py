import argparse
import os
from pathlib import Path

from xnnehanglab_tts.runtime.config import ensure_managed_dirs, load_runtime_config
from xnnehanglab_tts.runtime.environment import inspect_environment
from xnnehanglab_tts.runtime.models import CliEnvelope, RuntimeInspection, VerifyResult
from xnnehanglab_tts.runtime.targets import build_managed_paths, get_download_target
from xnnehanglab_tts.runtime.verify import verify_target


def _config_path_from_env() -> Path | None:
    value = os.getenv("XH_RUNTIME_CONFIG")
    return Path(value) if value else None


def emit_result(payload: dict) -> None:
    print(CliEnvelope(kind="result", payload=payload).model_dump_json(by_alias=True))


def build_runtime_inspection() -> RuntimeInspection:
    config, paths = load_runtime_config(_config_path_from_env())
    ensure_managed_dirs(paths)
    environment = inspect_environment()
    resource = verify_target(get_download_target("genie-base", paths))
    available_backends = (
        ["genie-tts"]
        if environment.mode == "cpu"
        else ["genie-tts", "gsv-tts-lite", "faster-qwen-tts"]
    )
    latest_message = f"运行驱动 {config.runtime_driver}，当前环境 {environment.mode.upper()}"
    return RuntimeInspection(
        runtime_driver=config.runtime_driver,
        default_backend=config.default_backend,
        environment=environment,
        available_backends=available_backends,
        managed_paths=build_managed_paths(paths),
        resources={"genie-base": resource},
        latest_message=latest_message,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xnnehanglab-tts")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inspect-runtime")
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("target")
    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("target")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect-runtime":
        emit_result(build_runtime_inspection().model_dump(by_alias=True))
        return 0

    if args.command == "verify":
        _, paths = load_runtime_config(_config_path_from_env())
        ensure_managed_dirs(paths)
        resource = verify_target(get_download_target(args.target, paths))
        emit_result(VerifyResult(resource=resource).model_dump(by_alias=True))
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
