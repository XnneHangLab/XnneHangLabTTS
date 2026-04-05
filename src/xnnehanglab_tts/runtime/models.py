from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


RuntimeMode = Literal["cpu", "gpu"]
ResourceStatus = Literal["missing", "partial", "ready"]
RuntimeDriver = Literal["uv"]
DownloadTaskStatus = Literal[
    "queued",
    "preparing",
    "downloading",
    "verifying",
    "completed",
    "failed",
    "cancelled",
]


class RuntimeBaseModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ManagedPath(RuntimeBaseModel):
    key: str
    label: str
    path: str


class ResourceState(RuntimeBaseModel):
    key: str
    label: str
    status: ResourceStatus
    path: str
    missing_paths: list[str] = Field(default_factory=list)


class EnvironmentState(RuntimeBaseModel):
    mode: RuntimeMode
    torch_available: bool
    torch_version: str | None = None
    cuda_available: bool = False
    issues: list[str] = Field(default_factory=list)


class RuntimeInspection(RuntimeBaseModel):
    runtime_driver: RuntimeDriver
    python_path: str
    default_backend: str
    environment: EnvironmentState
    available_backends: list[str]
    managed_paths: list[ManagedPath]
    resources: dict[str, ResourceState]
    latest_message: str


class VerifyResult(RuntimeBaseModel):
    resource: ResourceState


class CliEnvelope(RuntimeBaseModel):
    kind: Literal["result", "event"]
    payload: dict


class DownloadStep(RuntimeBaseModel):
    repo_id: str
    local_dir: Path
    allow_file_pattern: list[str] = Field(default_factory=list)


class DownloadTargetSpec(RuntimeBaseModel):
    target_id: str
    label: str
    provider: str
    repo_id: str
    allow_file_pattern: list[str]
    local_dir: Path
    cache_dir: Path
    resource_root: Path
    required_paths: list[str]
    required_file_paths: list[str] = Field(default_factory=list)
    required_dir_paths: list[str] = Field(default_factory=list)
    download_steps: list[DownloadStep] = Field(default_factory=list)
