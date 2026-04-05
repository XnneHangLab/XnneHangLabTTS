from xnnehanglab_tts.runtime.environment import inspect_environment


class FakeCuda:
    def __init__(self, available: bool):
        self._available = available

    def is_available(self) -> bool:
        return self._available


class FakeTorch:
    def __init__(self, version: str, cuda_available: bool):
        self.__version__ = version
        self.cuda = FakeCuda(cuda_available)


def test_inspect_environment_reports_gpu_when_cuda_is_available():
    result = inspect_environment(lambda: FakeTorch("2.6.0+cu118", True))

    assert result.mode == "gpu"
    assert result.torch_available is True
    assert result.cuda_available is True
    assert result.torch_version == "2.6.0+cu118"


def test_inspect_environment_reports_cpu_when_torch_import_fails():
    def raise_import_error():
        raise ImportError("torch missing")

    result = inspect_environment(raise_import_error)

    assert result.mode == "cpu"
    assert result.torch_available is False
    assert result.cuda_available is False
    assert result.issues == ["torch import failed: torch missing"]


class FakeCudaRaises:
    def is_available(self) -> bool:
        raise RuntimeError("cuda probe failed")


class FakeTorchCudaProbeFails:
    def __init__(self):
        self.__version__ = "2.6.0+cu118"
        self.cuda = FakeCudaRaises()


def test_inspect_environment_falls_back_to_cpu_when_cuda_probe_raises():
    result = inspect_environment(lambda: FakeTorchCudaProbeFails())

    assert result.mode == "cpu"
    assert result.torch_available is True
    assert result.cuda_available is False
    assert result.torch_version == "2.6.0+cu118"
    assert result.issues == ["torch cuda probe failed: cuda probe failed"]
