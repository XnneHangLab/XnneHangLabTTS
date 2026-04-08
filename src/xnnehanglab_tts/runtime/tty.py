"""Minimal stderr wrapper that lies isatty()=True so tqdm stays enabled
and uses \\r-overwrite mode even when the process's stderr is a pipe.
Every write is forwarded unchanged to the real stderr so the parent
process (Rust) sees all bytes and can split on \\r/\\n as needed.
"""

from __future__ import annotations


class FakeTtyStderr:
    def __init__(self, real) -> None:
        self._real = real

    def write(self, text: str) -> int:
        return self._real.write(text)

    def flush(self) -> None:
        try:
            self._real.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        return self._real.fileno()

    @property
    def encoding(self) -> str:
        return getattr(self._real, "encoding", "utf-8")

    @property
    def errors(self) -> str:
        return getattr(self._real, "errors", "replace")
