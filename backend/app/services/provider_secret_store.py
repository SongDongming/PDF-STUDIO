"""Atomic local storage for user-supplied provider credentials.

The file path is injected by the operations layer, is Git-ignored, and must
remain owner-readable only.  Public APIs receive a write-only secret and return
only provider metadata; no read path exposes credential values.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from threading import RLock


ALLOWED_ENV_KEYS = frozenset(
    {
        "MOONSHOT_API_KEY",
        "KIMI_API_KEY",
        "DASHSCOPE_API_KEY",
        "BAILIAN_API_KEY",
        "OPENAI_API_KEY",
    }
)


class ProviderSecretStoreError(RuntimeError):
    pass


class ProviderSecretStore:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._lock = RLock()

    def update(self, values: dict[str, str]) -> None:
        if not values or set(values) - ALLOWED_ENV_KEYS:
            raise ProviderSecretStoreError("credential update contains unsupported keys")
        if any(not value.strip() for value in values.values()):
            raise ProviderSecretStoreError("credential value cannot be blank")
        with self._lock:
            current = self._read()
            current.update({key: value.strip() for key, value in values.items()})
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self.path.name}.",
                dir=self.path.parent,
            )
            temporary = Path(temporary_name)
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(current, handle, ensure_ascii=False, sort_keys=True)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
                os.chmod(self.path, 0o600)
            finally:
                temporary.unlink(missing_ok=True)

    def _read(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        if self.path.stat().st_mode & 0o077:
            raise ProviderSecretStoreError(
                "credential file must not be group/world accessible"
            )
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProviderSecretStoreError("credential file is unreadable") from exc
        if not isinstance(payload, dict) or set(payload) - ALLOWED_ENV_KEYS:
            raise ProviderSecretStoreError("credential file has an invalid shape")
        if any(not isinstance(value, str) or not value for value in payload.values()):
            raise ProviderSecretStoreError("credential file contains an invalid value")
        return dict(payload)
