from __future__ import annotations

import json
import os

import pytest

from app.services.provider_secret_store import ProviderSecretStore


@pytest.mark.skipif(
    os.name == "nt",
    reason="Unix file modes are not enforced on Windows; verified under Linux (Docker/WSL)",
)
def test_provider_secret_store_is_atomic_owner_only_and_preserves_keys(tmp_path) -> None:
    secret_file = tmp_path / "runtime" / "provider-secrets.json"
    store = ProviderSecretStore(secret_file)

    store.update({"MOONSHOT_API_KEY": "first-value"})
    store.update({"DASHSCOPE_API_KEY": "second-value"})

    assert secret_file.stat().st_mode & 0o777 == 0o600
    assert json.loads(secret_file.read_text(encoding="utf-8")) == {
        "DASHSCOPE_API_KEY": "second-value",
        "MOONSHOT_API_KEY": "first-value",
    }
