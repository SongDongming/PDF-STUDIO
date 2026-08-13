import os

import pytest
from fastapi.testclient import TestClient

os.environ["APP_ENV"] = "test"
os.environ["APP_DATABASE_URL"] = "sqlite+pysqlite:///:memory:"
os.environ["APP_CORS_ORIGINS"] = '["http://localhost:4321"]'

from app.main import app  # noqa: E402
from app.store import store  # noqa: E402


@pytest.fixture(autouse=True)
def reset_store() -> None:
    store.reset()


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client

