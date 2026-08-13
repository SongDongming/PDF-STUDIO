import os

import pytest
from fastapi.testclient import TestClient


def create_kb(client: TestClient) -> dict:
    return client.post(
        "/api/v1/knowledge-bases",
        json={"name": "多模态 PDF", "description": ""},
    ).json()


def test_thread_message_and_sse_contract(client: TestClient) -> None:
    kb = create_kb(client)
    thread_response = client.post(
        "/api/v1/chat/threads",
        json={"knowledge_base_id": kb["id"], "title": "图表问答"},
    )
    assert thread_response.status_code == 201
    thread = thread_response.json()

    answer = client.post(
        f"/api/v1/chat/threads/{thread['id']}/messages",
        json={"content": "图 3 表达了什么？"},
    )
    assert answer.status_code == 201
    assert answer.json()["role"] == "assistant"
    assert answer.json()["blocks"][0]["type"] == "text"

    messages = client.get(f"/api/v1/chat/threads/{thread['id']}/messages").json()
    assert [item["role"] for item in messages] == ["user", "assistant"]

    with client.stream(
        "POST",
        f"/api/v1/chat/threads/{thread['id']}/messages/stream",
        json={"content": "继续解释"},
    ) as stream:
        body = "".join(stream.iter_text())
    assert "event: answer.completed" in body
    assert "answer.completed" in body


def test_graph_and_wiki_contract(client: TestClient) -> None:
    kb = create_kb(client)
    graph = client.get(f"/api/v1/knowledge-bases/{kb['id']}/graph")
    assert graph.status_code == 200
    assert graph.json()["nodes"][0]["properties"]["root"] is True

    wiki = client.get(f"/api/v1/knowledge-bases/{kb['id']}/wiki/pages")
    assert wiki.status_code == 200
    assert wiki.json() == []


def test_settings_never_expose_secret_values(client: TestClient) -> None:
    response = client.get("/api/v1/settings")
    assert response.status_code == 200
    body = response.text.lower()
    assert "api_key" not in body
    assert "password" not in body
    assert "secret" not in body

    provider = response.json()["providers"][0]
    connection = client.post(
        "/api/v1/settings/connection-tests",
        json={"provider_id": provider["id"]},
    )
    assert connection.status_code == 200
    assert connection.json()["reachable"] is False


@pytest.mark.skipif(
    os.name == "nt",
    reason="Unix file modes are not enforced on Windows; verified under Linux (Docker/WSL)",
)
def test_provider_credential_update_is_write_only_and_owner_protected(
    client: TestClient, monkeypatch, tmp_path
) -> None:
    secret_file = tmp_path / "provider-secrets.json"
    monkeypatch.setenv("APP_PROVIDER_SECRET_FILE", str(secret_file))
    monkeypatch.setenv("MOONSHOT_API_KEY", "previous-test-value")
    monkeypatch.setenv("KIMI_API_KEY", "previous-test-value")

    response = client.post(
        "/api/v1/settings/credentials",
        json={"provider_id": "vision-chat", "api_key": "replacement-test-value"},
    )

    assert response.status_code == 200
    assert response.json()["configured"] is True
    assert "replacement-test-value" not in response.text
    assert secret_file.stat().st_mode & 0o777 == 0o600
