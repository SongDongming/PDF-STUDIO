from fastapi.testclient import TestClient


def test_health_and_openapi_are_parseable(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    schema = openapi.json()
    assert schema["info"]["title"] == "多模态 PDF 知识库"
    required_paths = {
        "/api/v1/knowledge-bases",
        "/api/v1/jobs",
        "/api/v1/chat/threads",
        "/api/v1/settings",
    }
    assert required_paths <= set(schema["paths"])


def test_cors_allows_configured_origin_only(client: TestClient) -> None:
    allowed = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://localhost:4321",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:4321"

    denied = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://attacker.invalid",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert "access-control-allow-origin" not in denied.headers

