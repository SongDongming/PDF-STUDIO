import asyncio
import os
from copy import deepcopy

from fastapi import APIRouter, HTTPException, status

from app.config import get_settings as get_runtime_settings
from app.schemas import (
    ConnectionTestRequest,
    ConnectionTestResponse,
    ProviderCredentialStatus,
    ProviderCredentialUpdate,
    SettingsPatch,
    SystemSettings,
    utc_now,
)
from app.services.embeddings import BailianEmbeddingProvider, OpenAIEmbeddingProvider
from app.services.ocr_client import OCRServiceError, PaddleOCRHttpClient
from app.services.provider_secret_store import (
    ProviderSecretStore,
    ProviderSecretStoreError,
)
from app.services.providers import (
    DeepSeekProvider,
    ProviderConfigurationError,
    ProviderUnavailableError,
)
from app.store import store

router = APIRouter(prefix="/settings", tags=["settings"])

_CREDENTIAL_ENV_KEYS = {
    "embedding-primary": ("DASHSCOPE_API_KEY", "BAILIAN_API_KEY"),
    "embedding-openai": ("OPENAI_API_KEY",),
    "vision-chat": ("MOONSHOT_API_KEY", "KIMI_API_KEY"),
    "answer-primary": ("MOONSHOT_API_KEY", "KIMI_API_KEY"),
}


def _configured(provider_id: str) -> bool:
    runtime = get_runtime_settings()
    if runtime.env == "test":
        return False
    if provider_id == "embedding-primary":
        return bool(os.getenv("DASHSCOPE_API_KEY"))
    if provider_id == "embedding-openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    if provider_id == "ocr-primary":
        return bool(runtime.ocr_base_url)
    if provider_id in {"vision-chat", "answer-primary"}:
        return bool(os.getenv("MOONSHOT_API_KEY"))
    return False


def _public_settings() -> dict:
    payload = deepcopy(store.settings)
    for provider in payload["providers"]:
        provider["configured"] = _configured(provider["id"])
        if not provider["enabled"]:
            provider["health"] = "disabled"
        elif not provider["configured"]:
            provider["health"] = "unavailable"
        # A settings read is deliberately not a network probe.
        elif provider["health"] == "disabled":
            provider["health"] = "unknown"
        provider["credential_ref"] = None
    from app.services.agent_runtime_registry import agent_runtime_registry

    deep = agent_runtime_registry.build
    if deep is None:
        deep = agent_runtime_registry.refresh(store)
    persistence = agent_runtime_registry.persistence
    persistence_detail = (
        f"；状态存储：{persistence.backend}"
        if persistence is not None
        else ""
    )
    payload["agent_framework"] = {
        "name": "deepagents",
        "mode": (
            "deepagents"
            if deep.status.available
            else "bounded_grounding_validator"
        ),
        "available": deep.status.available,
        "code": deep.status.code,
        "versions": dict(deep.status.versions),
        "detail": deep.status.detail + persistence_detail,
    }
    return payload


@router.get("", response_model=SystemSettings, operation_id="getSettings")
def get_settings() -> SystemSettings:
    return SystemSettings(**_public_settings())


@router.patch("", response_model=SystemSettings, operation_id="updateSettings")
def update_settings(payload: SettingsPatch) -> SystemSettings:
    changes = payload.model_dump(exclude_none=True)
    # Provider entries contain only display/configuration metadata.  Secret
    # material is never accepted by this public contract.
    store.settings.update(deepcopy(changes))
    store.settings["updated_at"] = utc_now()
    store.persist_state()
    return SystemSettings(**_public_settings())


@router.post(
    "/credentials",
    response_model=ProviderCredentialStatus,
    operation_id="updateProviderCredential",
)
def update_provider_credential(
    payload: ProviderCredentialUpdate,
) -> ProviderCredentialStatus:
    secret_path = os.getenv("APP_PROVIDER_SECRET_FILE")
    if not secret_path:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "credential_store_unavailable",
                "message": "服务端未配置受保护的凭证存储",
            },
        )
    keys = _CREDENTIAL_ENV_KEYS[payload.provider_id]
    value = payload.api_key.get_secret_value().strip()
    if not value:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "invalid_credential", "message": "API Key 不能为空"},
        )
    try:
        ProviderSecretStore(secret_path).update({key: value for key in keys})
    except ProviderSecretStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "credential_store_failed",
                "message": "受保护凭证写入失败",
            },
        ) from exc
    for key in keys:
        os.environ[key] = value
    if payload.provider_id in {"vision-chat", "answer-primary"}:
        from app.services.agent_runtime_registry import agent_runtime_registry

        agent_runtime_registry.refresh(store)
    return ProviderCredentialStatus(
        provider_id=payload.provider_id,
        configured=True,
        detail="凭证已写入受保护存储；当前 API 进程已立即生效",
    )


@router.post(
    "/connection-tests",
    response_model=ConnectionTestResponse,
    operation_id="testProviderConnection",
)
async def test_connection(payload: ConnectionTestRequest) -> ConnectionTestResponse:
    provider = next(
        (
            item
            for item in store.settings["providers"]
            if item["id"] == payload.provider_id
        ),
        None,
    )
    if provider is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "not_found", "message": "模型配置不存在"},
        )
    if not provider["enabled"]:
        return ConnectionTestResponse(
            provider_id=payload.provider_id,
            reachable=False,
            health="disabled",
            detail="该模型配置当前已禁用",
        )
    if not _configured(payload.provider_id):
        return ConnectionTestResponse(
            provider_id=payload.provider_id,
            reachable=False,
            health="unavailable",
            detail="服务端尚未绑定受保护凭证或服务地址",
        )
    try:
        if payload.provider_id == "embedding-primary":
            health = await BailianEmbeddingProvider.from_env().health()
            reachable, detail = health.healthy, health.detail
        elif payload.provider_id == "embedding-openai":
            health = await OpenAIEmbeddingProvider.from_env().health()
            reachable, detail = health.healthy, health.detail
        elif payload.provider_id == "ocr-primary":
            runtime = get_runtime_settings()
            client = PaddleOCRHttpClient(runtime.ocr_base_url or "")
            try:
                await asyncio.to_thread(client.health)
                reachable, detail = True, "PaddleOCR 服务可访问"
            finally:
                client.close()
        else:
            health = await DeepSeekProvider.from_env().health()
            reachable, detail = health.healthy, health.detail
    except (
        ProviderConfigurationError,
        ProviderUnavailableError,
        OCRServiceError,
        ValueError,
    ):
        reachable, detail = False, "服务连接失败，请检查受保护的服务端配置"
    return ConnectionTestResponse(
        provider_id=payload.provider_id,
        reachable=reachable,
        health="healthy" if reachable else "unavailable",
        detail=detail,
    )
