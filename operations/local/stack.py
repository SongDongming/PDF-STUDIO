#!/usr/bin/env python3
"""Safely manage the local multimodal PDF workbench.

The local delivery shape is intentionally split:

* Docker Compose owns only this project's data services.
* FastAPI and Vite run as host child processes tracked by PID files.
* Provider credentials are read from the repository credential store and are
  injected only into the child-process environment.

No credential value is printed or persisted by this script.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import ipaddress
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import ProxyHandler, build_opener

import yaml


SCRIPT_PATH = Path(__file__).resolve()
APP_ROOT = SCRIPT_PATH.parents[2]
REPOSITORY_ROOT = SCRIPT_PATH.parents[4]
BACKEND_ROOT = APP_ROOT / "backend"
INFRA_ROOT = APP_ROOT / "infra"
COMPOSE_FILE = INFRA_ROOT / "compose.yaml"
INFRA_ENV = INFRA_ROOT / ".env"
CREDENTIALS_FILE = REPOSITORY_ROOT / ".credentials.yaml"
RUNTIME_DIR = SCRIPT_PATH.parent / ".runtime"
STATE_FILE = RUNTIME_DIR / "state.json"
PROVIDER_SECRET_FILE = RUNTIME_DIR / "provider-secrets.json"

# Keep this workbench away from the conventional development port 8000, which
# is frequently used by course demos. The override remains available for
# operators who need a different dedicated port.
API_PORT = int(os.environ.get("PDFWIKI_API_PORT", "18800"))
FRONTEND_PORT = 4321
OCR_PORT = 18111

SERVICE_SPECS = {
    "api": {
        "marker": "uvicorn app.main:app",
        "cwd": BACKEND_ROOT,
        "command": [
            str(BACKEND_ROOT / ".venv" / "bin" / "uvicorn"),
            "app.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(API_PORT),
        ],
    },
    "frontend": {
        "marker": "npm run dev",
        "cwd": APP_ROOT,
        "command": ["npm", "run", "dev"],
    },
}


@dataclass(frozen=True)
class ProviderAvailability:
    moonshot: bool
    bailian: bool
    openai: bool

    @property
    def usable(self) -> bool:
        return self.moonshot and (self.bailian or self.openai)

    def summary(self) -> str:
        def state(value: bool) -> str:
            return "configured" if value else "missing"

        return (
            f"Moonshot={state(self.moonshot)}, "
            f"Bailian={state(self.bailian)}, "
            f"OpenAI={state(self.openai)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage the local multimodal PDF workbench without exposing secrets."
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    start = subparsers.add_parser("start", help="Start data, API and frontend")
    start.add_argument("--dry-run", action="store_true")
    start.add_argument("--lan-host", help="Explicit LAN IPv4 or hostname")
    start.add_argument("--skip-ocr", action="store_true")
    start.add_argument("--credentials", type=Path, default=CREDENTIALS_FILE)

    stop = subparsers.add_parser("stop", help="Stop only this project's services")
    stop.add_argument("--dry-run", action="store_true")
    stop.add_argument(
        "--keep-data-services",
        action="store_true",
        help="Leave this project's Compose data services running",
    )

    status = subparsers.add_parser("status", help="Show sanitized process and HTTP health")
    status.add_argument("--lan-host", help="Explicit LAN IPv4 or hostname")
    status.add_argument("--json", action="store_true", dest="as_json")

    doctor = subparsers.add_parser("doctor", help="Check prerequisites without mutations")
    doctor.add_argument("--lan-host", help="Explicit LAN IPv4 or hostname")
    doctor.add_argument("--credentials", type=Path, default=CREDENTIALS_FILE)
    doctor.add_argument("--json", action="store_true", dest="as_json")

    return parser.parse_args()


def ensure_runtime_dir() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(RUNTIME_DIR, 0o700)


def secure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label} does not exist: {path}")
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        raise RuntimeError(f"{label} must not be group/world accessible (current {oct(mode)})")


def read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key.strip()] = value
    return values


def nested_value(data: Any, dotted_path: str) -> str | None:
    current = data
    for segment in dotted_path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return None
        current = current[segment]
    if isinstance(current, str) and current.strip():
        return current.strip()
    return None


def first_nested(data: Any, paths: tuple[str, ...]) -> str | None:
    for path in paths:
        value = nested_value(data, path)
        if value:
            return value
    return None


def load_provider_environment(credentials_path: Path) -> tuple[dict[str, str], ProviderAvailability]:
    secure_file(credentials_path, "credential store")
    data = yaml.safe_load(credentials_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("credential store is not a mapping")

    moonshot = first_nested(
        data,
        (
            "api_keys.moonshot.key",
            "api_keys.moonshot.api_key",
            "api_keys.kimi_k3.key",
            "moonshot.key",
            "moonshot.api_key",
        ),
    )
    bailian = first_nested(
        data,
        (
            "api_keys.qwen.key",
            "api_keys.bailian.key",
            "api_keys.aliyun_bailian.key",
            "api_keys.dashscope.key",
            "bailian.key",
            "dashscope.api_key",
        ),
    )
    openai = first_nested(
        data,
        (
            "api_keys.openai.key",
            "api_keys.openai.api_key",
            "openai.key",
            "openai.api_key",
        ),
    )

    environment: dict[str, str] = {}
    if moonshot:
        environment["MOONSHOT_API_KEY"] = moonshot
        environment["KIMI_API_KEY"] = moonshot
        moonshot_base_url = first_nested(
            data,
            ("api_keys.moonshot.base_url", "moonshot.base_url"),
        )
        if moonshot_base_url:
            environment["MOONSHOT_BASE_URL"] = moonshot_base_url
    if bailian:
        environment["DASHSCOPE_API_KEY"] = bailian
        environment["BAILIAN_API_KEY"] = bailian
        bailian_base_url = first_nested(
            data,
            (
                "api_keys.qwen.base_url",
                "api_keys.bailian.base_url",
                "api_keys.dashscope.base_url",
            ),
        )
        if bailian_base_url:
            environment["DASHSCOPE_BASE_URL"] = bailian_base_url
    if openai:
        environment["OPENAI_API_KEY"] = openai
        openai_base_url = first_nested(
            data,
            ("api_keys.openai.base_url", "openai.base_url"),
        )
        if openai_base_url:
            environment["OPENAI_BASE_URL"] = openai_base_url

    availability = ProviderAvailability(
        moonshot=bool(moonshot),
        bailian=bool(bailian),
        openai=bool(openai),
    )
    return environment, availability


def load_runtime_provider_secrets() -> dict[str, str]:
    if not PROVIDER_SECRET_FILE.exists():
        return {}
    secure_file(PROVIDER_SECRET_FILE, "runtime provider secret store")
    try:
        payload = json.loads(PROVIDER_SECRET_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("runtime provider secret store is unreadable") from exc
    allowed = {
        "MOONSHOT_API_KEY",
        "KIMI_API_KEY",
        "DASHSCOPE_API_KEY",
        "BAILIAN_API_KEY",
        "OPENAI_API_KEY",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) - allowed
        or any(not isinstance(value, str) or not value for value in payload.values())
    ):
        raise RuntimeError("runtime provider secret store has an invalid shape")
    return dict(payload)


def detect_lan_host(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit

    candidates: list[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
            client.connect(("8.8.8.8", 80))
            candidates.append(client.getsockname()[0])
    except OSError:
        pass

    try:
        for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            candidates.append(item[4][0])
    except OSError:
        pass

    for candidate in candidates:
        try:
            address = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if address.version == 4 and address.is_private and not address.is_loopback:
            return candidate
    return None


def compose_command(*arguments: str) -> list[str]:
    return [
        "docker",
        "compose",
        "--env-file",
        str(INFRA_ENV),
        "-f",
        str(COMPOSE_FILE),
        *arguments,
    ]


def run_silent(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def command_exists(name: str) -> bool:
    result = run_silent(["/usr/bin/env", "which", name], timeout=5)
    return result.returncode == 0


def docker_healthy() -> bool:
    if not command_exists("docker"):
        return False
    return run_silent(["docker", "info"], timeout=10).returncode == 0


def http_healthy(url: str, timeout: float = 2.0) -> bool:
    try:
        opener = build_opener(ProxyHandler({}))
        with opener.open(url, timeout=timeout) as response:
            return 200 <= response.status < 400
    except (HTTPError, URLError, OSError, TimeoutError):
        return False


def port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for(predicate: Any, timeout: float, label: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.5)
    raise RuntimeError(f"{label} did not become healthy within {int(timeout)}s")


def pid_file(service: str) -> Path:
    return RUNTIME_DIR / f"{service}.pid"


def log_file(service: str) -> Path:
    return RUNTIME_DIR / f"{service}.log"


def read_pid(service: str) -> int | None:
    path = pid_file(service)
    if not path.exists():
        return None
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        path.unlink(missing_ok=True)
        return None


def process_command(pid: int) -> str:
    completed = run_silent(["ps", "-p", str(pid), "-o", "command="], timeout=5)
    return completed.stdout.strip() if completed.returncode == 0 else ""


def managed_process(service: str) -> tuple[int | None, bool]:
    pid = read_pid(service)
    if not pid:
        return None, False
    marker = str(SERVICE_SPECS[service]["marker"])
    return pid, marker in process_command(pid)


def open_private_log(service: str) -> Any:
    ensure_runtime_dir()
    descriptor = os.open(
        log_file(service),
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )
    os.chmod(log_file(service), 0o600)
    return os.fdopen(descriptor, "a", encoding="utf-8")


def start_process(service: str, environment: dict[str, str]) -> int:
    existing_pid, owned = managed_process(service)
    if existing_pid and owned:
        print(f"{service}: already running pid={existing_pid}")
        return existing_pid
    if existing_pid and not owned:
        raise RuntimeError(
            f"{service} PID file points to an unexpected process; refusing to replace it"
        )

    spec = SERVICE_SPECS[service]
    with open_private_log(service) as output:
        process = subprocess.Popen(
            list(spec["command"]),
            cwd=Path(spec["cwd"]),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
    pid_file(service).write_text(f"{process.pid}\n", encoding="utf-8")
    os.chmod(pid_file(service), 0o600)
    time.sleep(0.5)
    if process.poll() is not None:
        pid_file(service).unlink(missing_ok=True)
        raise RuntimeError(
            f"{service} exited during startup; inspect {log_file(service)}"
        )
    print(f"{service}: started pid={process.pid}")
    return process.pid


def stop_process(service: str, dry_run: bool = False) -> bool:
    pid, owned = managed_process(service)
    if not pid:
        print(f"{service}: stopped")
        return True
    if not owned:
        print(
            f"{service}: refusing to stop unexpected pid={pid}; inspect {pid_file(service)}",
            file=sys.stderr,
        )
        return False
    if dry_run:
        print(f"{service}: would stop managed pid={pid}")
        return True

    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pid_file(service).unlink(missing_ok=True)
        return True
    for _ in range(40):
        try:
            os.kill(pid, 0)
        except OSError:
            pid_file(service).unlink(missing_ok=True)
            print(f"{service}: stopped pid={pid}")
            return True
        time.sleep(0.25)
    print(f"{service}: pid={pid} did not stop after SIGTERM", file=sys.stderr)
    return False


def build_host_environment(
    infra_values: dict[str, str],
    provider_values: dict[str, str],
    lan_host: str | None,
) -> dict[str, str]:
    required = (
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "MINIO_ROOT_USER",
        "MINIO_ROOT_PASSWORD",
        "NEO4J_PASSWORD",
    )
    missing = [key for key in required if not infra_values.get(key)]
    if missing:
        raise RuntimeError(f"infra/.env is missing required keys: {', '.join(missing)}")

    origins = ["http://localhost:4321", "http://127.0.0.1:4321"]
    if lan_host:
        origins.append(f"http://{lan_host}:{FRONTEND_PORT}")

    environment = os.environ.copy()
    environment.update(provider_values)
    environment.update(
        {
            "APP_ENV": "development",
            "APP_HOST": "0.0.0.0",
            "APP_PORT": str(API_PORT),
            "APP_CORS_ORIGINS": json.dumps(origins, ensure_ascii=False),
            "APP_DATABASE_URL": (
                "postgresql+psycopg://"
                f"{quote_plus(infra_values['POSTGRES_USER'])}:"
                f"{quote_plus(infra_values['POSTGRES_PASSWORD'])}"
                f"@127.0.0.1:15432/{quote_plus(infra_values['POSTGRES_DB'])}"
            ),
            "APP_MINIO_ENDPOINT": "127.0.0.1:19000",
            "APP_MINIO_ACCESS_KEY": infra_values["MINIO_ROOT_USER"],
            "APP_MINIO_SECRET_KEY": infra_values["MINIO_ROOT_PASSWORD"],
            "APP_MINIO_SECURE": "false",
            "APP_MINIO_BUCKET": "pdfwiki",
            "APP_STORAGE_BACKEND": "minio",
            "APP_NEO4J_URI": "bolt://127.0.0.1:17687",
            "APP_NEO4J_USER": "neo4j",
            "APP_NEO4J_PASSWORD": infra_values["NEO4J_PASSWORD"],
            "APP_OCR_BASE_URL": f"http://127.0.0.1:{OCR_PORT}",
            "APP_COMPILE_MODE": "sync",
            "APP_PROVIDER_SECRET_FILE": str(PROVIDER_SECRET_FILE),
            # This workbench uses DeepSeek V4 Flash for chat/structured tasks
            # and Zhipu GLM embedding-3 for embeddings (SiliconFlow embedding
            # is configured in .local-secrets but is gated on account balance);
            # the model names are pinned here so the host API is self-contained.
            # Override via the parent environment when a different model is used.
            "MOONSHOT_CHAT_MODEL": os.environ.get("MOONSHOT_CHAT_MODEL", "deepseek-v4-flash"),
            "MOONSHOT_STRUCTURED_MODEL": os.environ.get(
                "MOONSHOT_STRUCTURED_MODEL", "deepseek-v4-flash"
            ),
            # Deep Agents relies on strict langchain structured output, which
            # the free GLM provider does not reliably produce; legacy is the
            # working runtime here. Override with deepagents if desired.
            "APP_AGENT_RUNTIME": os.environ.get("APP_AGENT_RUNTIME", "legacy"),
            "OPENAI_EMBEDDING_MODEL": os.environ.get(
                "OPENAI_EMBEDDING_MODEL", "embedding-3"
            ),
            # Embeddings use DashScope's native endpoint because it preserves
            # query/document text_type; do not reuse a chat-compatible base.
            "DASHSCOPE_BASE_URL": "https://dashscope.aliyuncs.com",
        }
    )
    return environment


def build_frontend_environment(lan_host: str | None) -> dict[str, str]:
    """Return a public-only environment for Vite.

    Vite only publishes VITE_* values, but the process itself must not inherit
    backend credentials either. Strip common secret-bearing environment names
    before adding the one public browser URL.
    """

    sensitive_fragments = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
    environment = {
        key: value
        for key, value in os.environ.items()
        if not any(fragment in key.upper() for fragment in sensitive_fragments)
    }
    environment["VITE_API_BASE_URL"] = (
        f"http://{lan_host}:{API_PORT}/api/v1"
        if lan_host
        else f"http://localhost:{API_PORT}/api/v1"
    )
    return environment


def ensure_infra_env(dry_run: bool) -> None:
    if INFRA_ENV.exists():
        secure_file(INFRA_ENV, "infra environment")
        return
    if dry_run:
        print("infra: would generate protected infra/.env")
        return
    completed = run_silent(
        [sys.executable, str(INFRA_ROOT / "bootstrap_env.py")],
        cwd=INFRA_ROOT,
        timeout=15,
    )
    if completed.returncode != 0:
        raise RuntimeError("failed to generate infra/.env")
    secure_file(INFRA_ENV, "infra environment")
    print("infra: generated protected environment")


def ocr_status() -> bool:
    return http_healthy(f"http://127.0.0.1:{OCR_PORT}/health")


def ensure_ocr(dry_run: bool, skip: bool) -> bool:
    # OCR is provided by the external PaddleOCR bridge (operations/local/
    # ocr_bridge.py, started by start_local.sh or start_docker.sh); this manager
    # only verifies it is healthy instead of managing its lifecycle.
    if skip:
        print("ocr: skipped by request")
        return False
    if ocr_status():
        print(f"ocr: healthy at http://127.0.0.1:{OCR_PORT}")
        return False
    if dry_run:
        print("ocr: would require the PaddleOCR bridge on 127.0.0.1:18111")
        return False
    raise RuntimeError(
        f"PaddleOCR bridge is not healthy at http://127.0.0.1:{OCR_PORT}; "
        "start operations/local/ocr_bridge.py first (start_local.sh does this automatically)"
    )


def write_state(lan_host: str | None) -> None:
    ensure_runtime_dir()
    content = {
        "version": 1,
        "lan_host": lan_host,
        "started_at": int(time.time()),
    }
    STATE_FILE.write_text(json.dumps(content, ensure_ascii=False) + "\n", encoding="utf-8")
    os.chmod(STATE_FILE, 0o600)


def read_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {}
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def print_start_plan(lan_host: str | None, skip_ocr: bool) -> None:
    print("dry-run: no processes or containers will be changed")
    print("1. validate Docker, local runtimes, protected infra and credential files")
    if skip_ocr:
        print("2. skip OCR bridge check by request")
    else:
        print("2. verify the PaddleOCR bridge on 127.0.0.1:18111")
    print("3. stop this Compose project's api service if running")
    print("4. start this project's postgres, minio and neo4j services")
    print(f"5. start host FastAPI on 0.0.0.0:{API_PORT}")
    print("6. start Vite on 0.0.0.0:4321")
    if lan_host:
        print(f"LAN origin: http://{lan_host}:{FRONTEND_PORT}")
        print(f"LAN API: http://{lan_host}:{API_PORT}/api/v1")
    else:
        print("LAN IP was not detected; pass --lan-host to configure LAN CORS explicitly")


def validate_prerequisites(credentials: Path) -> tuple[dict[str, str], ProviderAvailability]:
    if not COMPOSE_FILE.is_file():
        raise RuntimeError(f"Compose file is missing: {COMPOSE_FILE}")
    if not (BACKEND_ROOT / ".venv" / "bin" / "uvicorn").is_file():
        raise RuntimeError("backend virtual environment is missing uvicorn")
    if not command_exists("npm"):
        raise RuntimeError("npm is not installed")
    if not docker_healthy():
        raise RuntimeError("Docker is unavailable or the Docker daemon is not running")

    provider_values, availability = load_provider_environment(credentials)
    if not availability.usable:
        raise RuntimeError(
            "real provider configuration is incomplete; require Moonshot and "
            "at least one of Bailian/OpenAI"
        )
    return provider_values, availability


def start_stack(args: argparse.Namespace) -> int:
    lan_host = detect_lan_host(args.lan_host)
    provider_values, availability = validate_prerequisites(args.credentials)
    provider_values.update(load_runtime_provider_secrets())
    ensure_infra_env(args.dry_run)
    print(f"providers: {availability.summary()} (values hidden)")
    if args.dry_run:
        print_start_plan(lan_host, args.skip_ocr)
        return 0

    infra_values = read_dotenv(INFRA_ENV)
    environment = build_host_environment(infra_values, provider_values, lan_host)
    frontend_environment = build_frontend_environment(lan_host)
    _, frontend_owned = managed_process("frontend")
    if port_open("127.0.0.1", FRONTEND_PORT) and not frontend_owned:
        raise RuntimeError(
            "frontend port 4321 is already owned by an untracked process; "
            "stop that process explicitly before using this manager"
        )
    ensure_ocr(False, args.skip_ocr)
    write_state(lan_host)

    # The host API is the only process that receives provider credentials.
    # Stop only this Compose project's api before binding the dedicated
    # host API port. Unrelated Uvicorn processes are never stopped.
    _, api_owned = managed_process("api")
    if not api_owned:
        stopped = run_silent(
            compose_command("stop", "api"),
            cwd=INFRA_ROOT,
            timeout=60,
        )
        if stopped.returncode != 0:
            raise RuntimeError("could not stop this project's Compose api service")
        wait_for(
            lambda: not port_open("127.0.0.1", API_PORT),
            30,
            "Compose API port release",
        )

    started = run_silent(
        compose_command("up", "-d", "postgres", "minio", "neo4j"),
        cwd=INFRA_ROOT,
        timeout=180,
    )
    if started.returncode != 0:
        raise RuntimeError("could not start this project's Compose data services")
    print("infra: data services requested")

    wait_for(lambda: port_open("127.0.0.1", 15432), 90, "PostgreSQL")
    wait_for(
        lambda: http_healthy("http://127.0.0.1:19000/minio/health/live"),
        90,
        "MinIO",
    )
    wait_for(lambda: port_open("127.0.0.1", 17687), 120, "Neo4j")
    print("infra: data services reachable")

    newly_started: list[str] = []
    try:
        if not managed_process("api")[1]:
            start_process("api", environment)
            newly_started.append("api")
        if not managed_process("frontend")[1]:
            start_process("frontend", frontend_environment)
            newly_started.append("frontend")

        wait_for(
            lambda: http_healthy(f"http://127.0.0.1:{API_PORT}/api/v1/health"),
            45,
            "FastAPI",
        )
        wait_for(
            lambda: http_healthy(f"http://127.0.0.1:{FRONTEND_PORT}/"),
            45,
            "Vite",
        )
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        for service in reversed(newly_started):
            stop_process(service)
        raise
    write_state(lan_host)

    print(f"local UI: http://localhost:{FRONTEND_PORT}")
    print(f"local API: http://localhost:{API_PORT}/docs")
    if lan_host:
        print(f"LAN UI: http://{lan_host}:{FRONTEND_PORT}")
        print(f"LAN API: http://{lan_host}:{API_PORT}/docs")
        print(f"CORS: explicit LAN origin http://{lan_host}:{FRONTEND_PORT}")
    else:
        print("LAN IP not detected; restart with --lan-host to add its explicit CORS origin")
    return 0


def stop_stack(args: argparse.Namespace) -> int:
    success = True
    for service in ("frontend", "api"):
        success = stop_process(service, args.dry_run) and success

    state = read_state()
    print("ocr: left unchanged (managed by the external PaddleOCR bridge)")

    if args.keep_data_services:
        print("infra: left running by request")
    elif args.dry_run:
        print("infra: would run Compose down without deleting volumes")
    elif INFRA_ENV.exists():
        completed = run_silent(
            compose_command("down", "--remove-orphans"),
            cwd=INFRA_ROOT,
            timeout=120,
        )
        if completed.returncode == 0:
            print("infra: stopped this Compose project; data volumes preserved")
        else:
            print("infra: Compose down failed", file=sys.stderr)
            success = False
    else:
        print("infra: no infra/.env; Compose left unchanged")

    if not args.dry_run and success:
        STATE_FILE.unlink(missing_ok=True)
    return 0 if success else 1


def compose_status() -> dict[str, str]:
    if not INFRA_ENV.exists() or not docker_healthy():
        return {}
    completed = run_silent(
        compose_command("ps", "--format", "json"),
        cwd=INFRA_ROOT,
        timeout=15,
    )
    if completed.returncode != 0:
        return {}
    services: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        service = str(row.get("Service", "unknown"))
        state = str(row.get("State", "unknown"))
        health = str(row.get("Health", "")).strip()
        services[service] = f"{state}/{health}" if health else state
    return services


def collect_status(lan_host: str | None) -> dict[str, Any]:
    processes: dict[str, Any] = {}
    for service in SERVICE_SPECS:
        pid, owned = managed_process(service)
        processes[service] = {
            "running": bool(pid and owned),
            "pid": pid if pid and owned else None,
            "pid_conflict": bool(pid and not owned),
        }

    health = {
        "api": http_healthy(f"http://127.0.0.1:{API_PORT}/api/v1/health"),
        "frontend": http_healthy(f"http://127.0.0.1:{FRONTEND_PORT}/"),
        "ocr": ocr_status(),
    }
    urls: dict[str, str] = {
        "local_ui": f"http://localhost:{FRONTEND_PORT}",
        "local_api": f"http://localhost:{API_PORT}/docs",
        "ocr": f"http://127.0.0.1:{OCR_PORT}",
    }
    if lan_host:
        urls["lan_ui"] = f"http://{lan_host}:{FRONTEND_PORT}"
        urls["lan_api"] = f"http://{lan_host}:{API_PORT}/docs"

    return {
        "docker": docker_healthy(),
        "processes": processes,
        "http_health": health,
        "compose": compose_status(),
        "urls": urls,
        "lan_host": lan_host,
    }


def print_status(data: dict[str, Any]) -> None:
    print(f"docker: {'healthy' if data['docker'] else 'unavailable'}")
    for service, process in data["processes"].items():
        if process["running"]:
            print(f"{service}: managed pid={process['pid']}")
        elif process["pid_conflict"]:
            print(f"{service}: PID conflict (not managed)")
        else:
            print(f"{service}: no managed process")
    for service, healthy in data["http_health"].items():
        print(f"{service} HTTP: {'healthy' if healthy else 'unhealthy'}")
    if data["compose"]:
        compact = ", ".join(
            f"{name}={state}" for name, state in sorted(data["compose"].items())
        )
        print(f"compose: {compact}")
    else:
        print("compose: no running services detected")
    for label, url in data["urls"].items():
        print(f"{label}: {url}")
    if not data["lan_host"]:
        print("LAN host was not detected; use --lan-host for explicit URLs and CORS")


def doctor(args: argparse.Namespace) -> int:
    lan_host = detect_lan_host(args.lan_host)
    checks: dict[str, Any] = {
        "docker": docker_healthy(),
        "npm": command_exists("npm"),
        "backend_venv": (BACKEND_ROOT / ".venv" / "bin" / "uvicorn").is_file(),
        "compose_file": COMPOSE_FILE.is_file(),
        "infra_env": INFRA_ENV.is_file(),
        "ocr_tunnel": ocr_status(),
        "lan_host": lan_host,
    }
    try:
        _, availability = load_provider_environment(args.credentials)
        checks["providers"] = {
            "moonshot": availability.moonshot,
            "bailian": availability.bailian,
            "openai": availability.openai,
        }
        checks["provider_fallback_ready"] = availability.usable
    except (OSError, RuntimeError, yaml.YAMLError):
        checks["providers"] = {
            "moonshot": False,
            "bailian": False,
            "openai": False,
        }
        checks["provider_fallback_ready"] = False

    if args.as_json:
        print(json.dumps(checks, ensure_ascii=False, indent=2))
    else:
        for key, value in checks.items():
            if isinstance(value, dict):
                compact = ", ".join(
                    f"{name}={'configured' if present else 'missing'}"
                    for name, present in value.items()
                )
                print(f"{key}: {compact} (values hidden)")
            else:
                print(f"{key}: {value}")
    required_ok = all(
        bool(checks[key])
        for key in ("docker", "npm", "backend_venv", "compose_file", "infra_env")
    )
    return 0 if required_ok and checks["provider_fallback_ready"] else 1


def main() -> int:
    args = parse_args()
    try:
        if args.action == "start":
            return start_stack(args)
        if args.action == "stop":
            return stop_stack(args)
        if args.action == "status":
            data = collect_status(detect_lan_host(args.lan_host))
            if args.as_json:
                print(json.dumps(data, ensure_ascii=False, indent=2))
            else:
                print_status(data)
            return 0
        return doctor(args)
    except (OSError, RuntimeError, subprocess.TimeoutExpired, yaml.YAMLError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
