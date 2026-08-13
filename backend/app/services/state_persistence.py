"""Durable metadata snapshots for the development ``MemoryStore``.

The production-oriented adapter stores one versioned snapshot per namespace in
SQLAlchemy (PostgreSQL is the intended deployment).  A SQLite database can be
used explicitly for a single-machine development deployment.  Neither adapter
persists credential material.

The module deliberately does not import ``MemoryStore``.  ``save`` accepts
either a mapping or any object exposing the six public state attributes, which
keeps the persistence boundary usable by the store without creating an import
cycle.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from threading import RLock
from typing import Any, Protocol

from pydantic import SecretStr
from sqlalchemy import (
    JSON,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Column,
    create_engine,
    select,
)
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine


SCHEMA_VERSION = 2
STATE_FIELDS = (
    "knowledge_bases",
    "documents",
    "jobs",
    "threads",
    "messages",
    "wiki_pages",
    "settings",
)
LEGACY_V1_FIELDS = tuple(field for field in STATE_FIELDS if field != "wiki_pages")

_DATETIME_TAG = "__pdfwiki_datetime__"
_ALLOWED_SENSITIVE_METADATA = frozenset({"credential_ref", "configured"})
_SENSITIVE_EXACT = frozenset(
    {
        "access_key",
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "bearer",
        "client_secret",
        "credential",
        "credential_value",
        "credentials",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "secret_key",
        "token",
    }
)
_SENSITIVE_PARTS = frozenset(
    {"authorization", "credential", "password", "secret", "token"}
)


class StatePersistenceError(RuntimeError):
    """Base class for durable state failures."""


class CorruptStateError(StatePersistenceError):
    """Stored state is malformed or fails its integrity check."""


class UnsupportedSchemaVersionError(StatePersistenceError):
    """Stored state was produced by an unsupported schema version."""


class ConcurrentStateUpdateError(StatePersistenceError):
    """An optimistic save observed a different persisted revision."""


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """Validated state plus persistence metadata."""

    schema_version: int
    revision: int
    saved_at: datetime
    state: dict[str, Any]


class StatePersistence(Protocol):
    def load(self) -> StateSnapshot | None:
        """Load a validated snapshot, returning ``None`` when none exists."""

    def save(
        self, source: Mapping[str, Any] | object, *, expected_revision: int | None = None
    ) -> StateSnapshot:
        """Persist a sanitized snapshot atomically."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _normalized_key(value: object) -> str:
    return str(value).strip().lower().replace("-", "_")


def _is_sensitive_key(key: object) -> bool:
    normalized = _normalized_key(key)
    if normalized in _ALLOWED_SENSITIVE_METADATA:
        return False
    if normalized in _SENSITIVE_EXACT:
        return True
    parts = frozenset(part for part in normalized.split("_") if part)
    return bool(parts & _SENSITIVE_PARTS)


def _sanitize(value: Any) -> Any:
    """Return a JSON-safe copy with credential-bearing fields removed."""

    if isinstance(value, SecretStr):
        # A secret-bearing object is never converted to its underlying value.
        return None
    if isinstance(value, datetime):
        return {_DATETIME_TAG: _normalize_datetime(value).isoformat()}
    if isinstance(value, Enum):
        return _sanitize(value.value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            if _is_sensitive_key(key):
                continue
            result[key] = _sanitize(child)
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_sanitize(child) for child in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise StatePersistenceError(
        f"unsupported state value type: {type(value).__name__}"
    )


def _restore_types(value: Any) -> Any:
    if isinstance(value, list):
        return [_restore_types(child) for child in value]
    if isinstance(value, dict):
        if set(value) == {_DATETIME_TAG}:
            raw = value[_DATETIME_TAG]
            if not isinstance(raw, str):
                raise CorruptStateError("invalid datetime marker in persisted state")
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError as exc:
                raise CorruptStateError(
                    "invalid datetime value in persisted state"
                ) from exc
            return _normalize_datetime(parsed)
        return {str(key): _restore_types(child) for key, child in value.items()}
    return value


def _extract_state(source: Mapping[str, Any] | object) -> dict[str, Any]:
    state: dict[str, Any] = {}
    if isinstance(source, Mapping):
        missing = [field for field in STATE_FIELDS if field not in source]
        if missing:
            raise StatePersistenceError(
                f"state is missing required fields: {', '.join(missing)}"
            )
        for field in STATE_FIELDS:
            state[field] = source[field]
    else:
        missing = [field for field in STATE_FIELDS if not hasattr(source, field)]
        if missing:
            raise StatePersistenceError(
                f"state source is missing attributes: {', '.join(missing)}"
            )
        # A MemoryStore owns an RLock.  Taking it while copying produces one
        # coherent in-process snapshot without coupling this module to the
        # concrete store class.
        source_lock = getattr(source, "_lock", None)
        if source_lock is not None:
            with source_lock:
                for field in STATE_FIELDS:
                    state[field] = deepcopy(getattr(source, field))
        else:
            for field in STATE_FIELDS:
                state[field] = deepcopy(getattr(source, field))
    sanitized = _sanitize(state)
    if not isinstance(sanitized, dict):
        raise StatePersistenceError("sanitized state is not an object")
    return sanitized


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StatePersistenceError("state is not valid JSON metadata") from exc


def _checksum(state: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(state)).hexdigest()


def _validate_state(
    state: Any, *, fields: tuple[str, ...] = STATE_FIELDS
) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise CorruptStateError("persisted state must be an object")
    missing = [field for field in fields if field not in state]
    extra = [field for field in state if field not in fields]
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if extra:
            details.append(f"unknown fields: {', '.join(extra)}")
        raise CorruptStateError("; ".join(details))
    for collection in fields:
        if collection == "settings":
            continue
        if not isinstance(state[collection], dict):
            raise CorruptStateError(f"{collection} must be an object")
    if not isinstance(state["settings"], dict):
        raise CorruptStateError("settings must be an object")
    # Sensitive keys in an existing snapshot indicate a broken writer or
    # manual tampering.  Refuse to load rather than silently normalizing it.
    _assert_no_sensitive_keys(state)
    return state


def _assert_no_sensitive_keys(value: Any, path: str = "state") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if _is_sensitive_key(key):
                raise CorruptStateError(
                    f"sensitive field found in persisted state at {path}.{key}"
                )
            _assert_no_sensitive_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_sensitive_keys(child, f"{path}[{index}]")


def _parse_saved_at(raw: Any) -> datetime:
    if isinstance(raw, datetime):
        return _normalize_datetime(raw)
    if not isinstance(raw, str):
        raise CorruptStateError("saved_at must be an ISO-8601 timestamp")
    try:
        return _normalize_datetime(datetime.fromisoformat(raw.replace("Z", "+00:00")))
    except ValueError as exc:
        raise CorruptStateError("saved_at is not a valid ISO-8601 timestamp") from exc


def _build_snapshot(
    *,
    schema_version: Any,
    revision: Any,
    saved_at: Any,
    state: Any,
    checksum: Any,
) -> StateSnapshot:
    if schema_version not in {1, SCHEMA_VERSION}:
        raise UnsupportedSchemaVersionError(
            f"unsupported state schema version: {schema_version!r}"
        )
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        raise CorruptStateError("revision must be a positive integer")
    validation_fields = LEGACY_V1_FIELDS if schema_version == 1 else STATE_FIELDS
    validated = _validate_state(state, fields=validation_fields)
    if not isinstance(checksum, str) or checksum != _checksum(validated):
        raise CorruptStateError("persisted state checksum mismatch")
    if schema_version == 1:
        # V1 did not persist generated Wiki pages.  Upgrade in memory without
        # discarding documents, sessions, jobs, or settings; the next save
        # atomically publishes the V2 envelope.
        validated = {**validated, "wiki_pages": {}}
    return StateSnapshot(
        schema_version=SCHEMA_VERSION,
        revision=revision,
        saved_at=_parse_saved_at(saved_at),
        state=_restore_types(deepcopy(validated)),
    )


class SqlAlchemyStatePersistence:
    """Transactional snapshot persistence for PostgreSQL or SQLite.

    PostgreSQL is the deployment target.  File-backed SQLite is suitable for a
    single-machine development build.  In-memory SQLite does not survive a
    process restart and should only be used by tests.
    """

    def __init__(
        self,
        database: str | Engine,
        *,
        namespace: str = "default",
        table_name: str = "application_state_snapshots",
    ) -> None:
        if not namespace.strip():
            raise ValueError("namespace cannot be empty")
        self.namespace = namespace
        self.engine = (
            database
            if isinstance(database, Engine)
            else create_engine(database, pool_pre_ping=True)
        )
        self._lock = RLock()
        metadata = MetaData()
        self._table = Table(
            table_name,
            metadata,
            Column("namespace", String(200), primary_key=True),
            Column("schema_version", Integer, nullable=False),
            Column("revision", Integer, nullable=False),
            Column("saved_at", DateTime(timezone=True), nullable=False),
            Column("state_json", JSON, nullable=False),
            Column("checksum", String(71), nullable=False),
        )
        metadata.create_all(self.engine)

    def load(self) -> StateSnapshot | None:
        with self._lock, self.engine.connect() as connection:
            row = (
                connection.execute(
                    select(self._table).where(
                        self._table.c.namespace == self.namespace
                    )
                )
                .mappings()
                .one_or_none()
            )
        if row is None:
            return None
        return _build_snapshot(
            schema_version=row["schema_version"],
            revision=row["revision"],
            saved_at=row["saved_at"],
            state=row["state_json"],
            checksum=row["checksum"],
        )

    def save(
        self, source: Mapping[str, Any] | object, *, expected_revision: int | None = None
    ) -> StateSnapshot:
        state = _extract_state(source)
        saved_at = _utc_now()
        checksum = _checksum(state)
        values = {
            "namespace": self.namespace,
            "schema_version": SCHEMA_VERSION,
            "revision": 1,
            "saved_at": saved_at,
            "state_json": state,
            "checksum": checksum,
        }
        dialect = self.engine.dialect.name
        with self._lock, self.engine.begin() as connection:
            if dialect == "postgresql":
                insert_statement = postgresql_insert(self._table).values(**values)
            elif dialect == "sqlite":
                insert_statement = sqlite_insert(self._table).values(**values)
            else:
                raise StatePersistenceError(
                    f"unsupported SQLAlchemy persistence dialect: {dialect}"
                )
            if expected_revision == 0:
                # This is an atomic "create only if absent".  A prior SELECT
                # cannot lock a row that does not yet exist and would race
                # across worker processes.
                create_statement = insert_statement.on_conflict_do_nothing(
                    index_elements=[self._table.c.namespace]
                ).returning(self._table.c.revision)
                created = connection.execute(create_statement).scalar_one_or_none()
                if created is None:
                    raise ConcurrentStateUpdateError(
                        "expected no persisted state, but one already exists"
                    )
                revision = int(created)
            elif expected_revision is not None:
                # Compare-and-swap in one UPDATE statement is safe across
                # processes for both PostgreSQL and SQLite.
                revision = expected_revision + 1
                update_result = connection.execute(
                    self._table.update()
                    .where(
                        self._table.c.namespace == self.namespace,
                        self._table.c.revision == expected_revision,
                    )
                    .values(
                        schema_version=SCHEMA_VERSION,
                        revision=revision,
                        saved_at=saved_at,
                        state_json=state,
                        checksum=checksum,
                    )
                )
                if update_result.rowcount != 1:
                    raise ConcurrentStateUpdateError(
                        f"expected revision {expected_revision}, "
                        "but persisted state changed"
                    )
            else:
                statement = insert_statement.on_conflict_do_update(
                    index_elements=[self._table.c.namespace],
                    set_={
                        "schema_version": SCHEMA_VERSION,
                        "revision": self._table.c.revision + 1,
                        "saved_at": saved_at,
                        "state_json": state,
                        "checksum": checksum,
                    },
                ).returning(self._table.c.revision)
                revision = int(connection.execute(statement).scalar_one())
        return StateSnapshot(
            schema_version=SCHEMA_VERSION,
            revision=revision,
            saved_at=saved_at,
            state=_restore_types(deepcopy(state)),
        )
