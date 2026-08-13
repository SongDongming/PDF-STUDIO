from __future__ import annotations

import json
import mimetypes
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Protocol


class ObjectStorageError(RuntimeError):
    """Raised when an object cannot be safely persisted or retrieved."""


@dataclass(frozen=True, slots=True)
class StoredObject:
    key: str
    size: int
    content_type: str
    uri: str


class ObjectStorage(Protocol):
    def put_bytes(
        self, key: str, payload: bytes, content_type: str = "application/octet-stream"
    ) -> StoredObject: ...

    def get_bytes(self, key: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...

    def uri_for(self, key: str) -> str: ...

    def delete_prefix(self, prefix: str) -> None: ...


def normalize_object_key(key: str) -> str:
    """Return a portable object key and reject traversal/absolute paths."""

    normalized = PurePosixPath(key.replace("\\", "/"))
    if (
        not key
        or normalized.is_absolute()
        or any(part in {"", ".", ".."} for part in normalized.parts)
    ):
        raise ObjectStorageError(f"unsafe object key: {key!r}")
    return normalized.as_posix()


class LocalObjectStorage:
    """Filesystem-backed object storage for development and offline tests."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        normalized = normalize_object_key(key)
        candidate = (self.root / normalized).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ObjectStorageError(f"object key escapes storage root: {key!r}")
        return candidate

    def put_bytes(
        self, key: str, payload: bytes, content_type: str = "application/octet-stream"
    ) -> StoredObject:
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        try:
            temporary.write_bytes(payload)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return StoredObject(
            key=normalize_object_key(key),
            size=len(payload),
            content_type=content_type,
            uri=path.as_uri(),
        )

    def get_bytes(self, key: str) -> bytes:
        path = self._path_for(key)
        try:
            return path.read_bytes()
        except OSError as exc:
            raise ObjectStorageError(f"unable to read object {key!r}") from exc

    def exists(self, key: str) -> bool:
        return self._path_for(key).is_file()

    def delete_prefix(self, prefix: str) -> None:
        import shutil

        normalized = normalize_object_key(prefix)
        target = (self.root / normalized).resolve()
        if target != self.root and self.root in target.parents:
            shutil.rmtree(target, ignore_errors=True)

    def uri_for(self, key: str) -> str:
        return self._path_for(key).as_uri()


class MinioObjectStorage:
    """Small MinIO adapter with the same semantics as LocalObjectStorage."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        *,
        secure: bool = False,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                from minio import Minio
            except ImportError as exc:  # pragma: no cover - deployment dependency
                raise ObjectStorageError("minio package is not installed") from exc
            client = Minio(
                endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure,
            )
        self.client = client
        self.bucket = bucket
        if not self.client.bucket_exists(bucket):
            self.client.make_bucket(bucket)

    def put_bytes(
        self, key: str, payload: bytes, content_type: str = "application/octet-stream"
    ) -> StoredObject:
        from io import BytesIO

        normalized = normalize_object_key(key)
        try:
            self.client.put_object(
                self.bucket,
                normalized,
                BytesIO(payload),
                len(payload),
                content_type=content_type,
            )
        except Exception as exc:  # pragma: no cover - exercised against MinIO
            raise ObjectStorageError(f"unable to write object {normalized!r}") from exc
        return StoredObject(
            key=normalized,
            size=len(payload),
            content_type=content_type,
            uri=self.uri_for(normalized),
        )

    def get_bytes(self, key: str) -> bytes:
        normalized = normalize_object_key(key)
        response: BinaryIO | None = None
        try:
            response = self.client.get_object(self.bucket, normalized)
            return response.read()
        except Exception as exc:  # pragma: no cover - exercised against MinIO
            raise ObjectStorageError(f"unable to read object {normalized!r}") from exc
        finally:
            if response is not None:
                response.close()
                release = getattr(response, "release_conn", None)
                if release:
                    release()

    def exists(self, key: str) -> bool:
        normalized = normalize_object_key(key)
        try:
            self.client.stat_object(self.bucket, normalized)
            return True
        except Exception:
            return False

    def uri_for(self, key: str) -> str:
        return f"minio://{self.bucket}/{normalize_object_key(key)}"

    def delete_prefix(self, prefix: str) -> None:
        normalized = normalize_object_key(prefix)
        try:
            items = list(
                self.client.list_objects(
                    self.bucket, prefix=normalized, recursive=True
                )
            )
            for item in items:
                self.client.remove_object(self.bucket, item.object_name)
        except Exception as exc:  # pragma: no cover - exercised against MinIO
            raise ObjectStorageError(f"unable to purge prefix {normalized!r}") from exc


def put_json(storage: ObjectStorage, key: str, payload: Any) -> StoredObject:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return storage.put_bytes(key, encoded, "application/json; charset=utf-8")


def get_json(storage: ObjectStorage, key: str) -> Any:
    return json.loads(storage.get_bytes(key).decode("utf-8"))


def content_type_for(filename: str, default: str = "application/octet-stream") -> str:
    return mimetypes.guess_type(filename)[0] or default


def image_is_visually_blank(payload: bytes, *, white_threshold: int = 248) -> bool:
    """Detect near-empty OCR image resources without rejecting sparse diagrams."""

    try:
        from PIL import Image

        with Image.open(BytesIO(payload)) as source:
            rgba = source.convert("RGBA")
            flattened = Image.new("RGBA", rgba.size, "white")
            flattened.alpha_composite(rgba)
            grayscale = flattened.convert("L")
            width, height = grayscale.size
            inset_x = max(1, int(width * 0.03))
            inset_y = max(1, int(height * 0.03))
            if width > inset_x * 2 and height > inset_y * 2:
                grayscale = grayscale.crop(
                    (inset_x, inset_y, width - inset_x, height - inset_y)
                )
            grayscale.thumbnail((96, 96))
            histogram = grayscale.histogram()
    except Exception:
        return False
    pixel_count = sum(histogram)
    if not pixel_count:
        return True
    near_white = sum(histogram[white_threshold:]) / pixel_count
    return near_white >= 0.995


def crop_image_region(
    payload: bytes, bbox: tuple[float, float, float, float]
) -> bytes:
    """Crop a source page using pixel coordinates and return normalized PNG."""

    try:
        from PIL import Image

        with Image.open(BytesIO(payload)) as source:
            x0, y0, x1, y1 = bbox
            left = max(0, min(int(x0), source.width - 1))
            top = max(0, min(int(y0), source.height - 1))
            right = max(left + 1, min(int(x1), source.width))
            bottom = max(top + 1, min(int(y1), source.height))
            cropped = source.crop((left, top, right, bottom))
            output = BytesIO()
            cropped.save(output, format="PNG")
            return output.getvalue()
    except Exception as exc:
        raise ObjectStorageError("unable to crop source page region") from exc


def image_content_type(payload: bytes) -> str:
    try:
        from PIL import Image

        with Image.open(BytesIO(payload)) as source:
            return Image.MIME.get(source.format or "", "image/png")
    except Exception:
        return "application/octet-stream"


def storage_from_environment(settings: Any) -> ObjectStorage:
    """Build storage without ever exposing credential values to callers."""

    backend = os.getenv("APP_STORAGE_BACKEND", "local").strip().lower()
    if backend == "local":
        root = os.getenv("APP_LOCAL_STORAGE_ROOT")
        if not root:
            root = str(Path(__file__).resolve().parents[2] / "data" / "objects")
        return LocalObjectStorage(root)
    if backend != "minio":
        raise ObjectStorageError(f"unsupported storage backend: {backend}")

    access = getattr(settings, "minio_access_key", None)
    secret = getattr(settings, "minio_secret_key", None)
    if not access or not secret:
        raise ObjectStorageError("MinIO credentials are not configured")
    return MinioObjectStorage(
        settings.minio_endpoint,
        access.get_secret_value(),
        secret.get_secret_value(),
        settings.minio_bucket,
        secure=settings.minio_secure,
    )
