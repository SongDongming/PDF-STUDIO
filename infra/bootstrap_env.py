#!/usr/bin/env python3
"""Create a mode-600 local Compose environment without printing secrets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / ".env",
    )
    parser.add_argument(
        "--cors-origin",
        action="append",
        default=["http://localhost:4321"],
    )
    parser.add_argument(
        "--ocr-base-url",
        default="http://host.docker.internal:18111",
    )
    parser.add_argument("--update-ocr-base-url", action="store_true")
    parser.add_argument("--update-cors-origins", action="store_true")
    return parser.parse_args()


def random_secret(length: int = 32) -> str:
    return secrets.token_urlsafe(length)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        if args.update_ocr_base_url or args.update_cors_origins:
            lines = args.output.read_text(encoding="utf-8").splitlines()
            replacements: dict[str, str] = {}
            if args.update_ocr_base_url:
                replacements["APP_OCR_BASE_URL"] = args.ocr_base_url
            if args.update_cors_origins:
                replacements["APP_CORS_ORIGINS"] = json.dumps(
                    list(dict.fromkeys(args.cors_origin)), ensure_ascii=False
                )
            replaced: set[str] = set()
            updated: list[str] = []
            for line in lines:
                key, separator, _value = line.partition("=")
                if separator and key in replacements:
                    updated.append(f"{key}={replacements[key]}")
                    replaced.add(key)
                else:
                    updated.append(line)
            for key, value in replacements.items():
                if key not in replaced:
                    updated.append(f"{key}={value}")
            temporary = args.output.with_suffix(".tmp")
            descriptor = os.open(
                temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
            )
            try:
                os.write(descriptor, ("\n".join(updated) + "\n").encode("utf-8"))
            finally:
                os.close(descriptor)
            temporary.replace(args.output)
            os.chmod(args.output, 0o600)
            print(
                f"updated {len(replacements)} runtime setting(s) in {args.output} "
                f"mode={oct(args.output.stat().st_mode & 0o777)}"
            )
            return
        print(f"kept existing {args.output} mode={oct(args.output.stat().st_mode & 0o777)}")
        return

    values = {
        "POSTGRES_DB": "pdfwiki",
        "POSTGRES_USER": "pdfwiki",
        "POSTGRES_PASSWORD": random_secret(),
        "MINIO_ROOT_USER": f"pdfwiki-{secrets.token_hex(4)}",
        "MINIO_ROOT_PASSWORD": random_secret(),
        "NEO4J_PASSWORD": random_secret(),
        "APP_CORS_ORIGINS": json.dumps(args.cors_origin, ensure_ascii=False),
        "APP_OCR_BASE_URL": args.ocr_base_url,
    }
    content = "".join(f"{key}={value}\n" for key, value in values.items())
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, content.encode("utf-8"))
    finally:
        os.close(descriptor)
    print(f"created {args.output} mode=0o600 keys={len(values)}")


if __name__ == "__main__":
    main()
