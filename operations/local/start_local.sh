#!/usr/bin/env bash
# One-click local startup for the DeepSeek + PaddleOCR web-service workbench.
#
#  1. When invoked from Windows (Git Bash / PowerShell + bash), re-runs itself
#     inside WSL2 at ~/PDF-Studio, where the Linux venv and stack live.
#  2. In WSL: starts the local OCR bridge (reads PADDLE_OCR_TOKEN from
#     infra/ocr.env) if it is not already healthy on 127.0.0.1:18111, then runs
#     `stack.py start` with the provider credentials.
#
# Run from the repository root (Windows or WSL):
#   bash operations/local/start_local.sh
set -euo pipefail

# --- Windows detection: re-invoke inside WSL ---
if [ "$(uname -s)" != "Linux" ]; then
  echo "检测到 Windows 环境，自动切换到 WSL 的 ~/PDF-Studio 启动..."
  wsl bash -lc 'cd "$HOME/PDF-Studio" && bash operations/local/start_local.sh'
  exit $?
fi

cd "$(dirname "$0")/../.."

if ! curl -sf http://127.0.0.1:18111/health >/dev/null 2>&1; then
  echo "OCR bridge not running; starting it on 127.0.0.1:18111 ..."
  OCR_BRIDGE_BACKEND=paddle nohup backend/.venv/bin/python \
    operations/local/ocr_bridge.py >/tmp/ocr_bridge.log 2>&1 &
  sleep 3
fi

backend/.venv/bin/python operations/local/stack.py start \
  --credentials ./.local-secrets.yaml
