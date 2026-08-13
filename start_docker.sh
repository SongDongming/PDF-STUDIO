#!/usr/bin/env bash
# One-step Docker startup (Windows Git Bash / Linux server).
# Everything runs in containers: postgres / minio / neo4j / api / ocr-bridge / web.
# The frontend is served by nginx on :4321 and proxies /api/v1 to the backend.
set -e
cd "$(dirname "$0")"

echo "=== 1/2 构建并启动全部容器（postgres / minio / neo4j / api / ocr-bridge / web）==="
(cd infra && docker compose --env-file .env up -d --build)

echo "=== 2/2 就绪 ==="
echo "  前端:        http://localhost:4321"
echo "  API 文档:    http://localhost:18800/docs"
echo "  健康检查:    http://localhost:18800/api/v1/health"
echo
echo "提示：重复启动可省略 --build 加速（docker compose --env-file infra/.env up -d）。"
echo "服务器部署请见 infra/DEPLOY.md。"
