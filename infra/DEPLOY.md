# 服务器部署指南（Docker 一步到位）

整套系统全部容器化，服务器上只需一条命令即可启动：

```
postgres / minio / neo4j / api(后端) / ocr-bridge(OCR 桥) / web(nginx 前端 + 反向代理)
```

浏览器只访问 **一个端口**：`http://服务器IP:4321`。nginx 同时托管前端静态文件，
并把 `/api/v1/*` 反向代理给后端（同源，无需配置 CORS）。后端 `18800` 端口保留用于
调试 / Swagger / 健康检查。

## 一、服务器要求

- Linux 服务器 + **Docker** 与 **Docker Compose**（v2）
- 能访问外网（拉取镜像 + 调用 DeepSeek / 智谱 / PaddleOCR 官方 API）
- 防火墙放行 `4321`（前端）；如要远程看 API 文档再放行 `18800`

## 二、部署步骤

```bash
# 1. 获取代码（或直接推送你的仓库后 clone）
git clone <你的仓库地址> PDF-Studio && cd PDF-Studio

# 2. 生成并填写 infra/.env（数据服务密码 + DeepSeek/智谱密钥）
cd infra
cp .env.example .env
vi .env          # 必填：POSTGRES_PASSWORD / MINIO_ROOT_USER / MINIO_ROOT_PASSWORD /
                 #      NEO4J_PASSWORD / MOONSHOT_API_KEY(DeepSeek) / OPENAI_API_KEY(智谱)

# 3. 写 OCR token
vi ocr.env       # PADDLE_OCR_TOKEN=<飞桨 PaddleOCR-VL 网页获取的 access token>

# 4. 一步启动（首次会构建 api / ocr-bridge / web 三个镜像，需几分钟）
docker compose --env-file .env up -d --build

# 5. 验证
curl http://localhost:4321/api/v1/health      # 经 nginx 代理的后端健康检查
```

浏览器打开 `http://服务器IP:4321` 即可使用。

## 三、日常运维

```bash
docker compose --env-file .env ps                          # 查看状态
docker compose --env-file .env logs -f api                 # 后端日志
docker compose --env-file .env logs -f ocr-bridge          # OCR 桥日志
docker compose --env-file .env up -d                       # 快速重启（不重建）
docker compose --env-file .env up -d --build               # 拉了新代码后重建
docker compose --env-file .env down                        # 停止（数据卷保留）
```

> 改了代码重新部署：`git pull` → `docker compose --env-file .env up -d --build`。
> 如果构建用的是旧缓存，先 `docker compose build --no-cache api`。

## 四、密钥与安全

- `infra/.env`、`infra/ocr.env`、`.local-secrets.yaml` 均已被 `.gitignore` 排除，
  **不要提交到仓库**；`Dockerfile` 与 `.dockerignore` 也确保密钥不会被打进镜像。
- 密钥只通过 compose 注入容器环境变量，浏览器 / 前端永不接触。
- 数据卷（`postgres-data` / `minio-data` / `neo4j-data`）在 `down` 后保留；确认要清空
  才用 `docker compose down --volumes`。

## 五、常见问题

| 现象 | 原因 / 解决 |
| --- | --- |
| 编译任务卡 queued | `infra/.env` 缺 `APP_COMPILE_MODE=sync`，补上后重启 api |
| 前端能开但接口 404 | nginx 只代理 `/api/v1`；确认 `web` 与 `api` 在同一 compose 网络 |
| 上传大 PDF 失败 | nginx 已设 `client_max_body_size 100m`；超过 100MB 需改 `infra/nginx.conf` |
| 残留旧容器占端口 | `docker compose --env-file .env up -d --remove-orphans` |
| OCR 桥报错 | `docker compose logs ocr-bridge`；确认 `infra/ocr.env` 的 token 有效 |
| 服务器 4321 打不开 | 检查云防火墙 / 安全组是否放行该端口 |

## 六、本地开发对照

- **本仓库本地调试**：Windows 用 `start_docker.sh`（等价于 `docker compose up -d --build`）；
  WSL 开发路径仍是 `operations/local/start_local.sh`（前端/OCR 跑在宿主，数据服务走 Docker）。
- 两者的区别只是前端与 OCR 桥是否容器化；`infra/.env` 的密钥配置两处通用。
