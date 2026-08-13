# 基础设施

此 Compose 管理本产品自己的 PostgreSQL、MinIO、Neo4j 与 FastAPI 后端
（`api` 服务），不复用 Dspark 或讲研所其他项目的服务。任务在 API 进程内由
LocalJobRunner 执行，不再需要独立的 Celery Worker 或 Redis。

## 启动

```bash
cp .env.example .env
# 在本地 .env 中替换所有占位符；不要提交该文件
docker compose --env-file .env config
docker compose --env-file .env up --build
```

数据服务只绑定开发机的 `127.0.0.1`。只有 API 暴露 `18800`，OCR 服务通过
`APP_OCR_BASE_URL` 由后端代理（Windows 宿主上的 `ocr_bridge.py`），浏览器
不直接访问模型端口或凭证。

健康检查：

```bash
curl http://localhost:18800/api/v1/health
```

停止服务不会删除卷：

```bash
docker compose --env-file .env down
```

除非明确确认数据可丢弃，不要使用 `down --volumes`。

## 服务

| 服务 | 用途 | 端口（本机） |
| --- | --- | --- |
| `postgres` | 业务状态快照 + LangGraph 检查点 | 15432 |
| `minio` | PDF 源文件与编译资产对象存储 | 19000 / 19001 |
| `neo4j` | 知识图谱 | 17474 / 17687 |
| `api` | FastAPI 后端 | 18800 |
