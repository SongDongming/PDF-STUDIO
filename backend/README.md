# 多模态 PDF 知识库后端

Python 3.12 + FastAPI 后端，面向 PDF 编译入库、多轮问答、LLM Wiki、
知识图谱和模型设置。运行态使用 SQLAlchemy 状态快照（PostgreSQL）持久化业务
状态、进程内任务执行器（LocalJobRunner）、对象存储保存 PDF 与编译资产、
Neo4j 保存图谱，并通过 DeepSeek（LLM/问答/抽取）、智谱 GLM embedding-3
（向量）和飞桨 PaddleOCR-VL 官方 API（OCR 桥接）完成真实链路。内存适配器
只用于隔离测试。

## 目录

- `app/api/routes/`：按业务域拆分的 REST/SSE API；
- `app/services/`：编译、检索、图谱、Wiki、富化、Provider 适配；
- `app/store.py`：内存状态 + SQLAlchemy 快照持久化 + 级联删除编排；
- `tests/`：不依赖外部基础设施的合同测试。

## 本地开发

要求 Python 3.12。不要把真实 API Key 写入 `.env.example`、数据库或前端。

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/pytest
.venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 18800
```

- OpenAPI：<http://localhost:18800/openapi.json>
- Swagger：<http://localhost:18800/docs>
- 健康检查：<http://localhost:18800/api/v1/health>

完整基础设施启动方式见 `../infra/README.md`。

## API 合同

| 业务域 | 主入口 | 当前行为 |
| --- | --- | --- |
| 健康 | `GET /api/v1/health` | 返回服务版本和适配器状态 |
| 知识库 | `/api/v1/knowledge-bases` | CRUD、整库编译任务、级联删除 |
| 文档 | `/api/v1/knowledge-bases/{id}/documents` | 元数据登记、PDF 上传、编译 |
| 任务 | `/api/v1/jobs` | 列表、详情、执行、取消、重试、删除 |
| 会话 | `/api/v1/chat/threads` | 新建、切换、归档、消息和 SSE |
| Wiki | `/api/v1/knowledge-bases/{id}/wiki/pages` | 页面列表与详情 |
| 图谱 | `/api/v1/knowledge-bases/{id}/graph` | 子图和证据入口 |
| 设置 | `/api/v1/settings` | Provider、RAG、编译设置和连接检查 |

问答响应是有序 Block：`text`、`image`、`table`、`formula`。图片、表格、
公式只引用后端已存在的 `asset_id`；Citation 合同保留文档、页码、
归一化 bbox 和元素 ID，供 PDF 双栏查看器定位。

## Agentic RAG 运行时

运行态由 `APP_AGENT_RUNTIME` 决定（Docker/WSL 默认 `legacy`）：

- **legacy**：DeepSeek 规划器 + 混合检索工具，严格 JSON 结构化输出，带
  引用校验与自修复重试。
- **deepagents**：Deep Agents 图运行时，根据问题自行决定是否调用检索工具；
  需要 LangGraph 依赖与模型凭证。

通用问题可以零工具调用直接回答。最终结构化回答仍由应用层校验 Citation 与
asset_id，模型不能自行定义来源。PostgreSQL 部署同时承载 LangGraph checkpoint
和跨线程 Store；非 PostgreSQL 开发环境明确降级为进程内状态。

文档编译使用双层视觉合同：PaddleOCR-VL/PP-DocLayoutV3 是页码、阅读顺序、
元素 ID 与 BBox 的权威来源；LLM 对所有多模态元素裁图和最多 6 个高密度富媒体
整页补充检索语义。语义覆盖层使用独立 fingerprint 进入索引，不改写 Paddle 原始
产物。

## 凭证边界

设置页可以提交新的 API Key，但该字段以 `SecretStr` 进入后端后只写入
`600` 权限的受保护服务端文件，并立即从公开响应中消失。读取接口只返回是否
已配置，浏览器、OpenAPI 响应、日志和业务持久化均不保存或回显凭证原文。
CORS 只允许显式配置的本机与局域网来源。

## 持久化

- **业务状态**：`MemoryStore` 的内存集合通过 `SqlAlchemyStatePersistence`
  落库为版本化快照表，启动时恢复（Postgres 不可用时降级为纯内存并记录日志）。
- **编译产物 / 索引 / 素材**：通过不可变 manifest 进入对象存储。
- **图谱**：Neo4j（或内存实现），证据合同同步。
- **任务**：进程内 `LocalJobRunner`，单任务有超时看门狗；任务中心实时反映进度。
