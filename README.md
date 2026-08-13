# PDF-Studio 多模态 PDF 检索引擎

一个端到端的多模态 PDF 知识库系统：上传 PDF → OCR 版面解析 → 向量索引 → 知识图谱 → LLM Wiki → Agentic RAG 问答，所有数据全链路实时同步。

**前端** React 19 + Vite + AntV G6（知识图谱） + 服务端页面图双栏对照查看器
**后端** FastAPI + SQLAlchemy 状态快照（PostgreSQL）+ LangGraph 检查点 + MinIO + Neo4j
**模型** DeepSeek V4 Flash（LLM/问答/图谱抽取）+ 智谱 embedding-3（向量）+ PaddleOCR-VL-1.6（OCR 版面解析）

---

## 一、环境要求

- **Docker Desktop**（含 Docker Compose）——唯一需要安装的运行时

> 前端、后端、OCR 桥、数据服务已全部容器化，宿主机不再需要 Node.js / Python 环境。
> WSL2 开发模式（可选）仍需要 Python 3.12 + Node 22，见「三、WSL2 模式」。

## 二、快速开始（Docker 模式，推荐）

在项目根目录（Git Bash / PowerShell）执行：

```bash
bash start_docker.sh
```

脚本会**构建并启动全部容器**（postgres / minio / neo4j / api / ocr-bridge / web），首次构建需几分钟：

- **web**（nginx）：托管前端静态文件 + 反向代理 `/api/v1` 到后端，浏览器只需访问一个端口
- **ocr-bridge**：把后端逐页图片转调飞桨官方 PaddleOCR-VL API

### 访问地址

| 服务 | 地址 |
| --- | --- |
| Web 前端 | http://localhost:4321 |
| API 文档 (Swagger) | http://localhost:18800/docs |
| API 健康检查 | http://localhost:18800/api/v1/health |

## 三、WSL2 模式（备选）

项目也可在 WSL2 的 Linux 环境运行（需要先在 WSL 内安装 Python 3.12、Node 22，并创建 `backend/.venv`）。`start_local.sh` 会自动检测 Windows 环境并切换到 WSL 执行：

```bash
bash operations/local/start_local.sh
```

WSL 模式用 `stack.py` 管理进程（api/frontend 作为宿主机进程，数据服务走 Docker Compose）。

## 四、服务器部署（Docker 一步到位）

整套系统已全部容器化（数据服务 + 后端 + OCR 桥 + nginx 前端反向代理），服务器上一条命令即可启动，浏览器只访问一个端口 `http://服务器IP:4321`：

```bash
docker compose --env-file infra/.env up -d --build
```

完整步骤（环境准备、密钥填写、日常运维、常见问题）见 **`infra/DEPLOY.md`**。

## 五、配置文件

| 文件 | 作用 | 是否提交 |
| --- | --- | --- |
| `.local-secrets.yaml` | Provider 密钥：`moonshot` 槽位 = DeepSeek，`openai` 槽位 = 智谱 embedding | ❌ 权限 600 |
| `infra/.env` | Docker 数据服务密码 + Provider 变量（含 `APP_COMPILE_MODE=sync`） | ❌ 权限 600 |
| `infra/ocr.env` | 飞桨 PaddleOCR-VL 官方 API 的 access token | ❌ 权限 600 |
| `.env.example` / `.local-secrets.example.yaml` | 配置模板 | ✅ |

### Provider 配置说明

后端通过统一的 OpenAI 兼容适配器（`app/services/providers.py`）连接模型，槽位映射：

```yaml
# .local-secrets.yaml
api_keys:
  moonshot:          # 聊天 / 问答 / 图谱抽取（LLM）
    key: "<你的 DeepSeek API Key>"
    base_url: "https://api.deepseek.com/v1"
  openai:            # 向量 Embedding（OpenAI 兼容）
    key: "<你的智谱 API Key>"
    base_url: "https://open.bigmodel.cn/api/paas/v4"
```

模型名由 `infra/.env` 注入：`MOONSHOT_CHAT_MODEL=deepseek-v4-flash`、`MOONSHOT_STRUCTURED_MODEL=deepseek-v4-flash`、`OPENAI_EMBEDDING_MODEL=embedding-3`。

> **想换硅基流动 embedding**（`Qwen/Qwen3-Embedding-0.6B`）：给硅基流动账号充值后，把 `openai` 槽位换成硅基流动 key + `https://api.siliconflow.cn/v1`，并改 `OPENAI_EMBEDDING_MODEL` 即可。

### 模型兼容性要点

- DeepSeek 官方 API **拒绝 `response_format: json_schema`**（400），后端统一改用 `json_object`，并在系统提示词中写入 JSON Schema 约束 + jsonschema 自修复重试（`MAX_SCHEMA_REPAIR_ATTEMPTS=4`），因此任何 OpenAI 兼容模型都能稳定产出结构化输出。
- DeepSeek 是纯文本模型：文档富化（图片/表格描述）会退化为「待增强」，纯文本 PDF 的问答与检索不受影响。

## 六、核心功能

- **知识库**：创建 / 上传 PDF / 编译 / 文档与知识库的**级联删除**
- **多模态编译**：PaddleOCR-VL 版面解析 → 元素归一化 → 富化 → 分块 → 向量索引（BM25 + dense + RRF 混合检索）
- **知识图谱**（Neo4j）：实体 / 主张 / 关系抽取，证据回溯到 PDF 页面坐标
- **LLM Wiki**：由知识图谱持续编译的可追溯技术百科
- **Agentic RAG 问答**：规划器判断是否需要检索 → 工具调用 → 带引用的 grounded 回答（拒绝无来源答案）

## 七、全链路实时更新机制

删除文档 / 知识库会级联清理，确保图谱、Wiki、检索索引、编译产物始终一致：

- **删文档**：移除该文档的检索 chunk、Neo4j 图谱贡献、MinIO 编译产物，并从更新后的图谱重建 Wiki（孤儿页自动消失）
- **删知识库**：删除全部文档、索引、产物、任务，清空该库的图谱与 Wiki
- **对账修复**：若历史数据因旧版删除残留了脏数据，执行对账即可清理：

```bash
curl -X POST http://localhost:18800/api/v1/knowledge-bases/<库ID>/reconcile
```

- 前端删除后立即 `refresh()` 全链路刷新；轮询先重列知识库（被删则自动切换或清空），图谱用内容签名只在数据真正变化时重建。

### 快速重建与实时进度

- **快速重建**：`build_graph_and_wiki` 只对新增 / 未抽取图谱的文档做 LLM 抽取，已有图谱贡献的文档直接复用 → 全库重建从十几分钟降到**几十秒**。
- **实时进度**：重建任务的图谱抽取阶段按**批次**上报进度（82% → 100%），任务中心实时显示；单个任务有 25 分钟超时看门狗（`APP_JOB_TIMEOUT_SECONDS`），挂起任务自动失败并放行后续任务，不会堵死并发队列。

## 八、常见问题排查

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| 编译 / 重建任务卡在 queued 不执行 | Docker 模式缺 `APP_COMPILE_MODE=sync` | 确认 `infra/.env` 有 `APP_COMPILE_MODE=sync` 并 `docker compose up -d api` |
| 重建任务在 graph_wiki 阶段较久 | **首次**抽取大文档图谱（DeepSeek 批量调用） | 属正常，任务中心进度条会实时跳；**后续**重建跳过已有贡献（几十秒完成） |
| 图谱 / Wiki 显示已删内容 | 历史脏数据（旧版删除未级联） | 调用 `/reconcile` 对账清理 |
| OCR 桥接启动失败 | token 无效 / 容器无法访问外网 | `docker compose logs ocr-bridge`；确认 `infra/ocr.env` 的 token 有效 |
| 前端报「模型服务暂不可用」 | DeepSeek 限额 / 模型名错误 | 检查 `infra/.env` 的 `MOONSHOT_*` 配置 |
| 切换启动模式 | Docker 与 WSL 模式不要同时开（抢 18800/4321 端口） | 停掉一边再启动另一边 |

## 九、项目结构

```
├── src/                    # React 前端
│   ├── pages/              # 问答、知识库、图谱、Wiki、任务、设置页
│   ├── components/         # PDF 查看器、侧栏、产品头部等
│   └── hooks/              # useWorkspaceApi（轮询 + 全链路状态）
├── backend/
│   ├── app/
│   │   ├── api/routes/     # REST/SSE 接口
│   │   ├── services/       # 编译、检索、图谱、Wiki、富化、Provider 适配
│   │   └── store.py        # 内存状态 + SQLAlchemy 快照持久化 + 级联删除
│   └── tests/              # pytest 契约/回归测试
├── infra/                  # Docker Compose、Dockerfile、nginx.conf、DEPLOY.md
├── operations/local/       # stack.py、OCR 桥接（WSL 开发模式）、启动脚本
└── start_docker.sh         # Docker 一键启动（Windows / 服务器通用）
```

## 十、构建与测试

```bash
npm run build                     # 前端构建（宿主机）
cd infra && docker compose --env-file .env build   # 后端/OCR/前端镜像构建
cd backend && .venv/Scripts/python.exe -m pytest   # 后端回归测试（文件权限用例需在 Linux 下运行）
```
