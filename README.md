# Flies Search Agent

Flies Search Agent 是一个面向本地文件检索的多工具 Agent。它使用 LangGraph / LangChain 编排大模型和本地工具，支持在 Windows 本地环境中搜索文件、读取 PDF、识别图片或扫描件文字，并把文档内容写入 Milvus Lite 向量库用于后续语义检索。


## 主要功能

- 本地文件名搜索：通过 NTFS 搜索 DLL 快速查找磁盘文件。
- PDF 内容读取：使用 PyMuPDF 提取 PDF 文本。
- 扫描件 OCR：当 PDF 页面没有可提取文本时，自动渲染为图片并调用 PaddleOCR 识别文字。
- 图片文字识别：对 PNG/JPG/JPEG/BMP 图片执行 OCR，或读取已入库的 OCR 结果。
- 语义检索：使用本地 Embedding 模型生成向量，并在 Milvus Lite 中检索 PDF / 图片 OCR 内容。
- 图片内容检索：使用 Chinese-CLIP 对图片建立视觉特征索引，支持按画面内容找图。
- 即时入库：当找到的 PDF 或图片尚未在知识库中时，可以现场解析并写入向量数据库。
- 会话记忆：使用 SQLite 保存 LangGraph checkpoint，并维护历史会话标签。

## 运行环境

项目使用 `uv` 管理 Python 环境，要求 Python 3.12 或更高版本。

因为底层文件搜索使用 NTFS 搜索能力，运行涉及全盘文件名搜索时需要以管理员权限启动终端。

启动入口：

```powershell
uv run python src/main.py
```

如果需要查看 Agent 工具调用过程，可以打开调试日志：

```powershell
$env:AGENT_DEBUG="1"
uv run python src/main.py
```

默认情况下，工具调用过程日志和 PaddleOCR DEBUG 日志会被静默，终端只显示最终回答和必要提示。


## 项目结构

```text
lib
|
.env
|
src/
├── main.py
├── load_model.py
├── files_index.py
├── memory_DB.py
├── agents/
│   └── agent_graph.py
├── tools/
│   ├── tool_file_search.py
│   ├── tool_semantic_search.py
│   ├── tool_image_search.py
│   ├── tool_read_full_document.py
│   └── tool_index_file.py
├── data_pipeline/
│   ├── pdf_parser.py
│   └── img_parser.py
├── vector_store/
│   ├── embedding_models.py
│   └── milvus_mgr.py
└── scripts/
    └── build_index.py

```

## 核心模块说明

### `src/main.py`

命令行入口，调用 `agents.agent_graph.chat_run()` 启动交互式 Agent。

### `src/agents/agent_graph.py`

负责构建 LangGraph 工作流，包括：

- 定义系统提示词。
- 初始化模型。
- 绑定本地工具。
- 配置工具节点和模型节点之间的循环。
- 处理会话恢复、会话删除和终端交互。

### `src/load_model.py`

封装模型初始化逻辑，从 `.env` 中读取不同模型供应商的 `BASE_URL` 和 `API_KEY`，并通过 LangChain 初始化聊天模型。

### `src/files_index.py`

封装 `lib/NTFS-Search.dll`，通过 `ctypes` 调用底层 NTFS 搜索能力。该模块用于快速按文件名搜索 D 盘文件。

### `src/memory_DB.py`

封装 SQLite 会话记忆，包括 LangGraph checkpoint 存储和会话标签管理。数据库文件位于：

```text
resources/memory.db
```

### `src/tools/`

存放暴露给 Agent 的 LangChain tools：

- `tool_file_search.py`：按文件名搜索 D 盘文件。
- `tool_semantic_search.py`：检索 Milvus Lite 中已入库的 PDF / 图片 OCR 文本。
- `tool_image_search.py`：使用 Chinese-CLIP 根据视觉内容搜索图片。
- `tool_read_full_document.py`：读取 PDF、图片 OCR 或已入库文本。
- `tool_index_file.py`：将找到但未入库的 PDF / 图片即时写入知识库。

### `src/data_pipeline/`

负责文件解析：

- `pdf_parser.py`：提取 PDF 文本；扫描件页面会转为图片后 OCR。
- `img_parser.py`：初始化 PaddleOCR 并从图片中提取文字。

### `src/vector_store/`

负责向量模型和 Milvus Lite：

- `embedding_models.py`：加载文本 Embedding 模型和 Chinese-CLIP 模型。
- `milvus_mgr.py`：创建、插入、查询和删除 Milvus Lite collection 数据。

### `src/scripts/build_index.py`

提供离线或即时入库流程：

- `build_index_for_pdf()`：解析 PDF，生成文本向量，写入文本 collection。
- `build_index_for_image()`：执行图片 OCR 和 CLIP 特征提取，分别写入文本 collection 和图片 collection。

## 数据和模型目录

```text
lib/
└── NTFS-Search.dll

models/
├── embedding_models/
├── clip_models/
└── paddle_models/

resources/
├── memory.db
└── milvus_lite.db/
```

- `lib/` 存放 NTFS 搜索 DLL。
- `models/` 存放本地 Embedding、CLIP 和 OCR 相关模型缓存。
- `resources/` 存放 SQLite 记忆库和 Milvus Lite 向量数据库。

## 注意事项
- NTFS 文件名搜索需要管理员权限。
- PaddleOCR 首次运行可能会下载或加载模型，耗时较长。
- lmdb和cn-clip包存在冲突，手动安装lmdb包即可
- 对复杂表格 PDF，OCR 文本可能会丢失原始二维表格结构，回答课程表类问题时可能需要模型根据 OCR 顺序推断列关系。
