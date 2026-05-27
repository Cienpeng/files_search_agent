# 一个本地检索文件的agent
```
src/
 ├── agents/           # 存放 LangGraph/LangChain 的组装逻辑、Prompt 定义
 ├── tools/            # 存放具体的 @tool 函数 (如文件查找、文档读取)
 ├── data_pipeline/    # 存放文档解析、OCR、CLIP 特征提取、Chunking 逻辑
 ├── vector_store/     # 封装 Milvus 的连接、插入、混合检索
 ├── files_index.py    # 底层 NTFS 检索 (已存在)
 ├── load_model.py     # 模型统一初始化 (已存在)
 ├── memory_DB.py      # SQLite 记忆持久化 (开发中)
 └── main.py           # 命令行/UI 的交互入口
 ```