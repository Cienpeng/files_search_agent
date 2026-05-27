import sys
import os
from langchain_core.tools import tool

# 动态引入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vector_store.milvus_mgr import MilvusManager
from vector_store.embedding_models import EmbeddingManager

# 全局初始化，避免每次调用工具都重新加载模型和数据库连接
try:
    milvus_mgr = MilvusManager()
    embed_mgr = EmbeddingManager()
except Exception as e:
    print(f"Warning: Semantic search tool initialization failed: {e}")

@tool
def search_local_knowledge(query: str) -> str:
    """
    当需要回答关于某个专业知识、寻找某个文件内容、或用户让你检索本地资料库时，请调用此工具。
    它会在本地 Milvus 向量库中进行余弦相似度检索并返回最相关的 PDF 或图片OCR片段。
    
    Args:
        query: 用户的查询问题或提取的搜索关键词。
    """
    try:
        # 1. 将查询转为向量
        query_vector = embed_mgr.get_text_embedding(query)
        
        # 2. 从 Milvus 检索 (提取前 3 个最相似的结果)
        results = milvus_mgr.search_text(query_vector, limit=3)
        
        if not results or len(results[0]) == 0:
            return "未能从本地知识库中检索到与此相关的文档片段。"
        
        # 3. 格式化结果发给大模型
        formatted_results = []
        for hit in results[0]:
            # entity 包含了插入时我们在 schema 中赋予的其他字段
            source = hit.get("entity", {}).get("source_path", "未知来源")
            text = hit.get("entity", {}).get("text", "")
            score = hit.get("distance", 0.0) # 这里的 distance 通常是余弦相似度或L2
            
            formatted_results.append(f"【来源路径】: {source} (相关度得分: {score:.3f})\n【内容】: {text}\n")
        
        return f"针对 '{query}'，检索到以下本地文档内容:\n\n" + "-"*40 + "\n" + "\n".join(formatted_results)
    except Exception as e:
        return f"语义检索工具发生异常: {str(e)}"