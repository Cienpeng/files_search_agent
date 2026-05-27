import sys
import os
from langchain_core.tools import tool

# 动态引入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vector_store.milvus_mgr import MilvusManager
from vector_store.embedding_models import CLIPManager

# 全局初始化，避免每次调用工具时重新加载巨大的 CLIP 模型
try:
    milvus_mgr = MilvusManager()
    clip_mgr = CLIPManager()
except Exception as e:
    print(f"Warning: Image search tool initialization failed: {e}")

@tool
def search_local_image_by_content(query: str) -> str:
    """
    当用户要求寻找“长什么样子”的图片、根据画面内容找图、或者搜索视觉元素（如一只狗、一辆车、某人在干嘛的截图）时，调用此工具。
    **注意：只用于搜图！如果用户是想搜文字/文档资料，请使用 search_local_knowledge。**
    
    Args:
        query: 描述图片画面的自然语言，例如 "一只红色的苹果" 或 "包含柱状图的报表"。
    """
    try:
        # 1. 使用 CN-CLIP 的文本编码器计算检索词的特征向量 (512维度)
        query_vector = clip_mgr.get_text_embedding(query)
        
        # 2. 在 Milvus 图片特征集合中尝试匹配 (Top 3)
        res = milvus_mgr.search_image(vector=query_vector, limit=3)
        
        if not res or not res[0]:
            return "未在本地图片库中搜索到匹配该视觉特征或场景的图片。"
        
        # 3. 组织返回结果给大模型
        result_texts = ["以下是基于视觉特征为您检索到的相关图片路径：\n"]
        for idx, hit in enumerate(res[0]):
            score = hit.get("distance", hit.get("score", 0.0))
            source_path = hit["entity"].get("source_path", "Unknown Path")
            result_texts.append(f"[{idx+1}] 相似度得分：{score:.4f} \n图片路径: {source_path}")
            
        return "\n".join(result_texts)
        
    except Exception as e:
        return f"检索图片库时发生错误：{str(e)}"
