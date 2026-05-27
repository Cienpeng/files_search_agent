import os
import sys
from langchain_core.tools import tool

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.build_index import build_index_for_image, build_index_for_pdf


@tool
def index_local_file_to_knowledge_base(file_path: str) -> str:
    """
    当用户已经找到某个本地 PDF 或图片文件，但知识库/向量库中没有该文件内容时，调用此工具将文件即时加入本地向量数据库。
    PDF 会先提取文本；扫描件页面会尝试 OCR。图片会提取 OCR 文本并建立 CLIP 图片特征。

    Args:
        file_path: 需要入库的本地文件绝对路径，如 D:/xxx.pdf 或 D:/xxx.png。
    """
    if not os.path.exists(file_path):
        return f"系统错误: 在路径 {file_path} 未找到该文件，无法入库。"

    lower_path = file_path.lower()
    try:
        if lower_path.endswith(".pdf"):
            result = build_index_for_pdf(file_path)
            insert_count = result.get("insert_count", 0) if isinstance(result, dict) else 0
            if insert_count:
                return f"已将 PDF 加入本地知识库，共写入 {insert_count} 个文本块: {file_path}"
            return f"PDF 已处理，但未提取到可入库文本: {file_path}"

        if lower_path.endswith((".png", ".jpg", ".jpeg", ".bmp")):
            build_index_for_image(file_path)
            return f"已将图片加入本地知识库/图片库: {file_path}"

        return "系统错误: 当前即时入库只支持 PDF 和主流图片格式。"
    except Exception as e:
        return f"文件即时入库失败: {str(e)}"
