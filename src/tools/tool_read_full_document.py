import os
import fitz  # PyMuPDF
import sys
from langchain_core.tools import tool

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vector_store.milvus_mgr import MilvusManager

@tool
def read_local_file_content(file_path: str, max_chars: int = 8000) -> str:
    """
    智能读取并返回本地 PDF 文档的内容，或提取本地图片（.png/.jpg等）上已建立索引的 OCR 文字内容。
    【重要指令】：作为AI模型，即使你自身没有视觉能力，也【绝对可以】通过调用此工具来“读取”图片上的内容！当用户询问“某张图片上有什么文字/内容”时，请务必从上下文中找出该图片的绝对路径，并传入此工具，工具会代劳并返回文字给你，千万不要直接回答说你无法看图。
    
    Args:
        file_path: 必须是本地文件的绝对路径（如 D:/xxx.pdf 或 D:/xxx.png）。
        max_chars: 截断以防止上下文溢出的最大字符数，默认8000。
    """
    if not os.path.exists(file_path):
        return f"系统错误: 在路径 {file_path} 未找到该文件，请提醒用户确认路径。"
    
    try:
        # 如果是图片，利用 MilvusManager 去向量库提取已索引的 OCR 文字
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            try:
                mgr = MilvusManager()
                results = mgr.query_text_by_path(file_path)
                if not results:
                    return f"该图片 {file_path} 尚未被 OCR 处理并存入向量库，无法读取上面的文字。"
                
                text = "\n".join([r.get("text", "") for r in results])
                if len(text) > max_chars:
                    text = text[:max_chars] + "\n\n...[由于长度限制，后续内容已截断]..."
                return f"来自图片向量库记录的内容如下:\n{text}"
            except Exception as e:
                return f"查询图片索引文字失败: {str(e)}"
                
        if not file_path.lower().endswith('.pdf'):
            return "系统错误: 当前工具实体读取仅支持 .pdf 格式文件以及主流图片。"
            
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
            # 防止超出大模型上下文窗口
            if len(text) > max_chars:
                text = text[:max_chars] + "\n\n...[由于长度限制，后续内容已截断]..."
                break
        
        return f"来自 {file_path} 的文档内容如下:\n{text}"
    except Exception as e:
        return f"读取文档时发生异常: {str(e)}"
