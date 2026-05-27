import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

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
        # 如果是图片，优先读取已入库 OCR；没有命中时现场调用 PaddleOCR。
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            indexed_text = ""
            try:
                mgr = MilvusManager()
                results = mgr.query_text_by_path(file_path)
                indexed_lines = []
                for result in results:
                    text_value = result.get("text", "")
                    if text_value and text_value not in indexed_lines:
                        indexed_lines.append(text_value)
                indexed_text = "\n".join(indexed_lines)
            except Exception as e:
                indexed_error = str(e)
            else:
                indexed_error = ""

            if indexed_text:
                text = indexed_text
                source = "图片向量库记录"
            else:
                try:
                    from data_pipeline.img_parser import ImageParser

                    parser = ImageParser()
                    text = parser.extract_text(file_path)
                    source = "实时 OCR 识别"
                except Exception as e:
                    detail = f"；查询图片索引也失败: {indexed_error}" if indexed_error else ""
                    return f"图片 OCR 识别失败: {str(e)}{detail}"

            if not text:
                detail = f"；查询图片索引失败: {indexed_error}" if indexed_error else ""
                return f"未能从图片 {file_path} 识别到文字{detail}。"

            if len(text) > max_chars:
                text = text[:max_chars] + "\n\n...[由于长度限制，后续内容已截断]..."
            return f"来自{source}的图片文字如下:\n{text}"
                
        if not file_path.lower().endswith('.pdf'):
            return "系统错误: 当前工具实体读取仅支持 .pdf 格式文件以及主流图片。"
            
        doc = fitz.open(file_path)
        text = ""
        try:
            for page_num, page in enumerate(doc):
                page_text = page.get_text("text").strip()
                if page_text:
                    text += f"\n[第 {page_num + 1} 页]\n{page_text}\n"
                # 防止超出大模型上下文窗口
                if len(text) > max_chars:
                    text = text[:max_chars] + "\n\n...[由于长度限制，后续内容已截断]..."
                    break
        finally:
            doc.close()

        if not text.strip():
            from data_pipeline.pdf_parser import PDFParser

            parser = PDFParser(chunk_size=max_chars, chunk_overlap=0)
            chunks = parser.parse_and_chunk(file_path, enable_ocr=True)
            text = "\n\n".join([chunk["text"] for chunk in chunks])
            if not text.strip():
                return f"未能从 PDF {file_path} 提取到文字；它可能是空文件、加密文件，或 OCR 未识别成功。"
            if len(text) > max_chars:
                text = text[:max_chars] + "\n\n...[由于长度限制，后续内容已截断]..."
        
        return f"来自 {file_path} 的文档内容如下:\n{text}"
    except Exception as e:
        return f"读取文档时发生异常: {str(e)}"
