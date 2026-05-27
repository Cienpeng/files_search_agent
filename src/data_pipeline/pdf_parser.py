import os
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PDFParser:
    """
    PDF 文档解析与切片管道，专注于给 RAG 提供干净的 Chunk。
    """
    def __init__(self, chunk_size=500, chunk_overlap=50):
        # 使用递归分词器，按逻辑断句保留语义完整性
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )

    def parse_and_chunk(self, file_path: str):
        """
        读取本地 PDF 并切片，同时绑定元数据(Metadata)。
        Returns:
            list of dict: [{"text": chunk_text, "source": file_path, "page": page_num}]
        """
        if not os.path.exists(file_path):
            print(f"[-] 文件不存在: {file_path}")
            return []
            
        doc = fitz.open(file_path)
        chunks = []
        
        for page_num, page in enumerate(doc):
            # 获取文本并清理首尾多余空白
            page_text = page.get_text("text").strip()
            
            if not page_text:
                # TODO: 如果页面为空，说明可能是扫描件，预留接口后续对接 PaddleOCR 处理纯图PDF页
                continue
                
            # 执行切片
            page_chunks = self.text_splitter.split_text(page_text)
            for chunk_text in page_chunks:
                chunks.append({
                    "text": chunk_text,
                    "source": file_path,
                    "page": page_num + 1
                })
        
        return chunks

# if __name__ == "__main__":
#     parser = PDFParser()
#     print("✅ PDFParser 加载成功，等待测试文件录入...")
