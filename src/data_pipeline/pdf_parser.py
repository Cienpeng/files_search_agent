import os
import fitz  # PyMuPDF
import tempfile
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

    def parse_and_chunk(self, file_path: str, enable_ocr: bool = True):
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
        img_parser = None
        temp_dir = None
        
        try:
            for page_num, page in enumerate(doc):
                # 获取文本并清理首尾多余空白
                page_text = page.get_text("text").strip()

                if not page_text and enable_ocr:
                    if img_parser is None:
                        from data_pipeline.img_parser import ImageParser

                        img_parser = ImageParser()
                    if temp_dir is None:
                        temp_dir = tempfile.TemporaryDirectory()

                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    page_img_path = os.path.join(temp_dir.name, f"page_{page_num + 1}.png")
                    pix.save(page_img_path)
                    page_text = img_parser.extract_text(page_img_path).strip()

                if not page_text:
                    continue

                # 执行切片
                page_chunks = self.text_splitter.split_text(page_text)
                for chunk_text in page_chunks:
                    chunks.append({
                        "text": chunk_text,
                        "source": file_path,
                        "page": page_num + 1
                    })
        finally:
            doc.close()
            if temp_dir is not None:
                temp_dir.cleanup()
        
        return chunks

# if __name__ == "__main__":
#     parser = PDFParser()
#     print("✅ PDFParser 加载成功，等待测试文件录入...")
