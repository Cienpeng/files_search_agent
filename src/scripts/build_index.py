import sys
import os

# 将 src 添加进 path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_pipeline.pdf_parser import PDFParser
from data_pipeline.img_parser import ImageParser
from vector_store.embedding_models import EmbeddingManager, CLIPManager
from vector_store.milvus_mgr import MilvusManager

def build_index_for_pdf(pdf_path: str):
    """
    模拟一段流水线：提取PDF -> 切片 -> 生成向量 -> 灌入数据库
    此接口可以与 files_index.py 结合，扫描全盘后做批量入库。
    """
    print(f"\n[1] 开始解析 PDF: {pdf_path}")

    parser = PDFParser()
    chunks = parser.parse_and_chunk(pdf_path, enable_ocr=True)
    if not chunks:
        print("[-] PDF 未提取到可入库文本。")
        return {"insert_count": 0, "message": "PDF 未提取到可入库文本"}

    print(f"[2] 提取到 {len(chunks)} 个文本块，正在生成 Embedding...")
    embed_mgr = EmbeddingManager()
    vectors = embed_mgr.get_text_embeddings([chunk["text"] for chunk in chunks])

    data = []
    for chunk, vector in zip(chunks, vectors):
        data.append({
            "vector": vector,
            "source_path": pdf_path,
            "text": f"[第 {chunk['page']} 页]\n{chunk['text']}"
        })

    milvus_mgr = MilvusManager()
    milvus_mgr.delete_text_by_path(pdf_path)
    res = milvus_mgr.insert_text_vectors(data)
    print(f"[OK] PDF 文本入库成功: {res}")
    return {"insert_count": len(data), "result": res}

def build_index_for_image(img_path: str):
    """
    图片入库流水线：
    1. PaddleOCR 提取可能存在的文字 -> 发给文本表
    2. CLIP 提取整图特征 -> 发给图片表
    """
    print(f"\n[1] 开始解析图片: {img_path}")
    
    # 1. OCR 解析 (文本入库)
    img_parser = ImageParser()
    ocr_text = img_parser.extract_text(img_path)
    
    milvus_mgr = MilvusManager()
    milvus_mgr.delete_text_by_path(img_path)
    milvus_mgr.delete_image_by_path(img_path)
    
    if ocr_text:
        print("[2-A] 发现文字，正在嵌入 OCR 文本...")
        embed_mgr = EmbeddingManager()
        text_vector = embed_mgr.get_text_embedding(ocr_text)
        milvus_mgr.insert_text_vectors([{
            "vector": text_vector,
            "source_path": img_path,
            "text": ocr_text
        }])
        print("[OK] OCR 文本入库成功。")
    
    # 2. CLIP 解析 (跨模态入库)
    print("[2-B] 正在提取图像视觉特征 (CLIP)...")
    clip_mgr = CLIPManager()
    img_vector = clip_mgr.get_image_embedding(img_path)
    
    res = milvus_mgr.insert_image_vectors([{
        "vector": img_vector,
        "source_path": img_path
    }])
    print(f"[OK] CLIP 图像特征入库成功: {res}")

if __name__ == "__main__":
    # 可以把这里替换为真实某个 PDF 绝对路径来测试
    TEST_PDF = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../test_files/test.pdf"))
    TEST_IMG = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../test_files/test.png"))
    
    # 真正调用构建索引的函数
    build_index_for_pdf(TEST_PDF)
    build_index_for_image(TEST_IMG)
