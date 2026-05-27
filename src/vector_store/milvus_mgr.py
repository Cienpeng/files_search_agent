import os
from pymilvus import MilvusClient

# 获取项目根目录来保存本地数据库文件
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
RESOURCES_DIR = os.path.join(PROJECT_ROOT, "resources")
os.makedirs(RESOURCES_DIR, exist_ok=True)
DB_PATH = os.path.join(RESOURCES_DIR, "milvus_lite.db")

class MilvusManager:
    """
    Milvus Lite 管理器，用于存储和检索文本及图像的 Embedding 向量。
    """
    def __init__(self):
        # 初始化 MilvusClient (直接保存在本地，无需部署 Docker)
        self.client = MilvusClient(DB_PATH)
        self.text_collection = "Text_Collection"
        self.image_collection = "Image_CLIP_Collection"
        self._init_collections()

    def _init_collections(self):
        """初始化双 Collection：一个存文本(以及OCR)，一个存跨模态图像"""
        
        # BGE-M3 维度通常是 1024
        if not self.client.has_collection(self.text_collection):
            self.client.create_collection(
                collection_name=self.text_collection,
                dimension=1024,
                auto_id=True # 自动生成主键ID
            )
        self.client.load_collection(self.text_collection)
        
        # Chinese-CLIP 图像特征向量通常是 512 维
        if not self.client.has_collection(self.image_collection):
            self.client.create_collection(
                collection_name=self.image_collection,
                dimension=512,
                auto_id=True
            )
        self.client.load_collection(self.image_collection)

    def insert_text_vectors(self, data_list: list[dict]):
        """
        插入文本向量。
        data_list 格式要求: [{"vector": [0.1, 0.2...], "source_path": "d:/xxx.pdf", "text": "内容片段..."}]
        """
        res = self.client.insert(
            collection_name=self.text_collection,
            data=data_list
        )
        return res

    def search_text(self, query_vector: list[float], limit: int = 3):
        """基于问题向量，在文档库中执行相似度最近邻检索"""
        res = self.client.search(
            collection_name=self.text_collection,
            data=[query_vector],
            limit=limit,
            output_fields=["source_path", "text"]
        )
        return res

    def insert_image_vectors(self, data_list: list[dict]):
        """
        插入图像特征向量。
        data_list 格式要求: [{"vector": [0.1, 0.2...], "source_path": "d:/xxx.jpg"}]
        """
        res = self.client.insert(
            collection_name=self.image_collection,
            data=data_list
        )
        return res

    def search_image(self, query_vector: list[float], limit: int = 3):
        """基于问题向量，在图片库中执行跨模态检索"""
        res = self.client.search(
            collection_name=self.image_collection,
            data=[query_vector],
            limit=limit,
            output_fields=["source_path"]
        )
        return res

    def _path_filter_candidates(self, file_path: str):
        def escape_filter_value(value: str) -> str:
            return value.replace("\\", "\\\\").replace('"', '\\"')

        candidates = []
        normalized_path = file_path.replace("\\", "/")
        for candidate in (normalized_path, file_path):
            if candidate not in candidates:
                candidates.append(candidate)
        return [f'source_path == "{escape_filter_value(candidate)}"' for candidate in candidates]

    def delete_text_by_path(self, file_path: str):
        """删除指定文件路径对应的文本/OCR索引，避免重复入库。"""
        for filter_expr in self._path_filter_candidates(file_path):
            self.client.delete(
                collection_name=self.text_collection,
                filter=filter_expr
            )

    def delete_image_by_path(self, file_path: str):
        """删除指定文件路径对应的图片特征索引，避免重复入库。"""
        for filter_expr in self._path_filter_candidates(file_path):
            self.client.delete(
                collection_name=self.image_collection,
                filter=filter_expr
            )

    def query_text_by_path(self, file_path: str):
        """基于文件的绝对路径，直接从知识库中查询它的所有文本块"""
        for filter_expr in self._path_filter_candidates(file_path):
            res = self.client.query(
                collection_name=self.text_collection,
                filter=filter_expr,
                output_fields=["text"]
            )
            if res:
                return res
        return []

