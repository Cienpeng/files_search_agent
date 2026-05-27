from sentence_transformers import SentenceTransformer
from pathlib import Path
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
import torch
import os
import dotenv
from PIL import Image

# 获取项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent.parent
dotenv.load_dotenv(BASE_DIR / ".env")
HF_token = os.getenv("HF_TOKEN")

class EmbeddingManager:
    """
    统一的词向量（Embedding）管理类。
    用于将文本问题、PDF片段转为多维浮点数组（向量）。
    推荐使用 BAAI/bge-m3（支持多语言，性能优异），或者可以换成轻量级的 shibing624/text2vec-base-chinese
    """
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B"):
        # 获取项目根目录，确保路径绝对，不受运行命令所在目录的影响
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        # 定义本地模型存储路径
        local_model_path = BASE_DIR / "models" / "embedding_models" / model_name.replace("/", "_")
    
        # 检查本地是否有模型
        if local_model_path.exists() and any(local_model_path.iterdir()):
            print(f"[*] 从本地加载 Embedding 模型: {local_model_path}")
            self.model = SentenceTransformer(str(local_model_path))
        else:
            print(f"[*] 本地未找到模型，正在从 HuggingFace 下载: {model_name}")
            print(f"[*] 模型将保存到: {local_model_path}")
            # 下载模型并保存到本地
            self.model = SentenceTransformer(model_name, cache_folder=str(local_model_path.parent), token=HF_token)
            # 保存模型到指定路径
            self.model.save(str(local_model_path))
            print(f"[✓] 模型已下载并保存到本地: {local_model_path}")

    def get_text_embedding(self, text: str) -> list[float]:
        """将单个文本转化为向量"""
        return self.model.encode(text).tolist()
        
    def get_text_embeddings(self, text_list: list[str]) -> list[list[float]]:
        """批量将文本列表转化为向量列表"""
        embeddings = self.model.encode(text_list)
        return [emb.tolist() for emb in embeddings]



class CLIPManager:
    """
    用于提取图像（和文本）的特征向量，专门支持跨模态。
    使用官方 cn_clip 库。默认加载 ViT-B-16 模型 (512维)
    """
    def __init__(self, model_name="ViT-B-16"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] 正在加载 Chinese CLIP 模型: {model_name} on {self.device}")
        
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        clip_model_path = BASE_DIR / "models" / "clip_models"
        
        # 加载模型和图像预处理方法
        self.model, self.preprocess = load_from_name(model_name, device=self.device, download_root=str(clip_model_path))
        self.model.eval()
        
    def get_text_embedding(self, text: str) -> list[float]:
        """将检索词转换为向量"""
        text_input = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()[0].tolist()
        
    def get_image_embedding(self, image_path: str) -> list[float]:
        """将图片转换为向量"""
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()[0].tolist()
