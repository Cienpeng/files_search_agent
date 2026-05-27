import os

# 修改 PaddlePaddle/PaddleOCR 默认下载路径缓存环境变量，防止写满 C 盘
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
PADDLE_CACHE_DIR = os.path.join(PROJECT_ROOT, "models", "paddle_models")
os.makedirs(PADDLE_CACHE_DIR, exist_ok=True)
os.environ["PADDLE_PDX_CACHE_DIR"] = PADDLE_CACHE_DIR
# 有些版本的 paddle 使用 HOME 作为基础目录：
os.environ["HOME"] = PADDLE_CACHE_DIR

# 修复新版 PaddlePaddle (3.0+) 带来的内核 OneDNN/PIR 未实现错误报错
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

try:
    # 需要安装 paddlepaddle 和 paddleocr
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False


class ImageParser:
    """
    图片解析管道：主要负责提取图片中的文字信息（OCR）
    跨模态 CLIP 搜索可以作为后续独立的向量插入逻辑。
    """
    def __init__(self):
        if PADDLE_AVAILABLE:
            # 开启方向分类(use_angle_cls)，识别中文与英文
            # 去除 show_log 参数以适配不同版本的 PaddleOCR
            self.ocr = PaddleOCR(use_angle_cls=True, lang="ch")
            print("[*] PaddleOCR 引擎已就绪。")
        else:
            self.ocr = None
            print("[-] 未安装 PaddleOCR，图片文字识别功能不可用。请运行 uv pip install paddlepaddle paddleocr")

    def extract_text(self, img_path: str) -> str:
        """从图片提取纯文本结果"""
        if not self.ocr:
            return ""
        if not os.path.exists(img_path):
            print(f"[-] 图片不存在: {img_path}")
            return ""
            
        # OC R识别
        # 新版 paddleocr 的 ocr 方法/predict 方法去掉了 cls=True 参数
        result = self.ocr.ocr(img_path)
        if not result or not result[0]:
            return ""
            
        # 结果是一个列表，包含box坐标，文本和置信度，我们只要文本
        text_lines = [line[1][0] for line in result[0]]
        return "\n".join(text_lines)

if __name__ == "__main__":
    img_parser = ImageParser()
