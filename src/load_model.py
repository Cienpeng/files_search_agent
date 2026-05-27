# 模型统一初始化
import os
import logging
from pathlib import Path
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from dotenv import load_dotenv
from enum import Enum

# 获取项目根目录并加载.env
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# 定义一个Enum类来表示模型，如cahtgpt，Gemini，Claude等
class Model(Enum):
    CHATGPT = "chatgpt"
    GEMINI = "gemini"
    CLAUDE = "claude"
    QWEN = "qwen"
    DEEPSEEK = "deepseek"
    KIMI = "kimi"
    MIMO = "mimo"
    GLM = "glm"


class init_url_api:
    # 初始化模型的base_url和 api_key，如果模型名称包含Model枚举关键字，就提取出模型名称，否则直接使用模型名称
    def __init__(self, model_name):
        model_list = [model.value for model in Model]
        # 如果模型名称包含Model枚举关键字，比如deepseek v4 flash，则url为DEEPSEEK_BASE_URL,api为DEEPSEEK_API_KEY
        matched_model = None
        for model in model_list:
            if model.lower() in model_name.lower():
                matched_model = model.upper()
                break
        if matched_model is None:
            # 如果模型名称不包含Model枚举关键字，比如custom_model，则url为CUSTOM_MODEL_BASE_URL,api为CUSTOM_MODEL_API_KEY
            matched_model = model_name.upper()
            logging.warning(
                f"Model name '{model_name}' does not match any known models. Using '{matched_model}' as the prefix for environment variables."
            )
        self.base_url = os.getenv(f"{matched_model}_BASE_URL")
        self.api_key = os.getenv(f"{matched_model}_API_KEY")

    def get_base_url_api_key(self):
        return [self.base_url, self.api_key]


class InitModel:
    """
    初始化模型类，包含模型名称，模型提供商，base_url，api_key，温度等参数
    args:
        model_name: 模型名称，比如chatgpt，Gemini，Claude等
        model_provider: 模型提供商，比如openai，google，anthropic等
        base_url: 模型的base_url，如果为None，则从环境变量中获取
        api_key: 模型的api_key，如果为None，则从环境变量中获取
        temperature: 模型的温度参数，默认为1.0
    """

    def __init__(
        self,
        model_name: str,
        model_provider: str,
        base_url: str = None,
        api_key: str = None,
        temperature: float = 1.0,
    ):
        self.model_name = model_name
        self.model_provider = model_provider
        
        url_api = init_url_api(model_name)
        extracted_base_url, extracted_api_key = url_api.get_base_url_api_key()
        
        self.base_url = extracted_base_url if base_url is None else base_url
        self.api_key = extracted_api_key if api_key is None else api_key
        self.temperature = temperature

    def create_chat_model(self):
        try:
            chat_model = init_chat_model(
                model=self.model_name,
                model_provider=self.model_provider,
                base_url=self.base_url,
                api_key=self.api_key,
                temperature=self.temperature,
            )
            return chat_model
        except Exception as e:
            logging.error(f"Error creating chat model: {e}")
            return None


class InitAgent:
    """
    初始化agent类
    """

    def __init__(self):
        self.system_prompt = """
The user will provide some exam text. Please parse the "question" and "answer" and output them in JSON format. 

EXAMPLE INPUT: 
Which is the highest mountain in the world? Mount Everest.

EXAMPLE JSON OUTPUT:
{
    "question": "Which is the highest mountain in the world?",
    "answer": "Mount Everest"
} 
"""

    def create_agent(self, chat_model):
        try:
            agent = create_agent(model=chat_model, system_prompt=self.system_prompt)
            return agent
        except Exception as e:
            logging.error(f"Error creating agent: {e}")
            return None
