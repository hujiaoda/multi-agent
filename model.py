"""模型构建：所有 LLM 实例统一从这里创建"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain.chat_models import init_chat_model
from config import DEEPSEEK_API_KEY as apikey, BASE_URL as burl


def build_llm():
    """创建 DeepSeek 模型（裸模型，不绑定任何工具）"""
    return init_chat_model(
        model="deepseek-v4-flash",
        model_provider="deepseek",
        temperature=0.7,
        api_key=apikey,
        base_url=burl,
        max_tokens=10000,
        timeout=60,
        max_retries=3,
    )


# 模块级裸模型：plan / code / review / test 生成文本
model = build_llm()
