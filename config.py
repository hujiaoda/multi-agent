"""项目配置：从 .env 读取敏感信息"""
import os
from dotenv import load_dotenv

load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL="https://api.deepseek.com"
QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_API_KEY=os.getenv("QWEN_API_KEY")

TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")

