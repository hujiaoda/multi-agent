"""结构化输出：用 Qwen（dashscope 兼容接口）产出机器可读的判定字段"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from config import QWEN_API_KEY, QWEN_BASE_URL


class ClarifyDecision(BaseModel):
    """需求澄清节点的判定结果"""
    need_more_info: bool = Field(description="信息是否足够开始写计划")
    question: str = Field(description="信息不足时要问用户的问题；信息足够则为空字符串")


# 你之前在 output.py / output-adv.py 验证过的组合：qwen3.8-max + dashscope 兼容地址
qwen_model = ChatOpenAI(
    model="qwen3.8-max",
    temperature=0,
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
    max_tokens=10000,
    timeout=30,
    max_retries=3,
    extra_body={"enable_thinking": False},  # 关闭思考模式，保证结构化输出稳定
)

# with_structured_output 返回的不再是字符串，而是 ClarifyDecision 对象
clarify_model = qwen_model.with_structured_output(ClarifyDecision)
