"""通用的"LLM + 工具"循环（React 循环）：
模型可以反复调用工具，直到它直接给出文字回答。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.messages import ToolMessage


def call_model(model, messages, tools_map, max_turns=5):
    """调用带工具的模型，自动处理工具调用循环。

    args：
        model      带工具的模型（model.bind_tools(...) 之后的那个）
        messages   消息列表（会被原地追加 AIMessage / ToolMessage）
        tools_map  {"工具名": 工具对象}，用于执行模型请求的工具
        max_turns  工具调用轮次上限（硬切断，防止模型无限调工具烧 token）

    returns：
        最终的 AIMessage（content 里是文字回答，一定非空）
    """
    for _ in range(max_turns):
        response = model.invoke(messages)
        messages.append(response)

        # 模型没有想调工具 → 它直接回答了，这就是我们要的最终结果
        if not response.tool_calls:
            return response

        # 模型想调工具 → 逐个执行，把结果以 ToolMessage 喂回去
        for tc in response.tool_calls:
            fn = tools_map[tc["name"]]
            result = fn.invoke(tc["args"])
            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tc["id"],
            ))

    # 达到 max_turns 还没结束：返回最后一条消息，硬切断
    return messages[-1]
