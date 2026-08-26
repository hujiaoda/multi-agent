"""工具集：web_search（LLM 决策用）/ write_file / run_python_file（节点直接调用）"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import subprocess
from pathlib import Path

from langchain_core.tools import tool
from tavily import TavilyClient

from config import TAVILY_API_KEY as Tapi
from model import model

_tavily_client = None


def _get_tavily():
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilyClient(api_key=Tapi)
    return _tavily_client


@tool
def web_search(query: str) -> str:
    """搜索网络，返回与 query 相关的网页摘要。"""
    result = _get_tavily().search(query=query, max_results=5)
    return str(result)


def write_file(filename: str, content: str) -> str:
    """把内容写入指定路径的文件（相对项目根目录或绝对路径，UTF-8 编码）。"""
    p = Path(filename)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"已写入 {p}"


def run_python_file(path: str) -> dict:
    """用 subprocess 真实执行 Python 文件，返回 returncode / stdout / stderr。"""
    # 强制子进程用 UTF-8 输出，否则 Windows 上中文输出按 GBK 编码会导致解码崩溃
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    result = subprocess.run(
        ["python", path],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        env=env,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


# 带工具的模型：目前只有 web_search 留给 LLM 决策用；
# 写文件/跑代码是流程固定动作，由节点函数直接调用，不走 LLM 工具调用
TOOLS = [web_search]
TOOLS_MAP = {t.name: t for t in TOOLS}
model_with_tools = model.bind_tools(TOOLS)
