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
_workspace = None  # 当前运行的工作目录（main.py 启动时 set_workspace 设置）


def _get_tavily():
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilyClient(api_key=Tapi)
    return _tavily_client


def set_workspace(path: str):
    """设置当前运行的工作目录：相对路径的工具调用都会落到这里"""
    global _workspace
    _workspace = Path(path)


def _resolve(path: str) -> Path:
    """路径解析：绝对路径直接用；相对路径拼到工作目录下"""
    p = Path(path)
    if p.is_absolute():
        return p
    # 模型可能传 work/<时间戳>/xxx.py 这种"完整相对路径"：直接相对项目根目录用
    if _workspace and p.parts and Path(_workspace).parent.name == p.parts[0]:
        return p
    return Path(_workspace) / p if _workspace else p


@tool
def web_search(query: str) -> str:
    """搜索网络，返回与 query 相关的网页摘要。"""
    result = _get_tavily().search(query=query, max_results=5)
    return str(result)


def list_files(dir_path: str) -> list[str]:
    """列出指定目录下的所有文件（相对项目根目录或绝对路径）。"""
    p = _resolve(dir_path)
    if not p.exists() or not p.is_dir():
        return []
    return [str(f) for f in p.rglob("*") if f.is_file()]

def read_file(filename: str) -> str:
    """读取指定路径的文件内容（相对项目根目录或绝对路径，UTF-8 编码）。"""
    p = _resolve(filename)
    if not p.exists() or not p.is_file():
        return ""
    return p.read_text(encoding="utf-8")

def edit_file(filename: str, old_content: str, new_content: str) -> str:
    """编辑指定路径的文件内容（相对项目根目录或绝对路径，UTF-8 编码）。"""
    p = _resolve(filename)
    if old_content == new_content:
        return f"文件 {p} 内容未改变"
    if not p.exists() or not p.is_file():
        return f"文件 {p} 不存在"
    content = p.read_text(encoding="utf-8")
    if old_content not in content:
        return f"文件 {p} 中没找到要替换的内容"
    p.write_text(content.replace(old_content, new_content, 1), encoding="utf-8")
    return f"已编辑 {p}"

def write_file(filename: str, content: str) -> str:
    """把内容写入指定路径的文件（相对项目根目录或绝对路径，UTF-8 编码）。"""
    p = _resolve(filename)
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

FILE_TOOLS = [list_files, read_file, edit_file, write_file]
# 普通函数没有 .name，用 __name__；react.py 执行时兼容普通函数
FILE_TOOLS_MAP = {f.__name__: f for f in FILE_TOOLS}
model_with_tools_file = model.bind_tools(FILE_TOOLS)
# 带工具的模型：目前只有 web_search 留给 LLM 决策用；
# 写文件/跑代码是流程固定动作，由节点函数直接调用，不走 LLM 工具调用
TOOLS = [web_search]
TOOLS_MAP = {t.name: t for t in TOOLS}
model_with_tools = model.bind_tools(TOOLS)
