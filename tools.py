"""工具集：web_search（LLM 决策用）/ 文件读写 / run_file 按语言运行（节点直接调用）"""
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
    """设置当前运行的工作目录：所有文件工具只能在这个目录内读写"""
    global _workspace
    _workspace = Path(path).resolve()  # 存成绝对路径，沙箱比较基准稳定


def _safe_resolve(path: str) -> Path:
    """路径沙箱：任何路径先算出最终落点，落点在工作区外就抛错。

    三步：
    1. 拼接：相对路径 → 补上工作区前缀；绝对路径 → 直接用
    2. 归一化：resolve() 消掉 ./ 和 ../，变成唯一的绝对路径形式
    3. 检查：最终落点必须在工作区（含所有子目录）内
    """
    if _workspace is None:
        raise RuntimeError("工作目录未设置，先调用 set_workspace()")
    raw = Path(path)
    # 兼容层：模型爱传 work/<时间戳>/xxx.py（相对项目根的完整路径）。
    # 不剥的话会拼成 工作区/work/<时间戳>/xxx.py，位置错但又不越界。
    if not raw.is_absolute():
        try:
            ws_rel = _workspace.relative_to(Path.cwd()).parts
        except ValueError:
            ws_rel = ()  # 工作区不在项目根下（如临时目录），跳过兼容层
        if len(raw.parts) >= len(ws_rel) and raw.parts[:len(ws_rel)] == ws_rel:
            raw = Path(*raw.parts[len(ws_rel):])
    # ① 拼接：相对路径以工作区为锚点
    target = (raw if raw.is_absolute() else _workspace / raw).resolve()
    # ② 检查：落点在工作区内 → 放行
    if target.is_relative_to(_workspace):
        return target
    # ③ 越界 → 抛错，由调用方转成给模型看的错误信息
    raise ValueError(
        f"路径越界被拦截: {path} → {target}；"
        f"只允许在工作目录 {_workspace} 内，请改用相对路径如 main.py"
    )


@tool
def web_search(query: str) -> str:
    """搜索网络，返回与 query 相关的网页摘要。"""
    result = _get_tavily().search(query=query, max_results=5)
    return str(result)


def list_files(dir_path: str) -> list[str]:
    """列出指定目录下的所有文件（相对工作目录或绝对路径）。"""
    try:
        p = _safe_resolve(dir_path)
    except ValueError:
        return []  # 越界目录 → 当空目录处理
    if not p.exists() or not p.is_dir():
        return []
    # 返回工作区相对路径：模型看到的文件名，就是它该传给 write_file 的写法
    return [str(f.relative_to(_workspace)) for f in p.rglob("*") if f.is_file()]

def read_file(filename: str) -> str:
    """读取指定路径的文件内容（相对工作目录或绝对路径，UTF-8 编码）。"""
    try:
        p = _safe_resolve(filename)
    except ValueError as e:
        return f"读取被拒绝：{e}"
    if not p.exists() or not p.is_file():
        return ""
    return p.read_text(encoding="utf-8")

def edit_file(filename: str, old_content: str, new_content: str) -> str:
    """编辑指定路径的文件内容（相对工作目录或绝对路径，UTF-8 编码）。"""
    try:
        p = _safe_resolve(filename)
    except ValueError as e:
        return f"编辑被拒绝：{e}"
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
    """把内容写入指定路径的文件（相对工作目录或绝对路径，UTF-8 编码）。"""
    try:
        p = _safe_resolve(filename)
    except ValueError as e:
        return f"写入被拒绝：{e}"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"已写入 {p}"


def _run_cmd(cmd: list[str], timeout: int = 30) -> dict:
    """跑一条命令，统一返回 returncode / stdout / stderr / timed_out。"""
    # 强制子进程用 UTF-8 输出，否则 Windows 上中文输出按 GBK 编码会导致解码崩溃
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
        return {"returncode": result.returncode, "stdout": result.stdout,
                "stderr": result.stderr, "timed_out": False}
    except FileNotFoundError:
        return {"returncode": -1, "stdout": "", "stderr": f"命令不存在: {cmd[0]}（未安装或不在 PATH）", "timed_out": False}
    except subprocess.TimeoutExpired as e:
        return {"returncode": -1, "stdout": e.stdout or "", "stderr": e.stderr or "", "timed_out": True}


def run_file(path: str, timeout: int = 30) -> dict:
    """按文件后缀选运行方式：.py/.js 直接跑，.cpp/.c 先编译再跑，其余不支持。"""
    try:
        p = _safe_resolve(path)
    except ValueError as e:
        return {"returncode": -1, "stdout": "", "stderr": f"运行被拒绝：{e}", "timed_out": False}
    suffix = p.suffix.lower()
    if suffix == ".py":
        return _run_cmd(["python", str(p)], timeout)
    if suffix == ".js":
        return _run_cmd(["node", str(p)], timeout)
    if suffix in (".cpp", ".cc", ".cxx"):
        exe = p.with_suffix(".exe")
        # 多文件支持：入口 + 同目录其他 C++ 源文件一起编译链接（避免"未定义的引用"）
        sources = [str(p)]
        for pat in ("*.cpp", "*.cc", "*.cxx"):
            sources.extend(str(f) for f in p.parent.glob(pat) if f != p)
        comp = _run_cmd(["g++", *sources, "-o", str(exe)], timeout)
        if comp["returncode"] != 0:
            return comp
        return _run_cmd([str(exe)], timeout)
    if suffix == ".c":
        exe = p.with_suffix(".exe")
        comp = _run_cmd(["gcc", str(p), "-o", str(exe)], timeout)
        if comp["returncode"] != 0:
            return comp
        return _run_cmd([str(exe)], timeout)
    return {"returncode": -1, "stdout": "", "stderr": f"不支持运行 {suffix} 文件（支持 .py / .js / .cpp / .c）", "timed_out": False}

FILE_TOOLS = [list_files, read_file, edit_file, write_file]
# 普通函数没有 .name，用 __name__；react.py 执行时兼容普通函数
FILE_TOOLS_MAP = {f.__name__: f for f in FILE_TOOLS}
model_with_tools_file = model.bind_tools(FILE_TOOLS)
# 带工具的模型：目前只有 web_search 留给 LLM 决策用；
# 写文件/跑代码是流程固定动作，由节点函数直接调用，不走 LLM 工具调用
TOOLS = [web_search]
TOOLS_MAP = {t.name: t for t in TOOLS}
model_with_tools = model.bind_tools(TOOLS)
