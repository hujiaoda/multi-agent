"""多智能体代码开发小队：plan → code → review → test，失败强制回 code 重写"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage

from model import model
from tools import write_file, run_python_file, read_file
from state import agentstate
from prompt import planprompt, codeprompt, reviewprompt, testprompt, clarifyprompt
from react import call_model
from tools import TOOLS_MAP, model_with_tools, model_with_tools_file, FILE_TOOLS_MAP
from structured import clarify_model
MAX_ATTEMPTS = 5
MAX_CLARIFY = 3


def strip_code_fences(text: str) -> str:
    """去掉模型可能多加的 ```python ... ``` 代码块外壳（安全网）"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def agentclarify(state: agentstate):
    # 结构化输出：Qwen 返回 ClarifyDecision 对象，bool 和问题都是类型保证的
    dec = clarify_model.invoke([
        SystemMessage(clarifyprompt),
        *state["messages"],
    ])
    return {
        "need_more_info": dec.need_more_info,
        "question": dec.question,
        "clarify_count": state.get("clarify_count", 0) + 1,
    }


def should_clarify(state: agentstate):  # clarify 节点的出口
    if state.get("clarify_count", 0) >= MAX_CLARIFY:
        return "plan"     # 问太多轮了，按现有信息继续（独立上限，不占重写次数）
    if state.get("need_more_info", False):
        return END        # 信息不足 → 结束本轮，main.py 收到问题去问用户
    return "plan"


def agentplan(state: agentstate):
    # 裸模型：plan 只需要生成文字，不能让它看到工具（否则可能返回 tool_calls 导致空计划）
    msgs=[
            SystemMessage(planprompt),
            HumanMessage(f"用户需求：{state['requirement']}"),
            ]
    response = call_model(model=model_with_tools,messages=msgs,tools_map=TOOLS_MAP)
    return {"plan": response.content}


def agentcoder(state: agentstate):
    msgs = [
        SystemMessage(codeprompt),
        HumanMessage(
            f"计划：{state['plan']}\n需求：{state['requirement']}"
            f"\n工作目录：{state['workdir']}"
            f"\n现有文件：{state.get('files') or '（空）'}"
        ),
    ]
    # 不管是审查还是测试打回来的，把两个字段都给它看，让 LLM 自己分辨
    if state.get("review"):
        msgs.append(HumanMessage(f"审查意见：{state['review']}"))
    if state.get("test_result"):
        msgs.append(HumanMessage(f"测试结果：{state['test_result']}"))

    # 模型通过文件工具自己读写磁盘；最后的文字回答只作总结
    # max_turns 放宽：复杂项目可能要写多个文件，5 轮不够
    call_model(model=model_with_tools_file, messages=msgs, tools_map=FILE_TOOLS_MAP, max_turns=8)

    # 跑完确定性记录：工作目录里现在有哪些文件
    files = sorted(f.name for f in Path(state["workdir"]).rglob("*") if f.is_file())
    return {"files": files, "entry_file": "main.py"}


def agentreview(state: agentstate):
    work_dir = Path(state["workdir"])
    files = state.get("files") or []
    parts = []
    for f in files:
        parts.append(f"===== {f} =====\n{read_file(str(work_dir / f))}")
    files_text = "\n\n".join(parts) if parts else "（工作目录为空）"

    response = model.invoke([
        SystemMessage(reviewprompt),
        HumanMessage(
            f"需求：{state['requirement']}\n计划：{state['plan']}\n工作目录文件：\n{files_text}"
        ),
    ])
    return {
        "review": response.content,
        "review_passed": "不通过" not in response.content,
        "attempt_count": state.get("attempt_count", 0) + 1,
    }


def agenttest(state: agentstate):
    # 本次运行的工作目录：work/<时间戳>/（main.py 生成后放进 State 的 workdir 字段）
    work_dir = Path(state["workdir"])
    entry_file = state.get("entry_file") or "main.py"

    # ① 入口文件必须存在（编码节点通过工具写盘）
    if not (work_dir / entry_file).exists():
        return {
            "test_result": f"入口文件不存在: {entry_file}",
            "tests_passed": False,
            "attempt_count": state.get("attempt_count", 0) + 1,
        }

    # ② LLM 生成测试代码（import 入口模块）并落盘
    test_code = strip_code_fences(model.invoke([
        SystemMessage(testprompt),
        HumanMessage(f"需求：{state['requirement']}\n入口文件：{entry_file}\n代码见工作目录：{work_dir}"),
    ]).content)

    write_file(str(work_dir / "test_target.py"), test_code)
    # 判定测试文件本身是否合格
    test_code = test_code.strip()
    has_assert = "assert" in test_code
    has_fallback = "无法断言测试" in test_code or "无法测试" in test_code
    if not test_code:
        tests_passed = False
        test_result = "测试文件为空,判定失败"
    elif not has_assert and not has_fallback:
        tests_passed = False
        test_result = "测试文件没有断言,判定失败"
    else:
        # 执行测试文件
        result = run_python_file(str(work_dir / "test_target.py"))
        if has_fallback:
            # 降级冒烟：只验证能跑起来（例如游戏类程序）
            tests_passed = result["returncode"] == 0
        else:
            # 真测试：必须跑完且输出完成标记（断言失败会在打印前崩溃）
            tests_passed = (result["returncode"] == 0) and ("ALL TESTS PASSED" in result["stdout"])
        test_result = result["stdout"] + result["stderr"]
    return {
        "test_result": test_result,
        "tests_passed": tests_passed,
        "attempt_count": state.get("attempt_count", 0) + 1,
    }

def should_continue(state: agentstate):  # review 节点的出口
    if state.get("attempt_count", 0) >= MAX_ATTEMPTS:
        return END            # 重写太多次了，硬切断
    if state.get("review_passed", False):
        return "test"         # 审查通过 → 去测试
    return "code"             # 审查不通过 → 回编码重写


def should_end(state: agentstate):       # test 节点的出口
    if state.get("attempt_count", 0) >= MAX_ATTEMPTS:
        return END            # 硬切断
    if state.get("tests_passed", False):
        return END            # 测试通过 → 结束
    return "code"             # 测试失败 → 回编码重写


graph = StateGraph(agentstate)

graph.add_node("clarify", agentclarify)
graph.add_node("plan", agentplan)
graph.add_node("code", agentcoder)
graph.add_node("review", agentreview)
graph.add_node("test", agenttest)

graph.set_entry_point("clarify")
graph.add_conditional_edges("clarify", should_clarify, {
    "plan": "plan",
    END: END,
})
graph.add_edge("plan", "code")
graph.add_edge("code", "review")
graph.add_conditional_edges("review", should_continue, {
    "code": "code",
    "test": "test",
    END: END,
})
graph.add_conditional_edges("test", should_end, {
    "code": "code",
    END: END,
})

app = graph.compile()
