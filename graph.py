"""多智能体代码开发小队：plan → code → review → test，失败强制回 code 重写"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage

from model import model
from tools import write_file, run_python_file
from state import agentstate
from prompt import planprompt, codeprompt, reviewprompt, testprompt
from react import call_model
from tools import TOOLS_MAP, model_with_tools
MAX_ATTEMPTS = 5


def agentplan(state: agentstate):
    msgs = [
            SystemMessage(planprompt),
            HumanMessage(f"用户需求：{state['requirement']}"),
            ]
    response = call_model(model=model_with_tools, messages=msgs, tools_map=TOOLS_MAP)
    return {"plan": response.content}


def agentcoder(state: agentstate):
    msgs = [
        SystemMessage(codeprompt),
        HumanMessage(f"计划：{state['plan']}\n需求：{state['requirement']}"),
    ]
    # 不管是审查还是测试打回来的，把两个字段都给它看，让 LLM 自己分辨
    if state.get("review"):
        msgs.append(HumanMessage(f"审查意见：{state['review']}"))
    if state.get("test_result"):
        msgs.append(HumanMessage(f"测试结果：{state['test_result']}"))
    response = call_model(model=model_with_tools, messages=msgs, tools_map=TOOLS_MAP)
    return {"code": response.content}


def agentreview(state: agentstate):
    response = model.invoke([
        SystemMessage(reviewprompt),
        HumanMessage(
            f"需求：{state['requirement']}\n计划：{state['plan']}\n代码：{state['code']}"
        ),
    ])
    return {
        "review": response.content,
        "review_passed": "不通过" not in response.content,
        "attempt_count": state.get("attempt_count", 0) + 1,
    }


def agenttest(state: agentstate):
    # 本次运行的工作目录：work/<时间戳>/（State 里的 workdir 字段）
    work_dir = Path(state["workdir"])

    # ① 被测代码落盘（subprocess 只能跑磁盘文件，不能跑 State 里的字符串）
    write_file(str(work_dir / "target.py"), state["code"])

    # ② LLM 生成测试代码（含断言）并落盘
    test_code = model.invoke([
        SystemMessage(testprompt),
        HumanMessage(f"需求：{state['requirement']}\n代码：{state['code']}"),
    ]).content
    write_file(str(work_dir / "test_target.py"), test_code)

    # ③ subprocess 真跑，机器执行断言
    result = run_python_file(str(work_dir / "test_target.py"))

    # ④ 判定：退出码 0 = 通过
    return {
        "test_result": result["stdout"] + result["stderr"],
        "tests_passed": result["returncode"] == 0,
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

graph.add_node("plan", agentplan)
graph.add_node("code", agentcoder)
graph.add_node("review", agentreview)
graph.add_node("test", agenttest)

graph.set_entry_point("plan")
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
