"""多智能体代码开发小队：plan → code → review → test，失败自动回 code 重写；
复杂任务先问人确认方案；重试到上限时问人决定继续/重新规划/放弃"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import re
from pathlib import Path

from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from model import model
from tools import write_file, run_file, read_file
from state import agentstate
from prompt import planprompt, codeprompt, reviewprompt, testprompt, clarifyprompt, complexityprompt
from react import call_model
from tools import TOOLS_MAP, model_with_tools, model_with_tools_file, FILE_TOOLS_MAP
from structured import clarify_model, complexity_model
from memory import search_memory, format_memory_hits
from langgraph.types import interrupt
from langgraph.checkpoint.memory import InMemorySaver
MAX_ATTEMPTS = 5
MAX_CLARIFY = 3
COMPLEXITY_THRESHOLD = 0.6


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
    msgs = [SystemMessage(planprompt)]
    # 记忆检索：查历史项目经验，有命中就注入 prompt（避免重复踩坑）
    hits = search_memory(state["requirement"])
    if hits:
        print(f"\n[记忆] 命中 {len(hits)} 条历史项目经验")
        msgs.append(HumanMessage(
            f"历史项目经验（仅供参考；若与当前需求无关请忽略）：\n{format_memory_hits(hits)}"
        ))
    msgs.append(HumanMessage(f"用户需求：{state['requirement']}"))
    response = call_model(model=model_with_tools,messages=msgs,tools_map=TOOLS_MAP)
    plan = response.content
    # 复杂度判定（和 clarify 同一套 Qwen 结构化输出）：分数 + 压缩摘要一次返回
    dec = complexity_model.invoke([
        SystemMessage(complexityprompt),
        HumanMessage(f"需求：{state['requirement']}\n方案：{plan}"),
    ])
    return {"plan": plan, "complexity": dec.complexity, "plan_summary": dec.summary}


def should_ask_plan(state: agentstate):  # plan 节点的出口
    if state.get("complexity", 0) >= COMPLEXITY_THRESHOLD:
        return "human_plan"   # 复杂任务 → 先给人看方案、可给修改意见
    return "code"             # 简单任务 → 直接写代码


def human_plan(state: agentstate):
    # 复杂任务才走到这里：展示压缩摘要，让人确认或给修改意见
    answer = interrupt({
        "question": (
            f"任务复杂度 {state.get('complexity', 0):.2f}，方案摘要：\n"
            f"{state.get('plan_summary') or state['plan'][:200]}\n\n"
            f"直接回车 = 按方案继续；输入修改意见 = AI 带着你的指示写代码"
        ),
        "type": "plan_review",
    })
    return {"steering": answer}


def agentcoder(state: agentstate):
    msgs = [
        SystemMessage(codeprompt),
        HumanMessage(
            f"计划：{state['plan']}\n需求：{state['requirement']}"
            f"\n工作目录：{state['workdir']}"
            f"\n现有文件：{state.get('files') or '（空）'}"
        ),
    ]
    # 复杂任务时用户在 human_plan 给过修改意见 → 带上
    if state.get("steering"):
        msgs.append(HumanMessage(f"用户对方案的修改意见：{state['steering']}"))
    # 不管是审查还是测试打回来的，把两个字段都给它看，让 LLM 自己分辨
    if state.get("review"):
        msgs.append(HumanMessage(f"审查意见：{state['review']}"))
    if state.get("test_result"):
        msgs.append(HumanMessage(f"测试结果：{state['test_result']}"))

    # 模型通过文件工具自己读写磁盘；最后的文字回答只作总结
    # 产出校验：每轮结束后检查"是否真的调用了 write_file / edit_file"，
    # 没有产出 → 明确反馈再跑一轮（最多 3 轮），防止"只说不写"静默通过
    last_resp = None
    for _ in range(3):
        n_before = len(msgs)
        last_resp = call_model(model=model_with_tools_file, messages=msgs, tools_map=FILE_TOOLS_MAP, max_turns=8)
        wrote = any(
            isinstance(m, ToolMessage) and ("已写入" in m.content or "已编辑" in m.content)
            for m in msgs[n_before:]
        )
        if wrote:
            break
        msgs.append(HumanMessage(
            "上一轮你没有真正写任何文件。必须调用 write_file / edit_file 工具把代码写入磁盘，"
            "不要只输出文字或复述计划。按计划要求先写入口/主文件（Python 项目默认 main.py）。"
        ))

    # 跑完确定性记录：工作目录里现在有哪些文件
    files = sorted(f.name for f in Path(state["workdir"]).rglob("*") if f.is_file())
    # 解析模型声明的入口文件（总结最后一行：入口文件：xxx）；解析不到就兜底
    entry_file = "main.py"
    if last_resp is not None:
        m = re.search(r"入口文件\s*[:：]\s*([^\s,，；。]+)", last_resp.content or "")
        if m:
            entry_file = m.group(1).strip("`'\"")
        elif "main.py" not in files and len(files) == 1:
            entry_file = files[0]   # 没声明时：只有唯一文件就用它
    return {"files": files, "entry_file": entry_file}


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

def human_stop(state: agentstate):
    # 重试到上限才走到这里：摊开"尽力"的证据，让人决定
    answer = interrupt({
        "question": (
            f"AI 已自动重写 {state.get('attempt_count', 0)} 次仍未通过。\n"
            f"审查意见：{(state.get('review') or '')[:150]}\n"
            f"测试结果：{(state.get('test_result') or '')[:150]}\n\n"
            f"1) 继续重试  2) 重新规划  3) 放弃"
        ),
        "type": "stop_decision",
    })
    # 无论选哪个都重置尝试次数：继续=新预算；重规划=新方案；放弃=无所谓
    return {"user_decision": answer, "attempt_count": 0}


def after_human_stop(state: agentstate):  # human_stop 节点的出口
    d = (state.get("user_decision") or "").strip()
    if d.startswith("3") or "放弃" in d:
        return END
    if d.startswith("2") or "规划" in d:
        return "plan"
    return "code"   # 默认：继续重试

def agenttest(state: agentstate):
    work_dir = Path(state["workdir"])
    entry_file = state.get("entry_file") or "main.py"
    entry_path = work_dir / entry_file

    # ① 入口文件必须存在（编码节点通过工具写盘）
    if not entry_path.exists():
        return {
            "test_result": f"入口文件不存在: {entry_file}",
            "tests_passed": False,
            "attempt_count": state.get("attempt_count", 0) + 1,
        }

    suffix = entry_path.suffix.lower()

    # ② 纯文本交付物：无法运行，降级为内容检查（明确写清是降级，不假装通过）
    if suffix in (".txt", ".md"):
        content = entry_path.read_text(encoding="utf-8", errors="replace")
        length = len(content.strip())
        passed = length >= 20
        return {
            "test_result": f"[降级：文本内容检查] {entry_file} 内容 {length} 字符，{'通过' if passed else '过短，疑似空模板'}",
            "tests_passed": passed,
            "attempt_count": state.get("attempt_count", 0) + 1,
        }

    # ③ 非 Python：编译 + 冒烟（机器只验证"能编译、能跑起来"；超时=交互程序存活）
    if suffix != ".py":
        result = run_file(str(entry_path), timeout=5)
        if result.get("timed_out"):
            passed, note = True, "（5 秒超时：程序在运行/等待输入，视为交互程序存活）"
        elif result["returncode"] == 0:
            passed, note = True, ""
        else:
            passed, note = False, result["stderr"] or result["stdout"]
        test_result = f"[降级：编译+冒烟] {entry_file}\n{note}".strip()
        return {
            "test_result": test_result,
            "tests_passed": passed,
            "attempt_count": state.get("attempt_count", 0) + 1,
        }

    # ④ Python：完整逻辑（LLM 写断言，机器判结果）
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
        result = run_file(str(work_dir / "test_target.py"))
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
        return "human_stop"   # 重写太多次 → 问人决定，不再默默切断
    if state.get("review_passed", False):
        return "test"         # 审查通过 → 去测试
    return "code"             # 审查不通过 → 回编码重写


def should_end(state: agentstate):       # test 节点的出口
    if state.get("attempt_count", 0) >= MAX_ATTEMPTS:
        return "human_stop"   # 测试反复失败 → 问人决定
    if state.get("tests_passed", False):
        return END            # 测试通过 → 结束
    return "code"             # 测试失败 → 回编码重写


graph = StateGraph(agentstate)

graph.add_node("clarify", agentclarify)
graph.add_node("plan", agentplan)
graph.add_node("human_plan", human_plan)
graph.add_node("code", agentcoder)
graph.add_node("review", agentreview)
graph.add_node("test", agenttest)
graph.add_node("human_stop", human_stop)

graph.set_entry_point("clarify")
graph.add_conditional_edges("clarify", should_clarify, {
    "plan": "plan",
    END: END,
})
graph.add_conditional_edges("plan", should_ask_plan, {
    "human_plan": "human_plan",
    "code": "code",
})
graph.add_edge("human_plan", "code")
graph.add_edge("code", "review")
graph.add_conditional_edges("review", should_continue, {
    "code": "code",
    "test": "test",
    "human_stop": "human_stop",
})
graph.add_conditional_edges("test", should_end, {
    "code": "code",
    "human_stop": "human_stop",
    END: END,
})
graph.add_conditional_edges("human_stop", after_human_stop, {
    "plan": "plan",
    "code": "code",
    END: END,
})

app = graph.compile(checkpointer=InMemorySaver())
