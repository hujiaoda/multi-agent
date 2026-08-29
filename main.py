"""入口：跑一次多智能体代码开发小队"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage

from graph import app, MAX_CLARIFY
from tools import set_workspace

DEFAULT_REQU = "写一个命令行版贪吃蛇游戏，用方向键控制，分数实时显示"


def read_requirement(input_fn=input) -> str:
    """读取多行需求：连续两个空行结束；第一行直接回车用默认例子。"""
    print("请输入需求（多行支持：粘贴后连按两个回车结束；直接回车用默认例子）:")
    lines = []
    blank_count = 0
    while True:
        line = input_fn()
        if not line.strip():
            if not lines:
                break          # 第一行就回车 → 用默认例子
            blank_count += 1
            if blank_count >= 2:
                break          # 连续两个空行 → 输入结束
            continue           # 单个空行：只是记一笔，继续等
        lines.append(line)
        blank_count = 0
    return "\n".join(lines).strip() or DEFAULT_REQU


def main():
    requ = read_requirement()

    # 每次运行一个独立文件夹：work/<时间戳>/，不冲突、可追溯
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = Path("work") / run_id
    work_dir.mkdir(parents=True, exist_ok=True)
    set_workspace(str(work_dir))   # 让文件工具的相对路径都落在本次运行目录
    print(f"本次运行目录: {work_dir}\n")

    messages = [HumanMessage(requ)]
    state = {
        "requirement": requ,
        "workdir": str(work_dir),
        "messages": messages,
        "clarify_count": 0,
    }
    # 需求澄清循环：clarify 说信息不足 → 打印问题 → 收用户回答 → 重新跑一遍图
    while True:
        final_state = {}
        need_ask = False

        # 双模式流式：messages = 模型逐字输出；updates = 节点完成时汇报字段
        for mode, chunk in app.stream(state, stream_mode=["updates", "messages"]):
            if mode == "messages":
                # chunk 是 (消息片段, 元数据)，片段通常是几个字/token
                msg, metadata = chunk
                # 只打印给人看的纯文本；结构化输出的 JSON/工具调用片段跳过
                if msg.content and not msg.tool_call_chunks:
                    text = msg.content if isinstance(msg.content, str) else ""
                    if text.strip() and not text.strip().startswith(("{", "[")):
                        print(text, end="", flush=True)
            else:  # mode == "updates"
                # chunk 是 {节点名: 该节点更新的字段}，每次只有一个节点
                node_name, update = next(iter(chunk.items()))
                print(f"\n===== {node_name} 节点更新 =====")
                for key, value in update.items():
                    if key == "messages":
                        continue
                    if isinstance(value, str) and len(value) > 300:
                        print(f"{key}: {value[:300]}...")
                    else:
                        print(f"{key}: {value}")
                final_state.update(update)
                if node_name == "clarify" and update.get("need_more_info"):
                    need_ask = True

        # 信息足够（或问够了）→ 图已经一路跑到结束，退出循环
        if not need_ask or final_state.get("clarify_count", 0) >= MAX_CLARIFY:
            break

        # 信息不足 → 把问题抛给用户，回答追加进 messages，下一轮 clarify 能看到
        print(f"\nAI 需要补充信息：{final_state.get('question')}")
        answer = input("你的回答：").strip()
        messages.append(HumanMessage(answer))
        state = {
            "requirement": requ,
            "workdir": str(work_dir),
            "messages": messages,
            "clarify_count": final_state.get("clarify_count", 0),
        }

    print("\n========== 最终结论 ==========")
    print("审查通过:", final_state.get("review_passed"))
    print("测试通过:", final_state.get("tests_passed"))
    print("尝试次数:", final_state.get("attempt_count"))
    print(f"所有文件在: {work_dir}")


if __name__ == "__main__":
    main()
