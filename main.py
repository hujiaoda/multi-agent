"""入口：跑一次多智能体代码开发小队"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage

from graph import app

DEFAULT_REQU = "写一个命令行版贪吃蛇游戏，用方向键控制，分数实时显示"


def main():
    requ = input("请输入需求（直接回车用默认例子）: ").strip()
    if not requ:
        requ = DEFAULT_REQU

    # 每次运行一个独立文件夹：work/<时间戳>/，不冲突、可追溯
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = Path("work") / run_id
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"本次运行目录: {work_dir}\n")

    # stream_mode="updates"：每跑完一个节点，立刻打印它更新了哪些字段
    final_state = {}
    for chunk in app.stream(
        {
            "requirement": requ,
            "workdir": str(work_dir),
            "messages": [HumanMessage(requ)],
        },
        stream_mode="updates",
    ):
        for node_name, update in chunk.items():
            print(f"\n===== {node_name} 节点更新 =====")
            for key, value in update.items():
                if key == "messages":
                    continue
                if isinstance(value, str) and len(value) > 300:
                    print(f"{key}: {value[:300]}...")
                else:
                    print(f"{key}: {value}")
            final_state.update(update)

    print("\n========== 最终结论 ==========")
    print("审查通过:", final_state.get("review_passed"))
    print("测试通过:", final_state.get("tests_passed"))
    print("尝试次数:", final_state.get("attempt_count"))
    print(f"所有文件在: {work_dir}")


if __name__ == "__main__":
    main()
