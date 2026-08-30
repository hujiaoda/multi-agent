

from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class DevTeamState(TypedDict):
    requirement: str #用户需求
    plan: str #AI规划
    review: str #AI审查
    review_passed: bool #是否通过
    test_result: str #测试结果
    tests_passed: bool #是否通过测试
    attempt_count: int #尝试次数
    workdir: str #本次运行的工作目录（work/<时间戳>/）
    
    need_more_info: bool #需求澄清：信息是否足够
    question: str #需求澄清：要向用户提的问题
    clarify_count: int #需求澄清轮数

    files: list[str] #工作目录下的文件列表
    entry_file: str #工作目录下的入口文件（相对路径）
    
    complexity: float #plan 复杂度 0~1（>=阈值才问用户确认方案）
    plan_summary: str #plan 的简洁摘要（给用户快速确认用）
    steering: str #用户对方案的修改意见（空=按原方案继续）
    user_decision: str #重试到上限时用户的选择（继续/重新规划/放弃）
class agentstate(DevTeamState):
    messages:Annotated[list,add_messages]
