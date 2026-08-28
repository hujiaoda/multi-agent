

from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class DevTeamState(TypedDict):
    requirement: str #用户需求
    plan: str #AI规划
    code:str #AI编码
    review: str #AI审查
    review_passed: bool #是否通过
    test_result: str #测试结果
    tests_passed: bool #是否通过测试
    attempt_count: int #尝试次数
    workdir: str #本次运行的工作目录（work/<时间戳>/）
    
    need_more_info: bool #需求澄清：信息是否足够
    question: str #需求澄清：要向用户提的问题
    clarify_count: int #需求澄清轮数
class agentstate(DevTeamState):
    messages:Annotated[list,add_messages]
