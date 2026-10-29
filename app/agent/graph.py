"""把节点连成图。

对应 Java：AgentRunner 的整体控制流（max_steps=8）。

图结构：
    START → plan → [有 tool_calls?] → retrieve → [可拒答且空?] → refuse → END
                 └─(无)──────────────→ answer      └─(否则)───────→ answer
    answer → [有 tool_calls 且 steps < MAX_STEPS?] → retrieve → ...

要点：
- MAX_STEPS=8 对应 Java 的 max_steps（模型调用次数）
- 上限用两层守：steps 计数做语义上限，recursion_limit 做机械兜底
- recursion_limit 数的是节点执行次数，不是模型调用次数：1 次调用 ≈ 2 个节点
"""
from langgraph.graph import END, START, StateGraph

from app.agent.nodes import answer, plan, refuse, retrieve
from app.agent.state import AgentState

MAX_STEPS = 8

# 8 次模型调用 = 1 plan + 7 answer + 7 retrieve = 15 个节点，+2 余量
RECURSION_LIMIT = MAX_STEPS * 2 + 2


def _plan_branch(state: AgentState) -> str:
    """plan 是否真的产出了工具调用（防御空回合）。"""
    return "retrieve" if state["messages"][-1].tool_calls else "answer"


def _refuse_branch(state: AgentState) -> str:
    """精排开启且命中为空 → 确定性拒答。"""
    if state.get("can_refuse") and not state.get("hits"):
        return "refuse"
    return "answer"


def _continue_branch(state: AgentState) -> str:
    """answer 之后：还有工具调用且没到步数上限 → 回 retrieve 再查一轮。"""
    last = state["messages"][-1]
    if last.tool_calls and state.get("steps", 0) < MAX_STEPS:
        return "retrieve"
    return END


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("plan", plan)
    g.add_node("retrieve", retrieve)
    g.add_node("answer", answer)
    g.add_node("refuse", refuse)

    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", _plan_branch, {"retrieve": "retrieve", "answer": "answer"})
    g.add_conditional_edges("retrieve", _refuse_branch, {"refuse": "refuse", "answer": "answer"})
    g.add_conditional_edges("answer", _continue_branch, {"retrieve": "retrieve", END: END})
    g.add_edge("refuse", END)

    return g.compile()
