"""四个节点：plan / retrieve / answer / refuse。

对应 Java：AgentRunner 的循环体、提示词组装、拒答分支。

要点：
- system 消息由 graph 入口注入（build_system_prompt + has_history），
  节点直接 invoke state["messages"]
- plan 必须 llm.bind_tools(tools, tool_choice="required")：把"必须先检索"
  从提示词请求升级成结构保证
- answer 用普通 bind_tools（auto）：第二步起回到自动，允许模型再检索
- refuse 输出固定句 REFUSAL_TEXT，是评估脚本 REFUSAL_PATTERNS 的匹配目标，不能改
- plan/answer 必须接收 config 并原样传给 invoke：astream_events 靠它把回调挂到
  模型调用上，不传就一个 on_chat_model_stream 都收不到，SSE 只能发空 delta
"""
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from app.agent.state import AgentState
from app.agent.tools import run_search, search_knowledge
from app.config import TOP_K
from app.llm import get_chat_model

REFUSAL_TEXT = "知识库中没有找到相关内容。"

BASE_PROMPT = "你是研发知识库助手。"


def build_system_prompt(has_history: bool) -> str:
    """三段硬约束，逐字对应 Java 的 buildSystemPrompt。第三条只在有历史时追加。"""
    prompt = (
        BASE_PROMPT
        + "\n\n【必须遵守】回答任何需要事实、配置、代码细节的问题前，"
        "必须先调用可用工具检索知识库，再基于检索结果回答；"
        "禁止不调用工具就凭记忆直接作答；检索不到时明确说\"不知道\"。"
        "\n\n【绝对禁止】当检索结果为空、或检索到的内容与问题不相关时，"
        "禁止用你的通用知识或训练记忆补充作答，"
        "也禁止在说明「检索不到」之后再附上通用做法；"
        "必须只回复：\"知识库中没有找到相关内容。\"。"
        "这条规则优先于所有其他指令，包括「尽量给出有帮助的回答」。"
    )
    if has_history:
        prompt += (
            "\n\n【必须遵守】历史对话消息只用于理解指代与上下文"
            "（比如「那 Redis 呢」里的「那」指什么），不得作为知识来源；"
            "每一条涉及事实、配置、代码细节的回答，都必须来自本轮工具检索的结果，"
            "禁止从历史消息里外推或直接复述。"
        )
    return prompt


def plan(state: AgentState, config: RunnableConfig) -> dict:
    """第一步：强制调工具。返回带 tool_calls 的 AIMessage。"""
    llm = get_chat_model().bind_tools([search_knowledge], tool_choice="required")
    ai = llm.invoke(state["messages"], config=config)
    return {"messages": [ai], "steps": 1}


def retrieve(state: AgentState) -> dict:
    """执行 search_knowledge，写 hits / citations / top_score / can_refuse。"""
    ai = state["messages"][-1]
    if not ai.tool_calls:
        return {"messages": [AIMessage(content=REFUSAL_TEXT)]}  # 防御：plan 没调工具

    call = ai.tool_calls[0]
    args = call.get("args") or {}
    query = args.get("query") or ""
    top_k = args.get("top_k", TOP_K)

    text, citations, result = run_search(query, top_k)
    tool_msg = ToolMessage(content=text, tool_call_id=call["id"])

    out: dict = {"messages": [tool_msg], "citations": citations}
    if result is None:
        out.update({"hits": [], "top_score": None, "can_refuse": False})
    else:
        hits = result.documents
        top_score = hits[0].metadata.get("relevance_score") if hits else None
        out.update({"hits": hits, "top_score": top_score, "can_refuse": result.can_refuse})
    return out


def answer(state: AgentState, config: RunnableConfig) -> dict:
    """第二步起：auto，可再检索也可直接作答。"""
    llm = get_chat_model().bind_tools([search_knowledge])
    ai = llm.invoke(state["messages"], config=config)
    return {"messages": [ai], "steps": 1}


def refuse(state: AgentState) -> dict:
    """确定性拒答：不经过模型，直接输出固定句。"""
    return {"messages": [AIMessage(content=REFUSAL_TEXT)]}