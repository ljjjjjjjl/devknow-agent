"""非流式对话端点 + 两个端点共用的"一次回合"装配。

对应 Java：AgentController 的 /api/agent/chat，以及 ConversationService 的 resolve/append。

顺序不能反：
    解析会话 → 读历史 → 裁剪 → 压缩 → 落 user 消息 → 跑图 → 落 assistant
先落 user 再读历史，本轮问题会被当成历史，同一个问题在 messages 里出现两次。
"""
import logging
from dataclasses import dataclass
from functools import lru_cache

from fastapi import APIRouter, HTTPException
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from app.agent.graph import RECURSION_LIMIT, build_graph
from app.agent.nodes import build_system_prompt
from app.conversation import store
from app.conversation.summary import SUMMARY_PREFIX, maybe_summarize
from app.conversation.trim import trim

logger = logging.getLogger(__name__)

router = APIRouter()


@lru_cache
def get_graph():
    """图编译一次即可：编译结果无状态，跨请求复用安全。"""
    return build_graph()


class CitationModel(BaseModel):
    """引用条目，字段名与 Java 的 Citation 对齐。"""

    model_config = ConfigDict(populate_by_name=True)

    document_id: int = Field(alias="documentId")
    title: str = ""
    url: str = ""


class ChatRequest(BaseModel):
    """沿用 Java 版的 camelCase 字段名，前端和评估脚本不用改。"""

    model_config = ConfigDict(populate_by_name=True)

    agent_config_id: int = Field(alias="agentConfigId")
    conversation_id: int | None = Field(default=None, alias="conversationId")
    query: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    conversation_id: int = Field(alias="conversationId")
    answer: str
    steps: int
    citations: list[CitationModel]


@dataclass
class Turn:
    """一个回合的装配结果。"""

    conversation_id: int
    messages: list[BaseMessage]


def prepare_turn(req: ChatRequest) -> Turn:
    """解析会话 → 读历史 → 裁剪 → 压缩 → 落 user 消息 → 组装 messages。"""
    cid = store.resolve(req.agent_config_id, req.conversation_id, req.query)

    history = store.list_for_context(cid)  # 本轮问题还没写进去
    window = trim(history)
    summary_text = maybe_summarize(cid, history, len(window))

    store.append_user(cid, req.query)
    return Turn(cid, build_messages(window, summary_text, req.query))


def build_messages(
    window: list[dict], summary_text: str | None, query: str
) -> list[BaseMessage]:
    """system（三条硬约束）+ 可选摘要 + 历史窗口 + 本轮问题。"""
    has_history = bool(window) or bool(summary_text)
    msgs: list[BaseMessage] = [SystemMessage(content=build_system_prompt(has_history))]
    if summary_text:
        msgs.append(SystemMessage(content=SUMMARY_PREFIX + summary_text))
    for m in window:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        msgs.append(cls(content=m["content"]))
    msgs.append(HumanMessage(content=query))
    return msgs


def initial_state(messages: list[BaseMessage], query: str) -> dict:
    """图的初始状态：七个字段全给，steps 从 0 起（operator.add 归约需要初值）。"""
    return {
        "messages": messages,
        "query": query,
        "hits": [],
        "citations": [],
        "steps": 0,
        "top_score": None,
        "can_refuse": False,
    }


def message_text(msg) -> str:
    """取消息文本；兼容 content 为分块数组的情况。"""
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    out = []
    for part in content or []:
        if isinstance(part, str):
            out.append(part)
        elif isinstance(part, dict) and part.get("type") == "text":
            out.append(part.get("text") or "")
    return "".join(out)


def last_answer_text(state: dict) -> str:
    """最后一条 AI 消息的文本；拒答分支也是它。"""
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, AIMessage):
            return message_text(msg)
    return ""


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    try:
        turn = prepare_turn(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        final = await get_graph().ainvoke(
            initial_state(turn.messages, req.query),
            config={"recursion_limit": RECURSION_LIMIT},
        )
    except Exception as e:
        logger.exception("对话失败: conversationId=%s", turn.conversation_id)
        store.append_failure(turn.conversation_id, f"{type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="对话处理失败，请稍后重试") from e

    answer = last_answer_text(final)
    citations = final.get("citations") or []
    store.append_assistant(turn.conversation_id, answer, citations)

    return ChatResponse(
        conversationId=turn.conversation_id,
        answer=answer,
        steps=final.get("steps") or 0,
        citations=[CitationModel(**c) for c in citations],
    )
