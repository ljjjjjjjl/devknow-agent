"""SSE 流式端点，7 类事件。

对应 Java：AgentController 的 /api/agent/chat/stream + AgentStreamListener。

事件顺序：start → step* → tool → citation → delta* → done，异常则以 error 收尾。

astream_events 映射：
    on_chain_start name=plan/retrieve/answer/refuse → step
    on_chain_start name=retrieve                    → tool（query 取 state 末尾 tool_calls）
    on_chain_end   name=retrieve                    → citation
    on_chat_model_stream 且 langgraph_node=answer   → delta
    on_chain_end   name=LangGraph                   → done

节点必须把 config 传给模型调用，否则这里一个 on_chat_model_* 事件都收不到。

字段约定以 agent-demo.js 为准，改动前先看它读哪些键：
    step  → data.step（数字，前端显示"第 N 步"）
    done  → data.elapsedMs / data.model / data.answer / data.citations
    error → data.code / data.message
"""
import asyncio
import json
import logging
import time

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.agent.graph import RECURSION_LIMIT
from app.api.chat import (
    ChatRequest,
    get_graph,
    initial_state,
    last_answer_text,
    message_text,
    prepare_turn,
)
from app.config import get_settings
from app.conversation import store

logger = logging.getLogger(__name__)

router = APIRouter()

NODE_LABELS = {
    "plan": "理解问题",
    "retrieve": "检索知识库",
    "answer": "组织回答",
    "refuse": "无相关内容，拒答",
}

# 对应 Java ErrorCode 里的通用服务端错误
CODE_INTERNAL = 50000


def _sse(event: str, data: dict) -> dict:
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}


def _query_from_state(state) -> str | None:
    """从 state 末尾的 AIMessage.tool_calls 取本轮检索词。"""
    try:
        for msg in reversed(state.get("messages") or []):
            calls = getattr(msg, "tool_calls", None)
            if calls:
                return (calls[0].get("args") or {}).get("query") or None
    except Exception:
        return None
    return None


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> EventSourceResponse:
    # 参数/会话错误在开流之前抛出，客户端拿到 400 而不是一条 error 事件
    try:
        turn = prepare_turn(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    async def events():
        started = time.perf_counter()
        yield _sse("start", {"conversationId": turn.conversation_id, "query": req.query})

        answer = ""
        citations: list[dict] = []
        steps = 0
        step_no = 0
        streamed = False

        try:
            async for ev in get_graph().astream_events(
                initial_state(turn.messages, req.query),
                version="v2",
                config={"recursion_limit": RECURSION_LIMIT},
            ):
                kind = ev["event"]
                name = ev.get("name") or ""
                node = (ev.get("metadata") or {}).get("langgraph_node")

                if kind == "on_chain_start" and name in NODE_LABELS:
                    step_no += 1
                    yield _sse(
                        "step",
                        {"step": step_no, "name": name, "label": NODE_LABELS[name]},
                    )
                    if name == "retrieve":
                        query = _query_from_state(ev["data"].get("input"))
                        if query:
                            yield _sse(
                                "tool", {"name": "search_knowledge", "query": query}
                            )

                elif kind == "on_chain_end" and name == "retrieve":
                    out = ev["data"].get("output") or {}
                    got = out.get("citations") if isinstance(out, dict) else None
                    if got:
                        citations = got
                        yield _sse("citation", {"citations": got})

                elif kind == "on_chat_model_stream" and node == "answer":
                    text = message_text(ev["data"]["chunk"])
                    if text:
                        streamed = True
                        yield _sse("delta", {"text": text})

                elif kind == "on_chain_end" and name == "LangGraph":
                    final = ev["data"].get("output") or {}
                    steps = final.get("steps") or 0
                    citations = final.get("citations") or citations
                    answer = last_answer_text(final)

        except asyncio.CancelledError:
            # 客户端断开：不落占位，白占一条历史反而干扰下一轮
            logger.info("SSE 客户端断开: conversationId=%s", turn.conversation_id)
            raise
        except Exception as e:
            logger.exception("SSE 对话失败: conversationId=%s", turn.conversation_id)
            store.append_failure(turn.conversation_id, f"{type(e).__name__}: {e}")
            yield _sse("error", {"code": CODE_INTERNAL, "message": "对话处理失败，请稍后重试"})
            return

        # 拒答分支不经过模型，一个 delta 都不会有：补发整句，否则前端到 done 才见字
        if answer and not streamed:
            yield _sse("delta", {"text": answer})

        store.append_assistant(turn.conversation_id, answer, citations)
        yield _sse(
            "done",
            {
                "conversationId": turn.conversation_id,
                "answer": answer,
                "steps": steps,
                "citations": citations,
                "elapsedMs": int((time.perf_counter() - started) * 1000),
                "model": get_settings().chat_model,
            },
        )

    return EventSourceResponse(events())
