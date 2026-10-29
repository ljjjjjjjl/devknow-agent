"""会话列表 / 历史消息 / 软删，供前端左栏使用。

对应 Java：AgentController 的 /api/agent/conversations 三个端点。

返回统一套 ApiResponse 信封 {success, code, message, data}：
前端 apiJSON() 读的是 body.data，不套信封它拿到 undefined。
错误也必须套信封（HTTP 400 + body.code=40004），否则前端的
"会话已被删掉"分支（靠 e.code 判断）永远走不到。
"""
import logging

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from app.db import fetch_all

logger = logging.getLogger(__name__)

router = APIRouter()

# 对应 Java 的 ErrorCode：会话不存在或已删除
CODE_CONVERSATION_NOT_FOUND = 40004


def _ok(data) -> dict:
    return {"success": True, "code": 0, "message": "ok", "data": data}


def _not_found(conversation_id: int) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "code": CODE_CONVERSATION_NOT_FOUND,
            "message": f"会话不存在或已删除: {conversation_id}",
            "data": None,
        },
    )


@router.get("/conversations")
def list_conversations(agent_config_id: int = Query(alias="agentConfigId")) -> dict:
    """按 updated_at 倒序，只返回 ACTIVE。走的是 V6 的部分索引。"""
    rows = fetch_all(
        "SELECT id, title, created_at, updated_at FROM conversation "
        "WHERE agent_config_id = %s AND status = 'ACTIVE' "
        "ORDER BY updated_at DESC",
        (agent_config_id,),
    )
    return _ok(
        [
            {
                "id": r["id"],
                "title": r["title"],
                "createdAt": r["created_at"].isoformat(),
                "updatedAt": r["updated_at"].isoformat(),
            }
            for r in rows
        ]
    )


def _citation(c: dict) -> dict:
    """统一成 camelCase：同一张表里既有 Java 写的 documentId，也有早期 Python 写的 document_id。"""
    return {
        "documentId": c.get("documentId", c.get("document_id")),
        "title": c.get("title") or "",
        "url": c.get("url") or "",
    }


@router.get("/conversations/{conversation_id}/messages")
def list_messages(conversation_id: int):
    """历史消息按 seq 升序；失败占位（error 非空）也返回，前端单独渲染。"""
    head = fetch_all("SELECT id, status FROM conversation WHERE id = %s", (conversation_id,))
    if not head or head[0]["status"] != "ACTIVE":
        return _not_found(conversation_id)

    rows = fetch_all(
        "SELECT role, content, citations, error FROM conversation_message "
        "WHERE conversation_id = %s ORDER BY seq",
        (conversation_id,),
    )
    return _ok(
        [
            {
                "role": r["role"],
                "content": r["content"],
                "error": r["error"],
                "citations": [_citation(c) for c in (r["citations"] or [])],
            }
            for r in rows
        ]
    )


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: int):
    """软删：只改 status，消息行保留可审计。重复删除返回 40004。"""
    rows = fetch_all(
        "UPDATE conversation SET status = 'DELETED', updated_at = now() "
        "WHERE id = %s AND status = 'ACTIVE' RETURNING id",
        (conversation_id,),
    )
    if not rows:
        return _not_found(conversation_id)
    logger.info("软删会话: id=%s", conversation_id)
    return _ok({"id": conversation_id})
