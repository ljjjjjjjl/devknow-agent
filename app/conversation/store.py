"""会话与消息的读写。

对应 Java：ConversationService 的 resolve / appendUser / appendAssistant。

要点：
- 表是 Java 版建的，字段不能改：conversation / conversation_message
- 只写 user 与 assistant 两种角色，不存 tool 往返（理由见 V6 迁移注释）
- seq 用 MAX(seq)+1，只在单端串行下正确 —— 这是已知限制，别假装解决了
- 失败也要落一条 error 非空的占位，否则回看是"有问无答"
- 摘要压缩是另一块，list_for_context 只返回原始历史，摘要注入留给后面
"""
import logging

from psycopg.types.json import Jsonb

from app.db import execute, fetch_all

logger = logging.getLogger(__name__)

TITLE_MAX_CHARS = 60
ERROR_MAX_CHARS = 500


def resolve(agent_config_id: int, conversation_id: int | None, first_query: str) -> int:
    """解析本轮要用的会话 id：没传就新建，传了校验存在/未删/归属一致。"""
    if conversation_id is None:
        return create(agent_config_id, first_query)
    row = _require_active(conversation_id)
    if row["agent_config_id"] != agent_config_id:
        raise ValueError(
            f"会话 {conversation_id} 属于配置 {row['agent_config_id']}，"
            f"与本次请求的 {agent_config_id} 不一致"
        )
    return conversation_id


def create(agent_config_id: int, first_query: str) -> int:
    """新建会话，标题取首条 query 截断。"""
    title = build_title(first_query)
    rows = fetch_all(
        "INSERT INTO conversation (agent_config_id, title) VALUES (%s, %s) RETURNING id",
        (agent_config_id, title),
    )
    cid = rows[0]["id"]
    logger.info("新建会话: id=%s configId=%s title=%s", cid, agent_config_id, title)
    return cid


def append_user(conversation_id: int, content: str) -> None:
    _append(conversation_id, "user", content, None, None)


def append_assistant(conversation_id: int, content: str, citations: list[dict] | None) -> None:
    _append(conversation_id, "assistant", content, citations, None)


def append_failure(conversation_id: int, reason: str) -> None:
    """失败占位：role=assistant 但 error 非空，list_for_context 会过滤掉它。"""
    _append(conversation_id, "assistant", None, None, (reason or "")[:ERROR_MAX_CHARS])


def list_for_context(conversation_id: int) -> list[dict]:
    """拉取历史（过滤失败占位），按 seq 升序，供 trim 裁剪。"""
    return fetch_all(
        "SELECT seq, role, content FROM conversation_message "
        "WHERE conversation_id = %s AND error IS NULL ORDER BY seq",
        (conversation_id,),
    )


def build_title(query: str) -> str:
    """标题取首条 query：折叠空白、截断 60 字符；空则"新会话"。"""
    t = " ".join((query or "").strip().split())
    if not t:
        return "新会话"
    return t[:TITLE_MAX_CHARS] + ("…" if len(t) > TITLE_MAX_CHARS else "")


def _next_seq(conversation_id: int) -> int:
    rows = fetch_all(
        "SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq "
        "FROM conversation_message WHERE conversation_id = %s",
        (conversation_id,),
    )
    return rows[0]["next_seq"]


def _append(conversation_id: int, role: str, content: str | None,
            citations: list[dict] | None, error: str | None) -> None:
    seq = _next_seq(conversation_id)
    execute(
        "INSERT INTO conversation_message "
        "(conversation_id, seq, role, content, citations, error) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        (
            conversation_id,
            seq,
            role,
            content,
            Jsonb(citations) if citations is not None else None,
            error,
        ),
    )
    # updated_at 没有触发器，不显式刷新的话会话列表会一直按创建时间排
    execute("UPDATE conversation SET updated_at = now() WHERE id = %s", (conversation_id,))

def get_summary(conversation_id: int) -> dict | None:
    """读摘要与水位线；会话不存在返回 None。"""
    rows = fetch_all(
        "SELECT summary, summary_upto_seq FROM conversation WHERE id = %s",
        (conversation_id,),
    )
    if not rows:
        return None
    return {"summary": rows[0]["summary"], "upto": rows[0]["summary_upto_seq"]}


def update_summary_cas(conversation_id: int, summary: str, new_upto: int, expected_upto: int) -> bool:
    """CAS 更新摘要：水位线不匹配（已被并发推进）则更新 0 行，返回 False。"""
    rows = fetch_all(
        "UPDATE conversation SET summary = %s, summary_upto_seq = %s "
        "WHERE id = %s AND summary_upto_seq = %s RETURNING id",
        (summary, new_upto, conversation_id, expected_upto),
    )
    return bool(rows)
def _require_active(conversation_id: int) -> dict:
    rows = fetch_all(
        "SELECT id, agent_config_id, status FROM conversation WHERE id = %s",
        (conversation_id,),
    )
    if not rows or rows[0]["status"] != "ACTIVE":
        raise ValueError(f"会话不存在或已删除: {conversation_id}")
    return rows[0]