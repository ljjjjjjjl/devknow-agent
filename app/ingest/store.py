"""document / document_chunk 的读写。

一条铁律：**文档与它的全部 chunk 必须在同一个事务里写入**。
否则切分失败时 document 已经落库，下次同步因为 content_hash 未变被跳过，
这篇文档的 chunk 永远不会被创建，成为永久孤儿。

幂等靠 `uk_document (source_id, doc_type, external_id, version)`，但
**不能只用 ON CONFLICT**：issue 没有 milestone 时 version 是 NULL，
PostgreSQL 的唯一约束对 NULL 不去重（没加 NULLS NOT DISTINCT），
所以这里显式先查再决定 insert / update，比较用 `IS NOT DISTINCT FROM`。

`document_chunk.content_tsv` 是 V3/V4 建的 GENERATED 列，写入时不能出现在列清单里。
"""
import logging

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.db import get_pool
from app.ingest.models import Chunk, RawDoc

logger = logging.getLogger(__name__)

_DOC_SELECT = (
    "SELECT id, content_hash FROM document "
    "WHERE source_id = %s AND doc_type = %s AND external_id = %s "
    "  AND version IS NOT DISTINCT FROM %s"
)

_DOC_INSERT = (
    "INSERT INTO document "
    "(source_id, doc_type, external_id, url, title, version, version_order, path, "
    " raw_content, content_hash, lang, published_at, status, metadata) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'ACTIVE', %s) "
    "RETURNING id"
)

_DOC_UPDATE = (
    "UPDATE document SET url = %s, title = %s, version_order = %s, path = %s, "
    "  raw_content = %s, content_hash = %s, lang = %s, published_at = %s, "
    "  metadata = %s, updated_at = now() "
    "WHERE id = %s"
)

_CHUNK_INSERT = (
    "INSERT INTO document_chunk "
    "(document_id, chunk_index, chunk_type, heading_path, heading_level, "
    " content, content_tokens, content_hash, metadata) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, '{}'::jsonb)"
)


def get_source(source_id: int) -> dict | None:
    """读数据源配置。doc_path_pattern / last_synced_at 都从这里取。"""
    with get_pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM knowledge_source WHERE id = %s", (source_id,))
            return cur.fetchone()


def find_document(source_id: int, raw: RawDoc) -> dict | None:
    """按自然键查已有文档，只看 id 与 content_hash，用来在切分前判断要不要干活。"""
    with get_pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                _DOC_SELECT, (source_id, raw.doc_type, raw.external_id, raw.version)
            )
            return cur.fetchone()


def save_document(source_id: int, raw: RawDoc, chunks: list[Chunk]) -> tuple[int, str]:
    """写入文档与它的全部 chunk。

    返回 (document_id, status)，status ∈ {"created", "updated", "unchanged"}。
    内容未变时直接返回 "unchanged" 且不重写 chunk。

    调用方应先用 find_document 比较 content_hash —— 那样可以跳过切分这个 CPU 活。
    """
    with get_pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                _DOC_SELECT, (source_id, raw.doc_type, raw.external_id, raw.version)
            )
            row = cur.fetchone()

            if row and row["content_hash"] == raw.content_hash:
                return row["id"], "unchanged"

            meta = Jsonb(raw.metadata)
            if row:
                doc_id = row["id"]
                status = "updated"
                cur.execute(
                    _DOC_UPDATE,
                    (
                        raw.url, raw.title, raw.version_order, raw.path,
                        raw.body, raw.content_hash, raw.lang, raw.published_at,
                        meta, doc_id,
                    ),
                )
            else:
                status = "created"
                cur.execute(
                    _DOC_INSERT,
                    (
                        source_id, raw.doc_type, raw.external_id, raw.url, raw.title,
                        raw.version, raw.version_order, raw.path, raw.body,
                        raw.content_hash, raw.lang, raw.published_at, meta,
                    ),
                )
                doc_id = cur.fetchone()["id"]

            # 内容变了就整篇重切：旧 chunk 全部作废
            cur.execute("DELETE FROM document_chunk WHERE document_id = %s", (doc_id,))
            if chunks:
                cur.executemany(
                    _CHUNK_INSERT,
                    [
                        (
                            doc_id, c.chunk_index, c.chunk_type, c.heading_path,
                            c.heading_level, c.content, c.content_tokens, c.content_hash,
                        )
                        for c in chunks
                    ],
                )

    return doc_id, status


def select_unembedded(limit: int = 32) -> list[dict]:
    """取还没算向量的 chunk，走 V2 的部分索引。"""
    with get_pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT id, content FROM document_chunk "
                "WHERE embedding IS NULL ORDER BY id LIMIT %s",
                (limit,),
            )
            return cur.fetchall()


def write_embeddings(items: list[tuple[int, list[float]]], model: str) -> int:
    """批量写回向量。psycopg 不认 list[float]，转成 pgvector 的文本格式再 cast。"""
    if not items:
        return 0
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE document_chunk "
                "SET embedding = %s::vector, embedding_model = %s, embedded_at = now() "
                "WHERE id = %s",
                [(_vector_literal(vec), model, cid) for cid, vec in items],
            )
    return len(items)


def _vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in vec) + "]"


def mark_synced(source_id: int, at, commit_sha: str | None = None) -> None:
    """记录本次同步的起点与版本锚点。

    传"开始时刻"而不是 now()：否则同步期间被更新的 issue 会被下一轮的 since 跳过。
    """
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE knowledge_source SET last_synced_at = %s, "
                "  last_synced_commit = COALESCE(%s, last_synced_commit), "
                "  updated_at = now() "
                "WHERE id = %s",
                (at, commit_sha, source_id),
            )


def stats() -> dict:
    """命令行 `stats` 用：文档数、chunk 数、已向量化数。"""
    with get_pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT doc_type, count(*) AS n FROM document GROUP BY 1 ORDER BY 2 DESC")
            by_type = {r["doc_type"]: r["n"] for r in cur.fetchall()}
            cur.execute(
                "SELECT count(*) AS total, count(embedding) AS embedded FROM document_chunk"
            )
            row = cur.fetchone()
    return {"by_type": by_type, "chunks": row["total"], "embedded": row["embedded"]}
