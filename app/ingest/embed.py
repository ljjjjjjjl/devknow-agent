"""向量回填：把还没算向量的 chunk 送去 embedding 并写回。

对应 Java：`EmbeddingBackfillService`（那边还要自己管分批、重试、限流，
这里 `OpenAIEmbeddings` 全包了）。

幂等的关键在于"待办列表就是一条查询"：
`WHERE embedding IS NULL` 走 V2 建的部分索引，不需要进度表或游标列。
"""
import logging

from app.config import get_settings
from app.ingest import store
from app.ingest.config import get_ingest_settings
from app.llm import get_embeddings

logger = logging.getLogger(__name__)


def backfill(limit: int | None = None, batch_size: int | None = None) -> int:
    """循环处理直到没有待办或达到 limit。返回本次向量化的 chunk 条数。"""
    bsize = batch_size or get_ingest_settings().embed_batch_size
    model = get_settings().embedding_model
    embeddings = get_embeddings()

    total = 0
    while True:
        if limit is not None and total >= limit:
            break

        rows = store.select_unembedded(bsize)
        if not rows:
            break

        vectors = embeddings.embed_documents([r["content"] for r in rows])
        if len(vectors) != len(rows):
            raise RuntimeError(f"向量条数不匹配: 输入 {len(rows)} 输出 {len(vectors)}")

        total += store.write_embeddings(
            [(r["id"], vec) for r, vec in zip(rows, vectors)], model
        )
        logger.info("已向量化 %d 条（本轮 %d）", total, len(rows))

    return total
