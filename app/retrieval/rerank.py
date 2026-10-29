"""精排：把融合后的候选交给 bge-reranker-v2-m3 重排，再按阈值截断。

对应 Java：SiliconFlowRerankService + RetrievalService.rerankAndMerge。

三个关键点：
1. 送审文本 = heading_path + "\n" + content（标题是最强相关性信号），截断到 800 字符
2. 按 document_id 归并：同一文档多个 chunk 只留最高分那个，保证 top-k 是 k 篇不同文档
3. 融合只负责"合并"，顺序不可信；真正决定 top-k 的是这里
"""
import httpx
from langchain_core.documents import Document

from app.config import MIN_RELEVANCE, RERANK_MAX_DOC_CHARS, TOP_K, get_settings


def _rerank_text(doc: Document, max_chars: int) -> str:
    """送 rerank 的文本：heading_path + 正文，再截断。"""
    heading = doc.metadata.get("heading_path") or ""
    content = doc.page_content or ""
    text = f"{heading}\n{content}" if heading else content
    return text[:max_chars]


def _merge_by_document(
    docs: list[Document], score_by_index: dict[int, float]
) -> list[tuple[float, Document]]:
    """按 document_id 归并：同一文档多个 chunk 竞争，取最高分那个当代表。

    少了这步，top-k 会退化成"一篇文档的多个片段"。
    """
    best: dict[object, tuple[float, Document]] = {}
    for i, d in enumerate(docs):
        score = score_by_index.get(i)
        if score is None:
            continue
        key = d.metadata.get("document_id")
        if key is None:
            key = d.metadata.get("chunk_id") or i
        cur = best.get(key)
        if cur is None or score > cur[0]:
            d.metadata["relevance_score"] = score
            best[key] = (score, d)
    return sorted(best.values(), key=lambda x: x[0], reverse=True)


def rerank(
    query: str,
    docs: list[Document],
    *,
    top_k: int = TOP_K,
    max_chars: int = RERANK_MAX_DOC_CHARS,
    min_score: float = MIN_RELEVANCE,
) -> list[Document]:
    """按相关性降序、score >= min_score、document_id 去重后取前 top_k 条。

    精排分写进 metadata["relevance_score"]；原来的 similarity 是召回分，
    二者不同量纲，别混。
    """
    if not docs:
        return []

    s = get_settings()
    if not s.rerank_enabled:
        return docs[:top_k]   # 精排关：直接取融合前 top_k，对应 Java 的 RRF 降级

    resp = httpx.post(
        f"{s.siliconflow_base_url}/rerank",
        headers={"Authorization": f"Bearer {s.siliconflow_api_key}"},
        json={
            "model": s.rerank_model,
            "query": query,
            "documents": [_rerank_text(d, max_chars) for d in docs],
            "top_n": len(docs),
        },
        timeout=30,
    )
    resp.raise_for_status()
    results = resp.json()["results"]

    score_by_index = {r["index"]: r["relevance_score"] for r in results}
    ranked = _merge_by_document(docs, score_by_index)

    return [d for sc, d in ranked if sc >= min_score][:top_k]