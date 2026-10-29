"""RRF 融合：只看名次，不看分数。

余弦相似度（0~1）和 ts_rank（0~0.1）不在一个量纲，直接加权求和等于
偷偷给向量路加权重。RRF 只看"排第几"：score = Σ w/(c + rank)。

返回两路全并集（最多 100 条），交给精排重排 + 按文档归并 + 截断。
"""
from langchain.retrievers import EnsembleRetriever

from app.config import CANDIDATE_WINDOW, RRF_C
from app.db import HNSW_EF_SEARCH
from app.retrieval.retrievers import FullTextRetriever, VectorRetriever


def build_hybrid_retriever(
    k: int = CANDIDATE_WINDOW,
    weights: tuple[float, float] = (0.5, 0.5),
) -> EnsembleRetriever:
    """k 是每路的候选窗口，不是最终 top-k。

    weights 只有比例有意义：(0.5, 0.5) 与 (1, 1) 等价，不要求和为 1。
    纯中文查询词法路 0 命中，RRF 自动退化成单路；含英文词的混合查询两路都有货。
    """
    if k > HNSW_EF_SEARCH:
        raise ValueError(f"候选窗口 {k} 超过 hnsw.ef_search {HNSW_EF_SEARCH}，向量路会少回")

    return EnsembleRetriever(
        retrievers=[VectorRetriever(k=k), FullTextRetriever(k=k)],
        weights=list(weights),
        c=RRF_C,
        id_key="chunk_id",   # 默认按 page_content 判重，正文相同的 chunk 会被并成一条
    )