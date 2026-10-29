"""检索流水线：融合 → 精排（重排+归并+截断）→ top-k，一个入口。

对应 Java：SearchService.search()。agent 的检索工具只调 retrieve()，
不知道内部有几路。
"""
import logging
from dataclasses import dataclass

from langchain_core.documents import Document

from app.config import TOP_K, get_settings
from app.retrieval.fusion import build_hybrid_retriever
from app.retrieval.rerank import rerank

logger = logging.getLogger(__name__)

_warned_rerank_off = False


@dataclass(frozen=True)
class RetrievalResult:
    """retrieve() 的返回值。

    documents：≤ top_k 条候选。
    can_refuse：是否能用「documents 为空」作为拒答依据。
        True  —— 精排开启，documents 已按相关性阈值过滤，空 = 无相关。
        False —— 精排关闭，documents 只是融合结果截断，空不代表无相关。
    """
    documents: list[Document]
    can_refuse: bool


def _warn_rerank_off() -> None:
    """精排关闭时提示一次：拒答能力失效。全局只告警一次，避免刷屏。"""
    global _warned_rerank_off
    if _warned_rerank_off:
        return
    _warned_rerank_off = True
    logger.warning(
        "rerank 已关闭：retrieve() 不再按相关性阈值过滤，"
        "documents 可能无相关且不会为空，空列表不可作为拒答依据"
    )


def retrieve(
    query: str,
    *,
    top_k: int = TOP_K,
) -> RetrievalResult:
    """整个检索流程的入口。

    can_refuse=True 时，空 documents 才是可信的「无相关」信号；
    can_refuse=False 时，调用方必须停用拒答逻辑。
    """
    # 全并集进精排：RRF 对"只被一路命中"的 chunk 惩罚重，裁到 top-N 会把这类漏掉
    fused = build_hybrid_retriever().invoke(query)

    can_refuse = get_settings().rerank_enabled
    if not can_refuse:
        _warn_rerank_off()

    docs = rerank(query, fused, top_k=top_k)
    return RetrievalResult(documents=docs, can_refuse=can_refuse)