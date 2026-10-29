"""两条 SQL 包成 LangChain 的 BaseRetriever，供 EnsembleRetriever 融合。"""
import re

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.db import fetch_all
from app.llm import get_embeddings
from app.retrieval.sql import FULLTEXT_SQL, VECTOR_SQL

_ASCII_WORD = re.compile(r"[A-Za-z0-9]+")


def extract_english_terms(query: str) -> str:
    """只留查询里的 ASCII 词，供 english 分词器用。

    plainto_tsquery 会把中文词当成 lexeme 保留，再用 & 与英文词连接，
    只要有一个中文词整条查询就 0 命中。这里先过滤：纯中文返回空串，
    调用方据此跳过 DB。
    """
    return " ".join(_ASCII_WORD.findall(query))


def _to_docs(rows: list[dict]) -> list[Document]:
    return [
        Document(
            page_content=r["content"],
            metadata={
                "chunk_id": r["chunk_id"],
                "document_id": r["document_id"],
                "title": r["title"],
                "url": r["url"],
                "heading_path": r["heading_path"],
                "similarity": r["similarity"],
            },
        )
        for r in rows
    ]


class VectorRetriever(BaseRetriever):
    """向量路。k 是候选窗口（默认 50），不是最终 top-k。"""
    k: int = 50

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        vec = get_embeddings().embed_query(query)
        return _to_docs(fetch_all(VECTOR_SQL, {"vec": str(vec), "limit": self.k}))


class FullTextRetriever(BaseRetriever):
    """全文路。纯中文查询直接返回空、不碰 DB；含英文词才做 tsquery。"""
    k: int = 50

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        terms = extract_english_terms(query)
        if not terms:
            return []   # 跳过：english 分词器对纯中文无货，查了也是 0
        return _to_docs(fetch_all(FULLTEXT_SQL, {"query": terms, "limit": self.k}))