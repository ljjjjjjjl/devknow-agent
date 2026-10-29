"""数据库连接：连接池 + 取数函数。

对应 Java：Hikari 连接池那一层。

写入边界：
- document / document_chunk 只读，本模块不写
- conversation / conversation_message 可写，由会话模块负责

表结构以现库为准：document / document_chunk(
    content, embedding vector(1024), content_tsv, heading_path, chunk_type
) / conversation / conversation_message
"""
from functools import lru_cache

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.config import get_settings

HNSW_EF_SEARCH = 100  # 默认 40 < 候选窗口 50，不放大向量路回不满 50 条


def _configure(conn) -> None:
    """连接级 GUC 不跟库走，每条连接出池前都得重设一遍。

    手动 SET 必须提交：psycopg 的 execute 会隐式开事务，停在 INTRANS
    的连接会被池子丢弃，表现为一直 PoolTimeout。
    """
    conn.execute(f"SET hnsw.ef_search = {HNSW_EF_SEARCH}")
    conn.commit()


@lru_cache
def get_pool() -> ConnectionPool:
    """池子惰性创建：导入即连接，会让单测和无库环境 import 就失败。"""
    s = get_settings()
    return ConnectionPool(
        conninfo=s.conninfo,
        min_size=s.pool_min_size,
        max_size=s.pool_max_size,
        timeout=s.pool_timeout,
        kwargs={"row_factory": dict_row},   # 结果是 dict，不是 tuple
        configure=_configure,               # 每条连接出池前重设会话参数
        open=True,
    )


def fetch_all(sql: str, params=None) -> list[dict]:
    """执行查询并取回全部行，连接在 with 结束时归还池子。

    不返回 cursor：游标绑在连接上，连接一还就失效。
    """
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def execute(sql: str, params=None) -> None:
    """执行写操作（INSERT/UPDATE/DELETE），连接归还池子时自动提交。"""
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)


def close_pool() -> None:
    """应用关闭时调用（FastAPI lifespan），池子不关会留后台线程。"""
    get_pool().close()