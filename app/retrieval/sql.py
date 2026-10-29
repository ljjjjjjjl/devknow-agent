"""两条检索 SQL，返回字段对齐 Java 版的 SearchResult。

三个必须守住的地方：
1. 向量要显式转型 %(vec)s::vector
2. 全文路的点号归一化要与 V4 生成 content_tsv 的表达式逐字一致，否则静默 0 命中
3. ts_rank 不是 BM25
"""
VECTOR_SQL = """
    SELECT c.id           AS chunk_id,
           c.document_id  AS document_id,
           d.title        AS title,
           d.url          AS url,
           c.heading_path AS heading_path,
           c.content      AS content,
           1 - (c.embedding <=> %(vec)s::vector) AS similarity
    FROM document_chunk c
    JOIN document d ON d.id = c.document_id
    WHERE c.embedding IS NOT NULL
    ORDER BY c.embedding <=> %(vec)s::vector
    LIMIT %(limit)s
"""

FULLTEXT_SQL = """
    SELECT c.id           AS chunk_id,
           c.document_id  AS document_id,
           d.title        AS title,
           d.url          AS url,
           c.heading_path AS heading_path,
           c.content      AS content,
           ts_rank(c.content_tsv, query) AS similarity
    FROM document_chunk c
    JOIN document d ON d.id = c.document_id,
         plainto_tsquery('english', regexp_replace(%(query)s, '\\.', '. ', 'g')) query
    WHERE c.content_tsv @@ query
    ORDER BY similarity DESC
    LIMIT %(limit)s
"""