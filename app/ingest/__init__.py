"""离线采集模块：从 GitHub 拉取内容 → 解析 → 切分 → 向量化 → 入库。

对应 Java：devknow-ingestion 模块（23 个类）。

与在线服务的边界：
- 本模块**只写** document / document_chunk，不碰 conversation
- 在线服务**只读** document / document_chunk
- 两者共用同一个 PostgreSQL 库，接口就是表结构

入口：``python -m app.ingest --help``
"""
