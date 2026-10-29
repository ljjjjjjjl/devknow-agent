"""环境自检：连得上库、数据齐、向量齐。

跑法：python scripts/check_db.py（在项目根目录）
退出码 0 = 正常，1 = 有异常
"""
import sys
from pathlib import Path

# 让脚本从任意目录运行都能 import app：把项目根加进 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import fetch_all

row = fetch_all(
    """
    select
        (select count(*) from document) as documents,
        (select count(*) from document_chunk) as chunks,
        (select count(*) from document_chunk where embedding is not null) as embedded
    """
)[0]

print("documents=%(documents)s chunks=%(chunks)s embedded=%(embedded)s" % row)

bad = []
if row["documents"] != 5344:
    bad.append("documents 预期 5344")
if row["chunks"] != 8580:
    bad.append("chunks 预期 8580")
if row["embedded"] != row["chunks"]:
    bad.append("存在没有向量的 chunk")

if bad:
    print("FAIL: " + "；".join(bad))
    sys.exit(1)
print("OK")