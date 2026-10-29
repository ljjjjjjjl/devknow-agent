"""用 Java 版的评估集，在新库上跑同一套指标。

为什么要这个脚本：`scripts/eval_retrieval.py` 用的题是另写的（50 条中文长问句），
和 `week4-queries.jsonl`（known-item 30 + chinese 10 + symbol 10）不是一回事，
两者的 Recall@5 不能互相比较。要做「Python 重写有没有掉点」的结论，只能同一套题。

口径与 Java 的 EvaluationService 逐条对齐：
    检索窗口 = 20，按 document_id 去重保序
    Hit@1    = 第一名就是目标文档
    Recall@5 = 目标文档落在前 5 篇不同文档内（MRR@5 也只在 5 名内计分）
    Recall@20= 落在窗口内就算

数据集里的目标存的是 url 而不是 document.id：id 是灌库时的代理键，
重新灌一次就变了，url 才是稳定的。

用法：
    python -m scripts.eval_java_set --build-mapping   # 首次：生成 eval/java-eval.json
    python -m scripts.eval_java_set                   # 跑三种配置
"""
import json
import time
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from app.config import get_settings
from app.db import close_pool, fetch_all
from app.retrieval.fusion import build_hybrid_retriever
from app.retrieval.pipeline import retrieve
from app.retrieval.retrievers import VectorRetriever

SRC_JSONL = Path(
    r"D:\nodejs\workspace\devknow\devknow-retrieval\src\main\resources\eval\week4-queries.jsonl"
)
DATASET = Path("eval/java-eval.json")
OUT_PATH = Path("eval_out/java_set_eval.json")

WINDOW = 20          # 与 Java 的 RETRIEVAL_WINDOW 一致
CANDIDATE_K = 50     # 每路候选窗口，与线上一致


def build_mapping() -> None:
    """把旧库的 expectedDocumentIds 换成 url，写进项目内的数据集文件。"""
    s = get_settings()
    rows = [json.loads(x) for x in SRC_JSONL.read_text(encoding="utf-8").splitlines() if x.strip()]
    ids = sorted({i for r in rows for i in r["expectedDocumentIds"]})

    with psycopg.connect(host=s.db_host, port=s.db_port, dbname="devknow_utf8",
                         user=s.db_user, password=s.db_password,
                         row_factory=dict_row) as conn, conn.cursor() as cur:
        cur.execute("SELECT id, url FROM document WHERE id = ANY(%s)", (ids,))
        url_by_id = {r["id"]: r["url"] for r in cur.fetchall()}

    out, dropped = [], []
    for r in rows:
        urls = [url_by_id.get(i) for i in r["expectedDocumentIds"]]
        if None in urls:
            dropped.append(r["id"])
            urls = [u for u in urls if u]
        out.append({"id": r["id"], "set": r["set"], "query": r["query"], "expectedUrls": urls})

    DATASET.parent.mkdir(parents=True, exist_ok=True)
    DATASET.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已写 {DATASET}，{len(out)} 条，子集 {dict(Counter(r['set'] for r in out))}")
    if dropped:
        print(f"有目标 id 在旧库查不到（已跳过该目标）: {dropped}")


def load_dataset() -> list[dict]:
    if not DATASET.exists():
        raise SystemExit(f"缺少 {DATASET}，先跑 --build-mapping")
    cases = json.loads(DATASET.read_text(encoding="utf-8"))

    # url -> 新库 document.id（同一 url 可能有多个版本，都算命中）
    url_to_ids: dict[str, list[int]] = {}
    for r in fetch_all("SELECT id, url FROM document"):
        url_to_ids.setdefault(r["url"], []).append(r["id"])

    unresolved = []
    for c in cases:
        ids = {i for u in c["expectedUrls"] for i in url_to_ids.get(u, [])}
        if not ids:
            unresolved.append(c["id"])
        c["targets"] = ids
    if unresolved:
        print(f"注意：这些 query 的目标文档在新库中找不到，将必然未命中: {unresolved}")
    return cases


def top_docs(docs, k: int = WINDOW) -> list[int]:
    seen: set = set()
    out: list[int] = []
    for d in docs:
        did = d.metadata.get("document_id")
        if did in seen:
            continue
        seen.add(did)
        out.append(did)
        if len(out) >= k:
            break
    return out


def run_mode(mode: str, query: str):
    if mode == "vector":
        return VectorRetriever(k=CANDIDATE_K).invoke(query)
    if mode == "hybrid":
        return build_hybrid_retriever().invoke(query)
    if mode == "full":
        return retrieve(query, top_k=WINDOW).documents
    raise ValueError(mode)


def summarize(items: list[dict]) -> dict:
    n = len(items)
    if not n:
        return {"n": 0}
    hit1 = sum(1 for i in items if i["rank"] == 1)
    r5 = [i["rank"] for i in items if i["rank"] and i["rank"] <= 5]
    r20 = sum(1 for i in items if i["rank"] and i["rank"] <= WINDOW)
    return {
        "n": n,
        "hit@1": round(hit1 / n, 4),
        "recall@5": round(len(r5) / n, 4),
        "recall@20": round(r20 / n, 4),
        "mrr@5": round(sum(1 / r for r in r5) / n, 4),
        "in_pool_not_top5": sum(1 for i in items if i["rank"] and i["rank"] > 5),
    }


def evaluate(mode: str) -> dict:
    cases = load_dataset()
    items, t0 = [], time.time()
    for i, c in enumerate(cases, 1):
        ids = top_docs(run_mode(mode, c["query"]))
        rank = next((k for k, d in enumerate(ids, 1) if d in c["targets"]), None)
        items.append({"id": c["id"], "set": c["set"], "query": c["query"],
                      "rank": rank, "targets": sorted(c["targets"])})
        print(f"[{i:>2}/{len(cases)}] {c['id']:<7} @{rank if rank else 'MISS':<5} {c['query'][:44]}")

    by_set = {}
    for s in ("known-item", "chinese", "symbol"):
        sub = [x for x in items if x["set"] == s]
        if sub:
            by_set[s] = summarize(sub)
    return {
        "mode": mode,
        "elapsed_s": round(time.time() - t0, 1),
        "overall": summarize(items),
        "by_set": by_set,
        "items": items,
    }


def main() -> None:
    p = ArgumentParser(description="用 Java 评估集在新库上跑同口径指标")
    p.add_argument("--mode", default="all", choices=["all", "vector", "hybrid", "full"])
    p.add_argument("--build-mapping", action="store_true")
    args = p.parse_args()

    if args.build_mapping:
        build_mapping()
        close_pool()
        return

    modes = ["vector", "hybrid", "full"] if args.mode == "all" else [args.mode]
    report = {"window": WINDOW, "runs": []}
    for mode in modes:
        print(f"\n===== 配置: {mode} =====")
        report["runs"].append(evaluate(mode))

    print("\n===== 总览（对齐 Java 口径，窗口 20）=====")
    head = f"{'配置':<8}{'Hit@1':>8}{'Recall@5':>10}{'Recall@20':>11}{'MRR@5':>8}"
    print(head)
    for run in report["runs"]:
        o = run["overall"]
        print(f"{run['mode']:<8}{o['hit@1']:>8.4f}{o['recall@5']:>10.4f}"
              f"{o['recall@20']:>11.4f}{o['mrr@5']:>8.4f}")

    print("\n===== 分子集 Recall@5 =====")
    sets = ["known-item", "chinese", "symbol"]
    print(f"{'配置':<8}" + "".join(f"{s:>12}" for s in sets))
    for run in report["runs"]:
        cells = "".join(f"{run['by_set'].get(s, {}).get('recall@5', float('nan')):>12.4f}"
                        for s in sets)
        print(f"{run['mode']:<8}{cells}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细已写入 {OUT_PATH}")
    close_pool()


if __name__ == "__main__":
    main()
