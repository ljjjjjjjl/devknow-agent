"""从评测明细里按语料类型拆榜（不调模型，只读上次结果）。"""
import json
from pathlib import Path

report = json.loads(Path("eval_out/retrieval_eval.json").read_text(encoding="utf-8"))
DOC_N = 35  # 前 35 条指向官方文档，后 15 条指向 issue/PR


def split(run):
    return run["queries"][:DOC_N], run["queries"][DOC_N:]


def score(rows, k):
    hits = [q[f"rank@{k}"] for q in rows if q[f"rank@{k}"]]
    return len(hits) / len(rows), sum(1 / h for h in hits) / len(rows)


print(f"{'配置':<8}{'子集':<12}{'n':>3}{'R@5':>8}{'MRR@5':>8}{'R@10':>8}{'MRR@10':>8}")
for run in report["runs"]:
    docs, issues = split(run)
    for name, rows in (("官方文档", docs), ("issue/PR", issues)):
        r5, m5 = score(rows, 5)
        r10, m10 = score(rows, 10)
        print(f"{run['summary']['mode']:<8}{name:<12}{len(rows):>3}"
              f"{r5:>8.4f}{m5:>8.4f}{r10:>8.4f}{m10:>8.4f}")

print("\n各配置命中的 query 编号（@5）：")
for run in report["runs"]:
    miss = [i + 1 for i, q in enumerate(run["queries"]) if not q["rank@5"]]
    print(f"{run['summary']['mode']:<8} 未命中 {len(miss)} 条: {miss}")
