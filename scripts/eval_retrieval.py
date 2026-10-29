"""检索效果评测：50 条中文 query，对比三种配置。

配置：
    vector —— 只走向量路（bge-m3 余弦）
    hybrid —— 向量 + 全文 + RRF 融合，不做精排（对应 Java 的降级路径）
    full   —— hybrid + bge-reranker-v2-m3 精排 + 相关性阈值（线上实际用的）

指标按【文档】算，不按 chunk 算：
    Recall@k —— 目标文档是否出现在前 k 篇不同文档里
    MRR      —— 目标文档第一次出现的名次取倒数，未命中记 0
按文档算的原因：系统最终按 document_id 归并后返回 k 篇不同文档，
拿 chunk 算 Recall 会把"同一篇文档的 3 个片段"算成 3 次命中。

数据集是 known-item 检索：每条 query 指向一篇确定文档。这个构造方式
会把难度压低（query 与标题有语义重叠），所以数字用来看【配置之间】的
相对差异，不要当作绝对效果对外宣称。

用法：
    python -m scripts.eval_retrieval                 # 全跑
    python -m scripts.eval_retrieval --mode full     # 只跑线上配置
    python -m scripts.eval_retrieval --limit 10      # 先跑 10 条看耗时
"""
import json
import time
from argparse import ArgumentParser
from pathlib import Path

from app.db import close_pool, fetch_all
from app.retrieval.fusion import build_hybrid_retriever
from app.retrieval.pipeline import retrieve
from app.retrieval.retrievers import VectorRetriever

K_LIST = (5, 10)
CANDIDATE_K = 50          # 每路候选窗口，与线上一致
OUT_PATH = Path("eval_out/retrieval_eval.json")

# —— 数据集：35 条指向官方文档，15 条指向 issue/PR ——
# DOC 用精确标题定位（同名文档可能有多篇，视为都算命中）；issue/PR 用编号定位
EVAL_SET: list[dict] = [
    {"q": "不重启应用的情况下替换已经加载的类文件", "title": "Hot Swapping"},
    {"q": "把 Spring Boot 应用打包成一个可以直接运行的 jar", "title": "Packaging Executable Archives"},
    {"q": "把应用编译成原生可执行文件提升启动速度", "title": "GraalVM Native Applications"},
    {"q": "集成 Quartz 做定时任务", "title": "Quartz (`quartz`)"},
    {"q": "查看应用启动阶段每一步耗时", "title": "Application Startup (`startup`)"},
    {"q": "日志输出格式和输出到文件的配置", "title": "Logging"},
    {"q": "基于 JMS 收发消息", "title": "Messaging"},
    {"q": "记录每一次 HTTP 请求和响应用于审计", "title": "HTTP Exchanges (`httpexchanges`)"},
    {"q": "给内嵌服务器配置 HTTPS 证书", "title": "SSL"},
    {"q": "开启生产环境可用的健康检查等运维特性", "title": "Enabling Production-ready Features"},
    {"q": "启动时自动执行建表脚本初始化数据库", "title": "Database Initialization"},
    {"q": "把 Redis 集成进来做缓存", "title": "Working with NoSQL Technologies"},
    {"q": "编写批处理任务并定时执行", "title": "Batch Applications"},
    {"q": "导出线程栈来诊断卡死的请求", "title": "Thread Dump (`threaddump`)"},
    {"q": "自动配置类的生效条件和加载原理", "title": "Auto-configuration"},
    {"q": "actuator 对外暴露的 REST 接口有哪些", "title": "Actuator REST API"},
    {"q": "代码应该按什么包结构组织，启动类放哪里", "title": "Structuring Your Code"},
    {"q": "用命令行工具直接运行 Groovy 脚本", "title": "Spring Boot CLI"},
    {"q": "开发阶段用容器起数据库等外部依赖", "title": "Development-time Services"},
    {"q": "自己写一个 starter 让别人自动装配", "title": "Creating Your Own Auto-configuration"},
    {"q": "从旧版本升级到新版本要注意什么", "title": "Upgrading Spring Boot"},
    {"q": "运行中动态调整某个类的日志级别", "title": "Loggers"},
    {"q": "用 @SpringBootTest 写集成测试", "title": "Testing Spring Boot Applications"},
    {"q": "通过端点拿到日志文件的路径", "title": "Log File (`logfile`)"},
    {"q": "查看环境里所有生效的配置属性", "title": "Configuration Properties (`configprops`)"},
    {"q": "通过 HTTP 访问各监控端点", "title": "Monitoring and Management Over HTTP"},
    {"q": "把应用部署到 Cloud Foundry", "title": "Cloud Foundry Support"},
    {"q": "RabbitMQ 的连接和收发配置", "title": "AMQP"},
    {"q": "给管理端点加上访问认证", "title": "Security"},
    {"q": "用 RestClient 调用外部 HTTP 服务", "title": "HTTP Clients"},
    {"q": "可执行 jar 里面嵌套 jar 的目录结构", "title": "Nested JARs"},
    {"q": "测试里启动完整服务器用的工具类", "title": "Test Utilities"},
    {"q": "停机时等待正在处理的请求执行完", "title": "Graceful Shutdown"},
    {"q": "构建 Docker 镜像并推送", "title": "Packaging OCI Images"},
    {"q": "@SpringBootApplication 里包含哪几个注解", "title": "Using the @SpringBootApplication Annotation"},
    {"q": "上下文关闭后指标注册表没有被移除",
     "doc_type": "PR", "external_id": "50235"},
    {"q": "不安装浏览器插件也能用 LiveReload 热刷新",
     "doc_type": "PR", "external_id": "43697"},
    {"q": "原生镜像里配置属性是 Class 类型时反射提示没注册",
     "doc_type": "PR", "external_id": "52127"},
    {"q": "management 端点端口和 server 端口不一致导致客户端连错",
     "doc_type": "ISSUE", "external_id": "28423"},
    {"q": "extract 命令生成的 jar 每次构建结果都不一致",
     "doc_type": "ISSUE", "external_id": "48678"},
    {"q": "PathPatternRequestMatcher.Builder 没有被自动配置",
     "doc_type": "ISSUE", "external_id": "49988"},
    {"q": "构建期绑定的配置属性值把运行期的覆盖了",
     "doc_type": "ISSUE", "external_id": "50956"},
    {"q": "Gradle 构建开启配置缓存后失败",
     "doc_type": "ISSUE", "external_id": "51459"},
    {"q": "pid 文件写入时没有正确处理符号链接",
     "doc_type": "ISSUE", "external_id": "50173"},
    {"q": "测试属性源里的空系统属性被清掉了而不是恢复",
     "doc_type": "ISSUE", "external_id": "52208"},
    {"q": "自动配置 OpenTSDB 的指标注册表",
     "doc_type": "ISSUE", "external_id": "21063"},
    {"q": "解压 buildpack 层时输出路径没有归一化",
     "doc_type": "ISSUE", "external_id": "50141"},
    {"q": "MappingsEndpoint 把上下文自己的 ID 当成了 parentId",
     "doc_type": "ISSUE", "external_id": "50368"},
    {"q": "Kotlin 自动配置示例里漏了注解",
     "doc_type": "ISSUE", "external_id": "48220"},
    {"q": "AOT 测试时目录已存在的异常",
     "doc_type": "ISSUE", "external_id": "47837"},
]

# 拒答能力：库里没有答案的问题，应当返回空 documents
REFUSE_SET = [
    "asdfghjkl qwerty zxcvbn",
    "红烧肉怎么做才好吃",
    "帮我写一首关于秋天的唐诗",
    "如何挑选送女朋友的生日礼物",
    "今天上海天气怎么样",
]


def resolve_target(case: dict) -> set[int]:
    """把数据集里的目标转成 document.id 集合（同名文档可能不止一篇）。"""
    if "external_id" in case:
        rows = fetch_all(
            "SELECT id FROM document WHERE doc_type = %s AND external_id = %s",
            (case["doc_type"], case["external_id"]),
        )
    else:
        rows = fetch_all(
            "SELECT id FROM document WHERE doc_type = 'DOC' AND title = %s",
            (case["title"],),
        )
    return {r["id"] for r in rows}


def top_document_ids(docs, k: int) -> list[int]:
    """按文档去重后取前 k 篇，顺序即召回名次。"""
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


def first_rank(targets: set[int], doc_ids: list[int]) -> int | None:
    for i, did in enumerate(doc_ids, 1):
        if did in targets:
            return i
    return None


def run_mode(mode: str, query: str):
    if mode == "vector":
        return VectorRetriever(k=CANDIDATE_K).invoke(query)
    if mode == "hybrid":
        return build_hybrid_retriever().invoke(query)
    if mode == "full":
        return retrieve(query, top_k=max(K_LIST)).documents
    raise ValueError(mode)


def evaluate(mode: str, limit: int | None = None) -> dict:
    cases = EVAL_SET[:limit] if limit else EVAL_SET
    per_query = []
    t0 = time.time()

    for i, case in enumerate(cases, 1):
        targets = resolve_target(case)
        if not targets:
            raise SystemExit(f"数据集第 {i} 条没解析到目标文档: {case}")

        docs = run_mode(mode, case["q"])
        ranks = {}
        for k in K_LIST:
            ids = top_document_ids(docs, k)
            ranks[k] = first_rank(targets, ids)

        per_query.append(
            {
                "query": case["q"],
                "target": sorted(targets),
                "candidates": len(docs),
                **{f"rank@{k}": ranks[k] for k in K_LIST},
            }
        )
        hit = " ".join(f"@{k}={r if r else 'MISS'}" for k, r in ranks.items())
        print(f"[{i:>2}/{len(cases)}] {hit:<22} {case['q'][:38]}")

    elapsed = time.time() - t0
    summary = {"mode": mode, "n": len(cases), "elapsed_s": round(elapsed, 1)}
    for k in K_LIST:
        hits = [q[f"rank@{k}"] for q in per_query if q[f"rank@{k}"]]
        summary[f"recall@{k}"] = round(len(hits) / len(cases), 4)
        summary[f"mrr@{k}"] = round(sum(1 / h for h in hits) / len(cases), 4)
    summary["avg_candidates"] = round(
        sum(q["candidates"] for q in per_query) / len(cases), 1
    )
    return {"summary": summary, "queries": per_query}


def check_refusal() -> dict:
    """线上配置下，无关问题应当返回空，作为拒答依据。"""
    print("\n=== 拒答（无关问题）===")
    refused = 0
    for q in REFUSE_SET:
        docs = retrieve(q).documents
        ok = not docs
        refused += ok
        print(f"{'拒答' if ok else '未拒答'}  返回 {len(docs)} 条  {q}")
    return {"n": len(REFUSE_SET), "refused": refused,
            "refusal_rate": round(refused / len(REFUSE_SET), 4)}


def main() -> None:
    p = ArgumentParser(description="检索效果评测")
    p.add_argument("--mode", default="all",
                   choices=["all", "vector", "hybrid", "full"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, default=OUT_PATH)
    p.add_argument("--dry-run", action="store_true",
                   help="只校验 50 条目标文档能否解析，不调模型")
    args = p.parse_args()

    if args.dry_run:
        missing = []
        for i, case in enumerate(EVAL_SET, 1):
            n = len(resolve_target(case))
            if not n:
                missing.append((i, case))
            print(f"[{i:>2}] 目标 {n} 篇  {case['q'][:40]}")
        if missing:
            print("\n解析不到目标的条目：")
            for i, case in missing:
                print(f"  [{i}] {case}")
        else:
            print(f"\n全部 {len(EVAL_SET)} 条目标解析成功")
        close_pool()
        return

    modes = ["vector", "hybrid", "full"] if args.mode == "all" else [args.mode]
    report: dict = {"k_list": list(K_LIST), "runs": []}

    for mode in modes:
        print(f"\n===== 配置: {mode} =====")
        report["runs"].append(evaluate(mode, args.limit))

    if args.mode in ("all", "full"):
        report["refusal"] = check_refusal()

    print("\n===== 汇总 =====")
    print(f"{'配置':<10}{'Recall@5':>10}{'MRR@5':>9}{'Recall@10':>11}{'MRR@10':>9}{'候选数':>8}")
    for run in report["runs"]:
        s = run["summary"]
        print(f"{s['mode']:<10}{s['recall@5']:>10.4f}{s['mrr@5']:>9.4f}"
              f"{s['recall@10']:>11.4f}{s['mrr@10']:>9.4f}{s['avg_candidates']:>8.1f}")
    if "refusal" in report:
        r = report["refusal"]
        print(f"\n拒答: {r['refused']}/{r['n']} = {r['refusal_rate']:.0%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细已写入 {args.out}")
    close_pool()


if __name__ == "__main__":
    main()
