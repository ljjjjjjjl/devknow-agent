"""命令行入口：`python -m app.ingest <命令>`。

四个命令对应四条独立动作，可以分开跑、重复跑（幂等）：

    sync-issues  拉 issue 与 PR
    sync-docs    下载仓库文档
    embed        补向量
    stats        看入库统计
"""
import argparse
import logging
import sys
from datetime import datetime

from app.ingest import store


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _cmd_sync_issues(args: argparse.Namespace) -> int:
    from app.ingest.pipeline import sync_issues

    since = datetime.fromisoformat(args.since) if args.since else None
    stats = sync_issues(args.source, since=since, max_pages=args.max_pages)
    print(stats)
    return 0


def _cmd_sync_docs(args: argparse.Namespace) -> int:
    from app.ingest.pipeline import sync_docs

    stats = sync_docs(args.source, ref=args.ref, limit=args.limit)
    print(stats)
    return 0


def _cmd_embed(args: argparse.Namespace) -> int:
    from app.ingest.embed import backfill

    n = backfill(limit=args.limit, batch_size=args.batch)
    print(f"已向量化 {n} 条")
    return 0


def _cmd_stats(_args: argparse.Namespace) -> int:
    s = store.stats()
    print("文档:")
    for doc_type, n in s["by_type"].items():
        print(f"  {doc_type:<8} {n}")
    print(f"chunk 总数: {s['chunks']}")
    print(f"已向量化  : {s['embedded']}")
    print(f"待向量化  : {s['chunks'] - s['embedded']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.ingest", description="devknow 数据采集")
    parser.add_argument("-v", "--verbose", action="store_true", help="打印调试日志")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sync-issues", help="拉取 GitHub issue 与 PR")
    p.add_argument("--source", type=int, default=2, help="knowledge_source.id，默认 2")
    p.add_argument("--since", default=None, help="ISO 时间；不给则用库里的 last_synced_at")
    p.add_argument("--max-pages", type=int, default=None, help="最多翻多少页（调试用）")
    p.set_defaults(func=_cmd_sync_issues)

    p = sub.add_parser("sync-docs", help="下载并解析仓库文档")
    p.add_argument("--source", type=int, default=2, help="knowledge_source.id，默认 2")
    p.add_argument("--ref", default=None, help="分支或 tag，默认取数据源配置")
    p.add_argument("--limit", type=int, default=None, help="最多处理多少篇（调试用）")
    p.set_defaults(func=_cmd_sync_docs)

    p = sub.add_parser("embed", help="为缺向量的 chunk 补算向量")
    p.add_argument("--limit", type=int, default=None, help="本次最多处理多少条")
    p.add_argument("--batch", type=int, default=None, help="每批条数")
    p.set_defaults(func=_cmd_embed)

    p = sub.add_parser("stats", help="查看入库统计")
    p.set_defaults(func=_cmd_stats)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
