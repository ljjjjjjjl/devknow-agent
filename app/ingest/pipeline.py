"""采集编排：拉取 → 解析 → 切分 → 入库。

对应 Java：`JobWorker` + `IngestionJobService` 的调度部分。
Python 版不引入任务队列 —— 批处理由命令行直接驱动，状态就是数据库本身。
"""
import logging
from datetime import datetime, timezone

from app.ingest import store
from app.ingest.chunk import chunk
from app.ingest.config import get_ingest_settings
from app.ingest.github import GitHubClient
from app.ingest.models import RawDoc, SyncStats
from app.ingest.parse import parse
from app.ingest.repo import download_tarball, iter_doc_files

logger = logging.getLogger(__name__)


def issue_breadcrumb(raw: RawDoc) -> str:
    """issue / PR 的 heading_path 根，逐字对齐 Java 的 `ChunkBackfillService.issueBreadcrumb`。

    不带源名：源名是可变的，写进几千条 heading_path 后改名要批量刷这些行。
    带标题：issue 正文几乎不含 markdown 标题（4318 篇里只有 106 篇有），
    不带标题的话路径会退化成纯 ID，对关键词加权毫无用处。
    """
    label = "PR" if raw.doc_type == "PR" else "issue"
    title = (raw.title or "").strip()
    base = f"{label} #{raw.external_id}"
    return f"{base} > {title}" if title else base


def sync_issues(
    source_id: int,
    *,
    since: datetime | None = None,
    max_pages: int | None = None,
) -> SyncStats:
    """拉取 issue 与 PR（GitHub 的同一个接口两种类型）。"""
    settings = get_ingest_settings()
    source = _require_source(source_id)
    owner, repo = source["repo_owner"], source["repo_name"]

    started = datetime.now(timezone.utc)   # 用它做下次的 since，避免同步期间更新的漏掉
    start_from = since or source["last_synced_at"]
    logger.info("同步 issue: %s/%s since=%s", owner, repo, start_from)

    stats = SyncStats()
    with GitHubClient(settings) as gh:
        for raw in gh.iter_issues(owner, repo, since=start_from, max_pages=max_pages):
            stats.seen += 1
            _save(stats, source_id, raw, chunk(raw.body, raw.title, issue_breadcrumb(raw)))

    store.mark_synced(source_id, started)
    logger.info("issue 同步完成: %s", stats)
    return stats


def sync_docs(source_id: int, *, ref: str | None = None, limit: int | None = None) -> SyncStats:
    """下载仓库 tarball，解析文档并入库。"""
    settings = get_ingest_settings()
    source = _require_source(source_id)
    owner, repo = source["repo_owner"], source["repo_name"]
    branch = ref or source["branch"] or "main"
    patterns = settings.doc_patterns or _patterns_of(source)
    logger.info("同步文档: %s/%s@%s, glob=%s", owner, repo, branch, patterns)

    root = download_tarball(settings, owner, repo, branch, patterns)
    stats = SyncStats()

    for rel, text in iter_doc_files(root, patterns):
        if limit is not None and stats.seen >= limit:
            break
        parsed = parse(rel, text)
        if parsed is None:
            continue

        stats.seen += 1
        raw = RawDoc(
            doc_type="DOC",
            external_id=rel,
            title=parsed.title,
            body=parsed.content,
            url=f"https://github.com/{owner}/{repo}/blob/{branch}/{rel}",
            version=branch,
            path=rel,
            published_at=None,
            metadata=parsed.metadata,
        )
        _save(stats, source_id, raw, chunk(parsed.content, parsed.title, parsed.breadcrumb))

    store.mark_synced(source_id, datetime.now(timezone.utc))
    logger.info("文档同步完成: %s", stats)
    return stats


def _save(stats: SyncStats, source_id: int, raw: RawDoc, chunks) -> None:
    """先比 hash 再决定要不要写 —— 省掉切分这个 CPU 活。"""
    existing = store.find_document(source_id, raw)
    if existing and existing["content_hash"] == raw.content_hash:
        stats.skipped += 1
        return

    _, status = store.save_document(source_id, raw, chunks)
    stats.chunks += len(chunks)
    if status == "created":
        stats.inserted += 1
    else:
        stats.updated += 1


def _require_source(source_id: int) -> dict:
    source = store.get_source(source_id)
    if source is None:
        raise ValueError(f"knowledge_source 不存在: {source_id}")
    return source


def _patterns_of(source: dict) -> list[str]:
    pattern = source.get("doc_path_pattern") or ""
    return [p.strip() for p in pattern.split(",") if p.strip()]
