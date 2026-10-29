"""GitHub Issue / PR 拉取。

对应 Java：`GitHubClient` + `GitHubIssueSyncer` + `GitHubIssueMapper` + `PageResponse`。

两个取舍沿用 Java 版：
- **不拉评论**：每个 issue 单独请求评论会放大成数千次 API 调用，
  `comment_count` 只记录数量，正文靠 url 回溯
- **version 取 milestone 标题**：GitHub 上惯例把 milestone 命名成 3.2.x，
  正好可做版本过滤；没有 milestone 时为 None（写库时要走 NULL 安全的比较）
"""
import logging
import time
from collections.abc import Iterator
from datetime import datetime, timezone

import httpx

from app.ingest.config import IngestSettings
from app.ingest.models import RawDoc

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30.0


def _headers(token: str) -> dict[str, str]:
    h = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "devknow-ingest",
    }
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class GitHubClient:
    """带限速与重试的 GitHub REST 客户端。"""

    def __init__(self, settings: IngestSettings):
        self._s = settings
        self._client = httpx.Client(timeout=_DEFAULT_TIMEOUT, headers=_headers(settings.github_token))
        self._last_call = 0.0

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GitHubClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------------------------------------------------------- 内部

    def _throttle(self) -> None:
        """两次请求之间留最小间隔，避免触发二级限流。"""
        gap = time.monotonic() - self._last_call
        if gap < self._s.request_interval:
            time.sleep(self._s.request_interval - gap)
        self._last_call = time.monotonic()

    def _get(self, url: str, params: dict | None = None) -> httpx.Response:
        last_error = ""
        for attempt in range(1, self._s.max_retries + 1):
            self._throttle()
            resp = self._client.get(url, params=params)

            if resp.status_code in (403, 429):
                wait = self._rate_limit_wait(resp)
                last_error = f"HTTP {resp.status_code}，等待 {wait:.0f}s"
                logger.warning("被限流，%s 后重试（第 %d 次）", last_error, attempt)
                time.sleep(wait)
                continue

            if resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code}"
                backoff = min(2 ** attempt, 30)
                logger.warning("服务端错误 %s，%ds 后重试", last_error, backoff)
                time.sleep(backoff)
                continue

            resp.raise_for_status()
            return resp

        raise RuntimeError(f"请求 GitHub 失败（已重试 {self._s.max_retries} 次）: {last_error}")

    @staticmethod
    def _rate_limit_wait(resp: httpx.Response) -> float:
        """按 X-RateLimit-Reset 算等待秒数，上限 60 秒，兜底 10 秒。"""
        reset = resp.headers.get("x-ratelimit-reset")
        if not reset:
            return 10.0
        try:
            remain = float(reset) - datetime.now(timezone.utc).timestamp()
        except ValueError:
            return 10.0
        return max(1.0, min(remain, 60.0))

    # ---------------------------------------------------------------- 对外

    def iter_issues(
        self,
        owner: str,
        repo: str,
        since: datetime | None = None,
        max_pages: int | None = None,
    ) -> Iterator[RawDoc]:
        """按 updated_at 倒序翻页。返回的列表里既有 issue 也有 PR（GitHub 的语义）。"""
        page = 1
        limit = max_pages if max_pages is not None else self._s.max_pages
        url = f"{self._s.github_base_url}/repos/{owner}/{repo}/issues"

        while page <= limit:
            params = {
                "state": "all",
                "per_page": self._s.per_page,
                "sort": "updated",
                "direction": "desc",
                "page": page,
            }
            if since is not None:
                params["since"] = since.isoformat()

            items = self._get(url, params).json()
            if not isinstance(items, list) or not items:
                break

            logger.info("issue 第 %d 页：%d 条", page, len(items))
            for item in items:
                yield _to_raw_doc(item, self._s.max_content_chars)

            if len(items) < self._s.per_page:
                break
            page += 1

    def latest_commit_sha(self, owner: str, repo: str, ref: str) -> str | None:
        """拿分支最新 commit，作为文档增量同步的版本锚点（替代 git clone）。"""
        url = f"{self._s.github_base_url}/repos/{owner}/{repo}/commits/{ref}"
        try:
            data = self._get(url).json()
            return data.get("sha")
        except Exception as e:
            logger.warning("取最新 commit 失败，忽略版本锚点: %s", e)
            return None


def _to_raw_doc(item: dict, max_content_chars: int) -> RawDoc:
    """一条 issue/PR → RawDoc。字段映射与 Java 的 GitHubIssueMapper 保持一致。"""
    body = item.get("body") or ""
    if len(body) > max_content_chars:
        body = body[:max_content_chars]

    is_pr = "pull_request" in item
    milestone = item.get("milestone") or {}
    labels = [lb.get("name") for lb in (item.get("labels") or []) if lb.get("name")]

    return RawDoc(
        doc_type="PR" if is_pr else "ISSUE",
        external_id=str(item.get("number")),
        title=item.get("title") or "",
        body=body,
        url=item.get("html_url"),
        version=milestone.get("title"),
        version_order=None,
        path=None,
        lang="en",
        published_at=_parse_dt(item.get("created_at")),
        metadata={
            "is_pr": is_pr,
            "number": item.get("number"),
            "state": item.get("state"),
            "state_reason": item.get("state_reason"),
            "labels": labels,
            "comment_count": item.get("comments", 0),
            "updated_at": item.get("updated_at"),
            "closed_at": item.get("closed_at"),
            "milestone": milestone.get("title"),
        },
    )
