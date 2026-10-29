"""仓库文档拉取：只走 tarball，不做 git clone。

Java 保留浅克隆是因为能顺带拿到精确的 commit hash 做增量锚点；
Python 版改用 GitHub API 直接取 latest commit sha（一次请求就够），
于是不需要 git 协议，也顺带避开了国内网络下 JGit 连接失败的老问题。

tarball 体积也小得多（实测同一仓库 6 MB vs 35 MB）。

**只解压匹配 glob 的文档文件**，不是全量解压。两个理由：
1. spring-boot 仓库里有路径超过 260 字符的 `.gradle` 文件，
   Windows 的 MAX_PATH 限制会让全量解压直接失败
2. 只解压 `.adoc` 能省掉绝大部分 IO，顺带把解压目录压平一层
"""
import fnmatch
import logging
import re
import shutil
import tarfile
from collections.abc import Iterator
from pathlib import Path

import httpx

from app.ingest.config import IngestSettings

logger = logging.getLogger(__name__)

_DOWNLOAD_TIMEOUT = 180.0


def download_tarball(
    settings: IngestSettings,
    owner: str,
    repo: str,
    ref: str,
    patterns: list[str],
    work_dir: str | None = None,
) -> Path:
    """下载 `codeload.github.com/{owner}/{repo}/tar.gz/{ref}`，只解压匹配 patterns 的文件。

    返回解压根目录（已去掉 tarball 的顶层 `<repo>-<ref>/` 那一层）。
    """
    dest = Path(work_dir or settings.work_dir) / f"{owner}-{repo}-{ref}"
    dest.mkdir(parents=True, exist_ok=True)

    src_dir = dest / "src"
    if src_dir.exists():
        shutil.rmtree(src_dir)      # 重跑前清干净，避免残留文件混入
    src_dir.mkdir(parents=True)

    archive = dest / "archive.tar.gz"
    url = f"{settings.codeload_base_url}/{owner}/{repo}/tar.gz/{ref}"
    logger.info("下载 tarball: %s", url)

    with httpx.stream("GET", url, follow_redirects=True, timeout=_DOWNLOAD_TIMEOUT) as resp:
        resp.raise_for_status()
        total = 0
        with open(archive, "wb") as f:
            for chunk in resp.iter_bytes():
                f.write(chunk)
                total += len(chunk)
    logger.info("下载完成: %.1f MB", total / 1024 / 1024)

    with tarfile.open(archive, "r:gz") as tar:
        members = _select_members(tar, patterns)
        logger.info("tarball 内共 %d 项，命中 glob 的文档 %d 个", len(tar.getmembers()), len(members))
        if not members:
            raise RuntimeError(f"tarball 里没有匹配 {patterns} 的文件")
        tar.extractall(src_dir, members=members, filter="data")

    return src_dir


def _select_members(tar: tarfile.TarFile, patterns: list[str]) -> list[tarfile.TarInfo]:
    """挑出匹配的文件，并把成员名里的顶层目录去掉。

    tarball 形如 `spring-boot-main/documentation/.../nosql.adoc`，
    去掉 `spring-boot-main/` 之后解压结果就是 `src/documentation/...`，
    路径短一截，对 Windows 的 MAX_PATH 友好。
    """
    all_members = tar.getmembers()
    top = ""
    for m in all_members:
        if "/" in m.name:
            top = m.name.split("/", 1)[0] + "/"
            break

    selected: list[tarfile.TarInfo] = []
    for m in all_members:
        if not m.isfile():
            continue
        rel = m.name[len(top):] if top and m.name.startswith(top) else m.name
        if not rel or rel.startswith("/") or ".." in rel.split("/"):
            continue
        if any(_glob_match(rel, p) for p in patterns):
            m.name = rel
            selected.append(m)
    return selected


def _glob_match(path: str, pattern: str) -> bool:
    """支持 `**` 的 glob 匹配。

    不用 `fnmatch`：它把 `*` 翻译成 `.*`，会跨目录匹配，
    `*.adoc` 就会误伤 `documentation/x/y.adoc`。
    """
    regex = _glob_to_regex(pattern)
    return re.fullmatch(regex, path) is not None


def _glob_to_regex(pattern: str) -> str:
    out = []
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "*":
            if pattern.startswith("**/", i):
                out.append("(?:.*/)?")
                i += 3
                continue
            if pattern.startswith("**", i):
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
            i += 1
            continue
        if ch == "?":
            out.append("[^/]")
            i += 1
            continue
        out.append(re.escape(ch))
        i += 1
    return "".join(out)


def iter_doc_files(root: Path, patterns: list[str]) -> Iterator[tuple[str, str]]:
    """按 glob 选文件，产出 (相对路径, 文本内容)。重复匹配只算一次。"""
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            rel = path.relative_to(root).as_posix()
            try:
                yield rel, path.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                logger.warning("读取失败，跳过: %s (%s)", rel, e)
