"""仓库文档解析：AsciiDoc / Markdown 归一化成 Markdown。

对应 Java：`AsciiDocParser` + `MarkdownParser` + `PathUtils.toBreadcrumb`。

AsciiDoc 做的是**归一化**而不是完整解析：把核心结构转成 Markdown，
让切分器只需要处理一种语法。

    = 文档标题      →  # 文档标题
    == 一级节       →  ## 一级节
    [source,java]   →  ```java
    ----            →  ```
    include::x[]    →  移除

刻意不处理：表格、脚注、交叉引用 <<x>>、属性替换 {version}。

breadcrumb 的规则是实测出来的（不是猜的）：
    路径里有 `pages/`  → 取 pages/ 之后的部分
    没有 `pages/`      → 整条路径用 " > " 连起来
两种情况都要去掉扩展名。实测样例：
    .../pages/aot.adoc        → aot
    .../partials/nav.adoc     → build-plugin > ... > partials > nav
"""
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_SOURCE_ATTR = re.compile(r"^\[source(?:,\s*([A-Za-z0-9+#-]+))?.*\]$")
_INCLUDE = re.compile(r"^include::[^\[]+\[.*\]$")
_BLOCK_ATTR = re.compile(r"^\[[A-Za-z].*\]$")


@dataclass
class ParsedDoc:
    """解析产物。"""

    relative_path: str
    content: str
    title: str
    breadcrumb: str
    metadata: dict = field(default_factory=dict)


def to_breadcrumb(relative_path: str) -> str:
    parts = [p for p in relative_path.replace("\\", "/").split("/") if p]
    if not parts:
        return ""
    parts[-1] = _strip_extension(parts[-1])
    if "pages" in parts:
        parts = parts[parts.index("pages") + 1:]
    return " > ".join(parts)


def _strip_extension(name: str) -> str:
    idx = name.rfind(".")
    return name[:idx] if idx > 0 else name


def extract_title(content: str, relative_path: str) -> str:
    """优先取第一个 `# ` 标题，否则退回文件名。"""
    for line in content.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return _strip_extension(Path(relative_path).name)


def strip_leading_comment_block(content: str) -> str:
    """去掉文件开头的 `////` 注释块（Spring 文档每页都有 Apache 许可证头）。

    这些内容对检索是纯噪声，还会把不同页面的向量拉近。
    """
    lines = content.split("\n")
    i = 0
    n = len(lines)
    while i < n and not lines[i].strip():
        i += 1
    if i < n and lines[i].lstrip().startswith("////"):
        i += 1
        while i < n and lines[i].strip() != "////":
            i += 1
        i += 1  # 跳过结束行
    return "\n".join(lines[i:]) + ("\n" if i < n else "")


def parse_asciidoc(relative_path: str, raw: str) -> ParsedDoc | None:
    if not raw or not raw.strip():
        return None

    cleaned = strip_leading_comment_block(raw)
    out: list[str] = []
    in_code = False
    code_lang = ""
    title: str | None = None
    removed_includes = 0

    for line in cleaned.splitlines():
        trimmed = line.strip()

        # [source,java] 属性行本身不输出，等遇到 ---- 再开代码块
        if not in_code:
            m = _SOURCE_ATTR.match(trimmed)
            if m:
                code_lang = m.group(1) or ""
                continue

        if trimmed == "----":
            if in_code:
                out.append("```")
                in_code = False
            else:
                out.append("```" + code_lang)
                in_code = True
            continue

        if in_code:
            out.append(line)
            continue

        if _INCLUDE.match(trimmed):
            removed_includes += 1
            continue

        if trimmed.startswith("="):
            level = 0
            while level < len(trimmed) and trimmed[level] == "=":
                level += 1
            if level <= 5 and len(trimmed) > level and trimmed[level] == " ":
                heading = trimmed[level + 1:].strip()
                if level == 1 and title is None:
                    title = heading
                out.append("#" * (level + 1) + " " + heading)
                continue

        # NOTE / TIP / cols 这类块属性直接丢弃，不参与检索
        if _BLOCK_ATTR.match(trimmed):
            continue

        out.append(line)

    if in_code:
        out.append("```")   # 文件末尾未闭合的代码块兜底

    content = "\n".join(out).strip()
    resolved = title if title else extract_title(content, relative_path)

    return ParsedDoc(
        relative_path=relative_path,
        content=content,
        title=resolved,
        breadcrumb=to_breadcrumb(relative_path),
        metadata={
            "source_path": relative_path,
            "format": "asciidoc",
            "line_count": len(content.splitlines()),
            "removed_includes": removed_includes,
        },
    )


def parse_markdown(relative_path: str, raw: str) -> ParsedDoc | None:
    if not raw or not raw.strip():
        return None
    content = raw.strip()
    return ParsedDoc(
        relative_path=relative_path,
        content=content,
        title=extract_title(content, relative_path),
        breadcrumb=to_breadcrumb(relative_path),
        metadata={
            "source_path": relative_path,
            "format": "markdown",
            "line_count": len(content.splitlines()),
        },
    )


def parse(relative_path: str, raw: str) -> ParsedDoc | None:
    """按扩展名分派。"""
    suffix = Path(relative_path).suffix.lower()
    if suffix in (".adoc", ".asciidoc"):
        return parse_asciidoc(relative_path, raw)
    return parse_markdown(relative_path, raw)
