"""按标题层级切分 Markdown，逐行对齐 Java 的 `HeadingChunker`。

为什么不定长切分：技术文档以"节"为语义单位，定长 512 token 会把一个配置示例
从中间截断，检索命中后半段时模型拿到的是没有前提的碎片。

三个尺寸常量的分工（改之前务必分清）：
    TARGET_SIZE             聚合目标，本文件内部使用
    MAX_SIZE                调用方判断"要不要二次切分"的触发阈值，split_long_text 不用它
    HARD_MAX_SIZE           切分方法保证的输出上界（代码块用 CODE_HARD_MAX_SIZE）

与 Java 版的已知差异：Java `String.length()` 数的是 UTF-16 单元，Python `len()`
数的是码点，遇到 emoji 等辅助平面字符时长度判断会差 1。只在含 emoji 的文档上体现。
"""
import re

from app.conversation.trim import estimate_tokens
from app.ingest.models import Chunk

TARGET_SIZE = 1_600
MAX_SIZE = 2_400
MIN_SIZE = 200
HARD_MAX_SIZE = 4_000
CODE_HARD_MAX_SIZE = 6_000

FENCE = "```"
BLOCK_SEPARATOR = "\n\n"

_SENTENCE_END = re.compile(r"(?<=[.!?。！？])")


def chunk(content: str | None, doc_title: str | None, breadcrumb: str | None = None) -> list[Chunk]:
    """切分入口。

    content    Markdown / 已转成 Markdown 的正文
    doc_title  文档标题，作为 heading_path 的根
    breadcrumb 来自文件路径的层级（可为空）
    """
    if not content or not content.strip():
        return []

    root = breadcrumb if (breadcrumb and breadcrumb.strip()) else (doc_title or "")

    chunks: list[Chunk] = []
    stack: list[str] = []          # 标题栈，元素形如 "2:Data"
    buf: list[str] = []
    current_level = 0
    index = 0
    opening_fence: str | None = None

    for line in content.split("\n"):
        trimmed = line.strip()

        # 围栏内的一切都原样保留，代码块里的 # 不能被当成标题
        if opening_fence is not None:
            if _is_closing_fence(trimmed, opening_fence):
                opening_fence = None
            buf.append(line + "\n")
            continue

        if trimmed.startswith(FENCE):
            opening_fence = _fence_token(trimmed)
            buf.append(line + "\n")
            continue

        level = _heading_level(trimmed)
        if level > 0:
            # 遇新标题：先把之前累积的内容落块，且此时标题栈还是"上一节"的
            index = _flush(chunks, buf, stack, root, current_level, index)
            buf = []
            _update_stack(stack, level, _heading_text(trimmed, level))
            current_level = level
            # 标题本身进正文，让 embedding 能感知这一段的主题
            buf.append(trimmed + "\n")
            continue

        buf.append(line + "\n")

    _flush(chunks, buf, stack, root, current_level, index)
    return chunks


def _flush(chunks: list[Chunk], buf: list[str], stack: list[str], root: str,
           level: int, index: int) -> int:
    """把缓冲区落成 1~n 个块，返回下一个可用的 chunk_index。"""
    text = "".join(buf).strip()
    if not text:
        return index

    heading_path = _build_heading_path(root, stack)
    parts = [text] if len(text) <= MAX_SIZE else split_long_text(text)

    for part in parts:
        trimmed = part.strip()
        if not trimmed:
            continue
        chunks.append(
            Chunk(
                chunk_index=index,
                chunk_type=_detect_type(trimmed),
                heading_path=heading_path,
                heading_level=max(level, 1),
                content=trimmed,
                content_tokens=estimate_tokens(trimmed),
            )
        )
        index += 1
    return index


def split_long_text(text: str) -> list[str]:
    """把超长文本切成多个块：先聚单元，超上界的单元单独硬切。"""
    blocks = _to_blocks(text)
    result: list[str] = []
    cur: list[str] = []
    cur_len = 0

    def flush_cur() -> None:
        nonlocal cur, cur_len
        if not cur:
            return
        joined = "".join(cur).strip()
        cur = []
        cur_len = 0
        if joined:
            result.append(joined)

    for block in blocks:
        limit = _hard_limit_of(block)

        # 单元自身就超上界 —— 先给前面收口，再单独硬切它
        if len(block) > limit:
            flush_cur()
            result.extend(_hard_split(block, limit))
            continue

        # 塞不下且当前块已够大 —— 收口开新块；长度把分隔符算进去
        if cur_len + len(block) + len(BLOCK_SEPARATOR) > TARGET_SIZE and cur_len >= MIN_SIZE:
            flush_cur()

        cur.append(block + BLOCK_SEPARATOR)
        cur_len += len(block) + len(BLOCK_SEPARATOR)

    flush_cur()
    return result


def _to_blocks(text: str) -> list[str]:
    """行级扫描：围栏代码块整体是一个不可拆的单元，其余按空行分段。"""
    blocks: list[str] = []
    block: list[str] = []
    opening: str | None = None

    for line in text.split("\n"):
        stripped = line.strip()

        if opening is not None:
            if _is_closing_fence(stripped, opening):
                block.append(line + "\n")
                _flush_buffer(blocks, block)
                opening = None
            else:
                # 代码块内容原样保留，含空行和 "```java" 这种带 info string 的行
                block.append(line + "\n")
            continue

        if stripped.startswith(FENCE):
            _flush_buffer(blocks, block)
            block.append(line + "\n")
            opening = _fence_token(stripped)
            continue

        if not stripped:
            _flush_buffer(blocks, block)
            continue

        block.append(line + "\n")

    # 未闭合的围栏：剩余内容整体作为一个单元
    _flush_buffer(blocks, block)
    return blocks


def _fence_token(stripped: str) -> str:
    """取行首连续反引号，如 "```java" → "```"。"""
    n = 0
    while n < len(stripped) and stripped[n] == "`":
        n += 1
    return stripped[:n]


def _is_closing_fence(stripped: str, opening: str) -> bool:
    """整行只有反引号且不短于开启围栏才算关闭。

    这样代码块内部的 "```java" 不会被误判成关闭，把代码从中间截断。
    """
    if len(stripped) < len(opening):
        return False
    return all(ch == "`" for ch in stripped)


def _hard_limit_of(block: str) -> int:
    return CODE_HARD_MAX_SIZE if block.lstrip().startswith(FENCE) else HARD_MAX_SIZE


def _hard_split(block: str, limit: int) -> list[str]:
    """兜底硬切：先按行，单行仍超限再按句子，单句仍超限再按字符。"""
    pieces: list[str] = []
    cur: list[str] = []
    cur_len = 0

    for line in block.split("\n"):
        if len(line) > limit:
            _flush_buffer(pieces, cur)
            cur_len = 0
            pieces.extend(_split_long_line(line, limit))
            continue
        if cur_len + len(line) + 1 > limit:
            _flush_buffer(pieces, cur)
            cur_len = 0
        cur.append(line + "\n")
        cur_len += len(line) + 1

    _flush_buffer(pieces, cur)
    return pieces


def _split_long_line(line: str, limit: int) -> list[str]:
    """超长单行：先按中英文句末标点切，仍超限的再按字符数切。"""
    result: list[str] = []
    cur: list[str] = []
    cur_len = 0

    for sentence in _SENTENCE_END.split(line):
        if not sentence.strip():
            continue
        if len(sentence) > limit:
            _flush_buffer(result, cur)
            cur_len = 0
            _append_by_chars(result, sentence, limit)
            continue
        if cur_len + len(sentence) > limit:
            _flush_buffer(result, cur)
            cur_len = 0
        cur.append(sentence)
        cur_len += len(sentence)

    _flush_buffer(result, cur)
    return result


def _append_by_chars(target: list[str], sentence: str, limit: int) -> None:
    """按字符数硬切。Python 的切片以码点为单位，不需要 Java 那套代理对保护。"""
    start = 0
    while start < len(sentence):
        end = min(start + limit, len(sentence))
        piece = sentence[start:end].strip()
        if piece:
            target.append(piece)
        start = end


def _flush_buffer(target: list[str], buf: list[str]) -> None:
    """把缓冲区收成一个元素并清空；纯空白直接丢弃（原地清空，调用方共享同一个 list）。"""
    if not buf:
        return
    joined = "".join(buf).strip()
    buf.clear()
    if joined:
        target.append(joined)


def _update_stack(stack: list[str], level: int, text: str) -> None:
    while stack and _level_of(stack[-1]) >= level:
        stack.pop()
    stack.append(f"{level}:{text}")


def _build_heading_path(root: str, stack: list[str]) -> str:
    parts: list[str] = []
    if root and root.strip():
        parts.append(root)
    for entry in stack:
        parts.append(entry.split(":", 1)[1] if ":" in entry else "")
    return " > ".join(parts)


def _level_of(entry: str) -> int:
    idx = entry.find(":")
    return int(entry[:idx]) if idx > 0 else 0


def _heading_level(line: str) -> int:
    """# 开头的标题，要求 # 后必须有一个空格，且层级不超过 6。"""
    if not line.startswith("#"):
        return 0
    level = 0
    while level < len(line) and line[level] == "#":
        level += 1
    if level > 6 or level >= len(line) or line[level] != " ":
        return 0
    return level


def _heading_text(line: str, level: int) -> str:
    return line[level:].strip()


def _detect_type(text: str) -> str:
    trimmed = text.strip()
    if trimmed.startswith(FENCE) and trimmed.endswith(FENCE):
        return "CODE"
    if trimmed.startswith("|") or "\n|" in trimmed:
        return "TABLE"
    if trimmed.startswith("- ") or trimmed.startswith("* ") or trimmed.startswith("1. "):
        return "LIST"
    return "TEXT"
