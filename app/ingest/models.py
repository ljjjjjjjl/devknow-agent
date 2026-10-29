"""采集链路上的数据结构，以及内容指纹。

三个类型分别对应三个阶段的产物：
    RawDoc  —— 从 GitHub 拿到的原始内容（尚未切分）
    Chunk   —— 切分产物（检索与向量化的基本单位）
    SyncStats —— 一次同步的计数，命令行最后打印

指纹算法必须和 Java 版逐字节一致（`common/util/HashUtils`）：
    sha256( UTF-8( part0 + "\u0000" + part1 + ... ) )
用 \u0000 分隔是为了让 ("ab","c") 和 ("a","bc") 不撞车。
"""
import hashlib
from dataclasses import dataclass, field
from datetime import datetime


def sha256_hex(*parts: str | None) -> str:
    """多段拼接后取 sha256 十六进制（64 字符，对应库里的 CHAR(64)）。"""
    joined = "\u0000".join("" if p is None else p for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RawDoc:
    """一条待入库的原始文档。"""

    doc_type: str            # DOC | ISSUE | PR
    external_id: str         # issue 编号 / 仓库内相对路径
    title: str
    body: str                # 已截断的正文
    url: str | None = None
    version: str | None = None       # issue 取 milestone 标题；文档取分支名
    version_order: int | None = None
    path: str | None = None
    lang: str = "en"
    published_at: datetime | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """增量判断依据：标题 + 正文。"""
        return sha256_hex(self.title, self.body)


@dataclass(frozen=True)
class Chunk:
    """切分产物。字段与 document_chunk 表一一对应。"""

    chunk_index: int
    chunk_type: str          # TEXT | CODE | TABLE | LIST
    heading_path: str
    heading_level: int
    content: str
    content_tokens: int

    @property
    def content_hash(self) -> str:
        """跨文档去重用：同一段样例代码可能被多处复制。"""
        return sha256_hex(self.content)


@dataclass
class SyncStats:
    """一次同步的计数。"""

    seen: int = 0
    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    chunks: int = 0

    def merge(self, other: "SyncStats") -> None:
        self.seen += other.seen
        self.inserted += other.inserted
        self.updated += other.updated
        self.skipped += other.skipped
        self.chunks += other.chunks

    def __str__(self) -> str:
        return (
            f"看到 {self.seen} 篇 | 新增 {self.inserted} | 更新 {self.updated} | "
            f"跳过 {self.skipped} | 产出 chunk {self.chunks}"
        )
