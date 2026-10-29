"""
这是一个给大模型（LLM）用的“查资料工具”。

简单来说：
用户问了一个问题 -> 大模型发现不懂 -> 大模型按下了这个“查资料按钮”（search_knowledge）
-> 代码去数据库里搜出相关文档 -> 把文档变成文字交给大模型 -> 大模型看着文字回答用户。
"""
import logging

from langchain_core.documents import Document
from langchain_core.tools import tool

# 从配置文件里导入一些常量（比如每次最多查几条、每段文字最多截取多少字）
from app.config import RERANK_MAX_DOC_CHARS, TOP_K
from app.retrieval.pipeline import RetrievalResult, retrieve

# 配置日志，用来记录程序运行时的报错信息（就像飞机的黑匣子）
logger = logging.getLogger(__name__)

# 定义两句话，如果没搜到东西或者报错了，就把这两句话返回给大模型
NO_RESULT_TEXT = "未检索到相关内容。"
SEARCH_FAIL_TEXT = "检索失败，请稍后重试。"


# ==========================================
# 工具函数 1：把太长的文字砍短
# ==========================================
def _truncate(text: str, max_chars: int) -> str:
    # 为什么要砍短？因为大模型是按“字数”收费的（Token）。
    # 如果把几万字的长文全塞给它，不仅贵，大模型还会看“晕”（超出上下文限制）。
    return text[:max_chars]


# ==========================================
# 工具函数 2：把搜出来的文档“排版”成大模型爱看的样子
# ==========================================
def _format_docs(docs: list[Document], max_chars: int) -> str:
    """把数据库里搜出来的零散文档，拼成一段漂亮的纯文本。"""
    blocks = []
    # 给每段资料加上序号 [1], [2]，大模型看的时候就不会把两篇文章搞混
    for i, d in enumerate(docs, 1):
        m = d.metadata  # 文档的附属信息（标题、链接等）
        title = m.get("title") or "(无标题)"
        url = m.get("url") or ""
        # 把序号、标题、链接、正文拼在一起，段落之间空两行
        blocks.append(f"[{i}] {title}\nURL: {url}\n{_truncate(d.page_content, max_chars)}")
    return "\n\n".join(blocks)


# ==========================================
# 工具函数 3：生成“引用来源”（给前端页面展示用的）
# ==========================================
def _build_citations(docs: list[Document]) -> list[dict]:
    """从搜出来的文档中，提取出干净的“引用列表”，方便前端画小卡片。"""
    seen: set = set()  # 准备一个空盒子，用来去重（防止同一篇文章弹出来两次）
    out: list[dict] = []
    
    for d in docs:
        m = d.metadata
        doc_id = m.get("document_id")
        
        # 如果没ID，或者这个ID已经记录过了，就跳过
        if doc_id is None or doc_id in seen:
            continue
        
        seen.add(doc_id) # 把见过的ID扔进盒子里
        
        # 注意这里的 "documentId" 是驼峰命名。因为前端和Java后端都是这么叫的。
        # 必须跟他们对齐，否则前端展示引用时会报错（找不到字段）。
        out.append({
            "documentId": doc_id,
            "title": m.get("title") or "",
            "url": m.get("url") or "",
        })
    return out


# ==========================================
# 核心执行逻辑：真正干活的函数
# ==========================================
def run_search(query: str, top_k: int = TOP_K) -> tuple[str, list[dict], RetrievalResult | None]:
    """
    真正去数据库搜东西的函数。
    返回三样东西：(给大模型看的文本, 给前端看的引用列表, 原始的检索结果)
    """
    try:
        # 防止大模型瞎传参数（比如传个 top_k=99999），强制限制在 1 到系统最大值之间
        top_k = max(1, min(int(top_k), TOP_K))
        
        # 调用底层检索模块去数据库查资料
        result = retrieve(query, top_k=top_k)
    except Exception:
        # 如果数据库挂了、网络断了，千万别让整个程序崩溃！
        # 记录下错误原因，然后温和地返回一句话，让大模型去安抚用户。
        logger.exception("run_search 失败: query=%r", query)
        return SEARCH_FAIL_TEXT, [], None

    # 如果啥也没搜到
    if not result.documents:
        return NO_RESULT_TEXT, [], result
    
    # 如果搜到了，就进行排版和生成引用
    text = _format_docs(result.documents, RERANK_MAX_DOC_CHARS)
    citations = _build_citations(result.documents)
    
    # 把三样东西都交出去（这样后端可以直接用 citations，不用再查一次数据库）
    return text, citations, result


# ==========================================
# 给大模型看的“包装盒”（这就是 Function Call 的入口）
# ==========================================
@tool
def search_knowledge(query: str, top_k: int = TOP_K) -> str:
    """
    检索研发知识库，返回与 query 最相关的文档片段。回答需要依据知识库的问题前，必须先调用本工具。
    
    【注】上面这三行字非常重要，大模型就是靠读这段话来决定什么时候按这个按钮的！
    """
    # 真正执行搜索，但只把“文本”拿出来（用 _ 把引用列表和原始结果丢掉）
    # 为什么丢掉？因为大模型只需要看文字去思考，它不需要看结构化的引用数据。
    text, _, _ = run_search(query, top_k)
    return text