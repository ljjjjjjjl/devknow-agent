"""上下文摘要压缩：把被窗口丢弃的早期对话压成三段式摘要。

对应 Java：ConversationSummarizer + SummaryText。

三条红线：
1. 只压窗口外的：与原文窗口永不重叠，否则同一句话模型看到两遍
2. 水位线单调推进：同一段只压一次，靠 summary_upto_seq + CAS 保证
3. 失败绝不阻塞主链路：任何异常返回原摘要，调用方退回"直接丢"
"""
import logging
import re

from langchain_core.messages import HumanMessage

from app.conversation.store import get_summary, update_summary_cas
from app.conversation.trim import estimate_tokens
from app.llm import get_chat_model

logger = logging.getLogger(__name__)

# —— 配置（对应 Java SummaryProperties）——
SUMMARY_ENABLED = True
SUMMARY_TEMPERATURE = 0.1
SUMMARY_MAX_TOKENS = 400
MIN_MESSAGES = 4
BATCH_TOKEN_LIMIT = 2000
BATCH_MAX_MESSAGES = 20
MAX_ATTEMPTS = 2
SUMMARY_LINE_CHARS = 110   # 注入时每行截断长度（3 行 × 110 ≈ 330 token）

SUMMARY_PREFIX = (
    "【更早对话的摘要】以下内容仅用于理解指代与上下文，不得作为事实来源；"
    "涉及事实的回答必须来自本轮检索结果。\n"
)

NO_CONTENT = "无可提炼内容"
PREFIX_GOAL = "目标："
PREFIX_DONE = "已定："
PREFIX_TODO = "待续："

PROMPT_TEMPLATE = """你是对话摘要器。输入是一段更早的对话记录，以及上一次的摘要（可能为空）。
任务：把两者合并去重，压缩成一段不超过 300 中文字的纯文本记忆，供后续轮次理解上下文与指代。

输出格式（严格输出三行纯文本，禁止任何 Markdown 格式如加粗/列表，依次以下列词开头）：
目标：用户在做什么、想达成什么
已定：已经确认的结论、参数、约束、选型
待续：尚未解决的问题、被搁置的分支、未完成项

规则（必须严格遵守）：
1. 只保留对话里明确出现过的信息，禁止推测；老摘要与新对话冲突时，以新对话为准。
2. 不要写建议、不要回答问题、不要复述代码或日志原文；只抽象描述"做了 X"。
3. 被推翻或被修正的中间结论不要写，只写最终结论。
4. 具体标识（类名、配置项、文件名、报错关键字）必须原样保留，它们是后续指代的锚点。
5. 某一节没有内容时该行仍然保留，冒号后写「无」。
6. 若这次记录里没有可沉淀的信息（寒暄、重复提问），只输出「无可提炼内容」这一行，不输出三行结构。
7. 不要任何开场白或结尾说明，第一行必须直接以「目标：」开头。

上一次摘要：
{old_summary}

待压缩的对话记录：
{new_turns}

现在输出合并后的摘要正文：
"""


# —— 纯逻辑（对应 SummaryText）——

def select_to_compress(history: list[dict], window_min_seq: int, upto_seq: int,
                       min_messages: int) -> list[dict]:
    """挑出"可压"的那段：已被窗口丢掉（seq < window_min_seq）且没压过（seq > upto_seq）。"""
    picked = [
        m for m in history
        if m.get("seq") is not None and m["seq"] < window_min_seq and m["seq"] > upto_seq
    ]
    return picked if len(picked) >= min_messages else []


def batch(messages: list[dict], token_limit: int, max_count: int) -> list[list[dict]]:
    """按 token 预算与条数上限切批；单条自身超预算的整条跳过。"""
    batches: list[list[dict]] = []
    current: list[dict] = []
    used = 0
    for m in messages:
        cost = estimate_tokens(m.get("content"))
        if cost > token_limit:
            continue   # 单条超限：跳过，不切半条
        if current and (len(current) >= max_count or used + cost > token_limit):
            batches.append(current)
            current = []
            used = 0
        current.append(m)
        used += cost
    if current:
        batches.append(current)
    return batches


def build_prompt(old_summary: str | None, batch: list[dict]) -> str:
    previous = "（无）" if not old_summary else old_summary.strip()
    return (PROMPT_TEMPLATE
            .replace("{old_summary}", previous)
            .replace("{new_turns}", render_transcript(batch)))


def render_transcript(messages: list[dict]) -> str:
    return "".join(
        ("用户：" if m["role"] == "user" else "助手：")
        + (m["content"] or "（空）").strip() + "\n"
        for m in messages
    )


def clean_markdown(raw: str | None) -> str:
    """只删结构性 Markdown，不碰可能属于标识符的 - # * . 字符。"""
    if raw is None:
        return ""
    text = raw.strip()
    text = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)      # 行首标题
    text = re.sub(r"(?m)^\s*[-*+]\s+", "", text)       # 行首列表符号
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)       # 成对加粗
    text = re.sub(r"(?m)^\s*[-=]{3,}\s*$", "", text)   # 分隔线
    return text.strip()


def is_well_formed(text: str | None) -> bool:
    return bool(text) and text.startswith(PREFIX_GOAL) \
        and PREFIX_DONE in text and PREFIX_TODO in text


def is_no_content(text: str | None) -> bool:
    return bool(text) and NO_CONTENT in text


def clamp(text: str | None, max_chars: int) -> str | None:
    """按字符截断补省略号。Python 按码点切片，emoji 不会切半个。"""
    if text is None or len(text) <= max_chars:
        return text
    cut = max(1, max_chars)
    return text[:cut].rstrip() + "…"


def clamp_by_lines(text: str | None, per_line: int) -> str | None:
    """按行独立截断，避免整节丢掉「待续」。"""
    if text is None:
        return None
    return "\n".join(clamp(line, per_line) for line in text.split("\n"))


# —— 编排（对应 ConversationSummarizer）——

def maybe_summarize(conversation_id: int, history: list[dict], kept_count: int) -> str | None:
    """视情况压缩，返回可注入的摘要；None = 没摘要或失败，调用方按"直接丢"处理。"""
    if not SUMMARY_ENABLED or conversation_id is None or not history:
        return None
    row = get_summary(conversation_id)
    if row is None:
        return None
    upto = row["upto"] or 0
    summary = row["summary"]

    window_min_seq = float("inf") if kept_count <= 0 else history[len(history) - kept_count]["seq"]

    pending = select_to_compress(history, window_min_seq, upto, MIN_MESSAGES)
    if not pending:
        return summary

    for b in batch(pending, BATCH_TOKEN_LIMIT, BATCH_MAX_MESSAGES):
        merged = summarize(summary, b)
        if merged is None:
            return summary
        new_upto = b[-1]["seq"]
        if not update_summary_cas(conversation_id, merged, new_upto, upto):
            logger.warning("摘要水位已被并发推进，丢弃本次结果: conversationId=%s expectedUpto=%s",
                           conversation_id, upto)
            return summary
        logger.info("上下文摘要已更新: conversationId=%s seq %s -> %s, 摘要 %s 字",
                    conversation_id, upto, new_upto, len(merged))
        summary = merged
        upto = new_upto
    return summary


def summarize(old_summary: str | None, batch: list[dict]) -> str | None:
    """压一批。返回新摘要；"" = 无可提炼（算成功，水位照样推进）；None = 失败（水位不动）。"""
    prompt = build_prompt(old_summary, batch)
    last_error = None
    for _ in range(MAX_ATTEMPTS):
        try:
            llm = get_chat_model(temperature=SUMMARY_TEMPERATURE).bind(max_tokens=SUMMARY_MAX_TOKENS)
            resp = llm.invoke([HumanMessage(content=prompt)])
            raw = resp.content
            if not raw or not raw.strip():
                last_error = "模型返回空摘要"
                continue
            text = clean_markdown(raw)
            if is_no_content(text):
                return ""
            if not is_well_formed(text):
                last_error = "摘要缺少三行结构"
                continue
            return text
        except Exception as e:
            last_error = e
    logger.warning("摘要压缩失败，退回直接丢弃: %s", last_error)
    return None