"""上下文裁剪：4000 token + 10 条双上限。

对应 Java：ConversationService.trim + TokenEstimator（纯函数）。

要点：
- 从末尾往前取连续窗口，超出就停；不做跳跃挑选
- 取完剥掉开头的 assistant，保证窗口以用户消息开始
- token 估算口径必须和 Java 版一致：中文 1 字 1 token、其余 4 字符 1 token
  换 tiktoken 会让窗口大小和 Java 版不一致，评估数字就不可比
"""
from app.config import CONTEXT_MAX_MESSAGES, CONTEXT_TOKEN_BUDGET


def estimate_tokens(text: str | None) -> int:
    """中文（U+4E00..U+9FFF）1 字 1 token，其余 4 字符 1 token，向下取整。"""
    if not text:
        return 0
    cjk = sum(1 for c in text if 0x4E00 <= ord(c) <= 0x9FFF)
    return cjk + (len(text) - cjk) // 4


def trim(
    history: list[dict],
    token_budget: int = CONTEXT_TOKEN_BUDGET,
    max_messages: int = CONTEXT_MAX_MESSAGES,
) -> list[dict]:
    """从最近往前取连续窗口，超出 token 预算或条数上限就停。

    返回 [{"role", "content"}, ...]，按时间升序，且以 user 开头。
    """
    picked: list[dict] = []
    used = 0
    for m in reversed(history):
        if len(picked) >= max_messages:
            break
        cost = estimate_tokens(m.get("content"))
        if used + cost > token_budget:
            break
        picked.append({"role": m["role"], "content": m["content"]})
        used += cost

    picked.reverse()   # 从末尾往前取的，反转回时间升序
    while picked and picked[0]["role"] != "user":
        picked.pop(0)
    return picked