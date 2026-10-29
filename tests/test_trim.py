"""裁剪的边界测试。

对应 Java：ConversationServiceTest 的 trim 用例。

内容用中文："中" * 20 正好 20 token，断言里能直接数。
"""
from app.conversation.trim import estimate_tokens, trim


def chinese(n: int) -> str:
    return "中" * n


def user(content: str) -> dict:
    return {"role": "user", "content": content}


def assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}


def test_estimate_tokens():
    assert estimate_tokens(None) == 0
    assert estimate_tokens("") == 0
    assert estimate_tokens("中文") == 2          # 1 字 1 token
    assert estimate_tokens("abcd") == 1          # 4 字符 1 token
    assert estimate_tokens("abc") == 0           # 向下取整
    assert estimate_tokens("中abcd") == 2        # 1 + 4//4
    assert estimate_tokens(chinese(20)) == 20


def test_keeps_all_within_budget():
    history = [user("问题一"), assistant("回答一"), user("问题二"), assistant("回答二")]
    out = trim(history, 4000, 10)
    assert [m["role"] for m in out] == ["user", "assistant", "user", "assistant"]
    assert [m["content"] for m in out] == ["问题一", "回答一", "问题二", "回答二"]


def test_drops_oldest_over_budget():
    history = [user(chinese(20)), assistant(chinese(20)), user(chinese(20)), assistant(chinese(20))]
    out = trim(history, 50, 10)   # 最新 a2+u2=40 放得下，再加 a1 到 60 停
    assert [m["role"] for m in out] == ["user", "assistant"]
    assert len(out) == 2


def test_respects_message_count_limit():
    history = [user("问题一"), assistant("回答一"), user("问题二"), assistant("回答二")]
    out = trim(history, 4000, 2)
    assert [m["content"] for m in out] == ["问题二", "回答二"]


def test_window_starts_with_user():
    history = [user(chinese(20)), assistant(chinese(20)), user(chinese(20)), assistant(chinese(20))]
    out = trim(history, 25, 10)   # 只放得下最后一条 a2，是"没有提问的回答"，剥掉
    assert out == []


def test_empty_history():
    assert trim([], 4000, 10) == []


def test_single_oversized_dropped():
    history = [user(chinese(100)), assistant(chinese(100))]
    assert trim(history, 50, 10) == []