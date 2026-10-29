"""摘要压缩的纯逻辑测试：选段、分批、清洗、校验、截断、提示词。"""
from app.conversation.summary import (
    build_prompt, batch, clamp, clamp_by_lines, clean_markdown,
    is_no_content, is_well_formed, select_to_compress,
)


def chinese(n: int) -> str:
    return "中" * n


def msg(seq: int, role: str, content: str) -> dict:
    return {"seq": seq, "role": role, "content": content}


def test_select_only_dropped_and_unsummarized():
    history = [msg(1, "user", "q1"), msg(2, "assistant", "a1"),
               msg(3, "user", "q2"), msg(4, "assistant", "a2"),
               msg(5, "user", "q3"), msg(6, "assistant", "a3")]
    picked = select_to_compress(history, 5, 1, 1)
    assert [m["seq"] for m in picked] == [2, 3, 4]


def test_select_skips_below_min_messages():
    history = [msg(1, "user", "q1"), msg(2, "assistant", "a1"),
               msg(3, "user", "q2"), msg(4, "assistant", "a2"),
               msg(5, "user", "q3"), msg(6, "assistant", "a3")]
    assert select_to_compress(history, 5, 1, 4) == []


def test_select_everything_when_window_empty():
    history = [msg(1, "user", "q1"), msg(2, "assistant", "a1")]
    picked = select_to_compress(history, float("inf"), 0, 1)
    assert [m["seq"] for m in picked] == [1, 2]


def test_batch_skips_oversized_single():
    messages = [msg(1, "user", chinese(5)), msg(2, "assistant", chinese(50)), msg(3, "user", chinese(5))]
    batches = batch(messages, 20, 20)
    assert len(batches) == 1
    assert [m["seq"] for m in batches[0]] == [1, 3]


def test_batch_respects_count_and_token_limits():
    messages = [msg(1, "user", chinese(6)), msg(2, "assistant", chinese(6)),
                msg(3, "user", chinese(6)), msg(4, "assistant", chinese(6))]
    assert [len(b) for b in batch(messages, 1000, 2)] == [2, 2]
    assert [len(b) for b in batch(messages, 15, 20)] == [2, 2]


def test_clean_markdown_keeps_identifiers():
    raw = "**目标：** 配置多数据源\n- 已定：阈值 min-relevance-score=0.4，见 issue #49659\n* 待续：确认 *.java 通配符的写法\n---\n"
    text = clean_markdown(raw)
    assert "min-relevance-score=0.4" in text
    assert "#49659" in text
    assert "*.java" in text
    assert "**" not in text
    assert "---" not in text
    assert text.startswith("目标：")


def test_well_formed_requires_all_three():
    assert is_well_formed("目标：A\n已定：B\n待续：C") is True
    assert is_well_formed("目标：A\n已定：B") is False
    assert is_well_formed("A\nB\nC") is False
    assert is_well_formed(None) is False


def test_no_content_detection():
    assert is_no_content("无可提炼内容") is True
    assert is_no_content("目标：A\n已定：B\n待续：C") is False


def test_clamp():
    assert clamp("短文本", 10) == "短文本"
    # Python 按码点切片：emoji 整体保留，不回退丢弃（与 Java 的 UTF-16 行为不同）
    assert clamp("中中中🚀中中中", 4) == "中中中🚀…"


def test_clamp_by_lines_keeps_sections():
    text = "目标：" + chinese(50) + "\n已定：" + chinese(50) + "\n待续：" + chinese(50)
    clamped = clamp_by_lines(text, 20)
    lines = clamped.split("\n")
    assert len(lines) == 3
    assert all("目标：" in l or "已定：" in l or "待续：" in l for l in lines)
    assert all(len(l) <= 21 for l in lines)


def test_prompt_carries_old_summary_and_transcript():
    b = [msg(1, "user", "怎么配置多数据源"), msg(2, "assistant", "用 @Primary")]
    empty = build_prompt(None, b)
    assert "（无）" in empty
    assert "用户：怎么配置多数据源" in empty
    assert "助手：用 @Primary" in empty
    assert "无可提炼内容" in empty
    assert "{old_summary}" not in empty and "{new_turns}" not in empty

    with_old = build_prompt("目标：配置多数据源", b)
    assert "目标：配置多数据源" in with_old