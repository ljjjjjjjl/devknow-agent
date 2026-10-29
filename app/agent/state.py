"""图的 State 定义。

对应 Java：AgentRunner 循环里的局部变量（messages / step / citations）。

要点：
- State 只在一次请求内有效，历史从 conversation_message 表读，
  别用 checkpointer 替代那张表
- messages 用 add_messages 归约（自动去重相邻重复）；steps 用
  operator.add 累加（初始 0）；其余字段是"后写覆盖"
"""
import operator
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """一次检索问答请求的状态。"""

    # 对话消息：plan/retrieve/answer 往里追加
    messages: Annotated[list[BaseMessage], add_messages]

    # 用户原始问题（进图时写入，节点只读）
    query: str

    # 本次检索命中的文档（retrieve 写入，answer 据此引用）
    hits: list[Document]

    # 引用列表（retrieve 从 hits 生成，供 SSE citation 事件 / 前端展示）
    citations: list[dict]

    # 已走的步骤数（模型调用次数，operator.add 累加，初始 0）
    steps: Annotated[int, operator.add]

    # 最高精排分；rerank 开启且有命中时为 float，否则为 None
    top_score: float | None

    # 能否信任"hits 为空 = 无相关"：rerank 开启 True，关闭 False
    can_refuse: bool