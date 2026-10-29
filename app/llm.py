"""模型与嵌入：SiliconFlow（OpenAI 兼容协议）。

对应 Java：ChatClient + SiliconFlowEmbeddingService。

两个关键点：
- embedding 必须和库里已有向量同模型（bge-m3）、同维度（1024），否则检索全是噪声
- OpenAIEmbeddings 默认会把文本切成 token 数组再发（check_embedding_ctx_length=True），
  SiliconFlow 不认这种格式，必须关掉，改成发原始字符串
"""
from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import get_settings


def _require_key() -> str:
    key = get_settings().siliconflow_api_key
    if not key:
        raise RuntimeError("缺少 SILICONFLOW_API_KEY：设环境变量或写进 .env")
    return key


@lru_cache
def get_chat_model(temperature: float = 0.2) -> ChatOpenAI:
    """agent 用的聊天模型。温度默认 0.2，与 Java 版一致。"""
    s = get_settings()
    return ChatOpenAI(
        model=s.chat_model,
        base_url=s.siliconflow_base_url,
        api_key=_require_key(),
        temperature=temperature,
    )


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    """bge-m3，输出 1024 维。check_embedding_ctx_length 必须关。"""
    s = get_settings()
    return OpenAIEmbeddings(
        model=s.embedding_model,
        base_url=s.siliconflow_base_url,
        api_key=_require_key(),
        check_embedding_ctx_length=False,
    )