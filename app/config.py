"""配置：环境变量 + 全项目共用常量。

依赖方向：db → config → 第三方库。config 不 import 任何 app 内模块。
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

CONTEXT_TOKEN_BUDGET = 4000
CONTEXT_MAX_MESSAGES = 10
RRF_C = 60
CANDIDATE_WINDOW = 50   # 每一路取多少候选参与融合；rerank 用两路全并集，不再裁这个数
RERANK_MAX_DOC_CHARS = 800
MIN_RELEVANCE = 0.4
TOP_K = 5
EMBEDDING_DIM = 1024


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    db_host: str = "127.0.0.1"
    db_port: int = 5432
    db_name: str = "devknow_utf8"
    db_user: str = "devknow"
    db_password: str = ""

    pool_min_size: int = 1
    pool_max_size: int = 5
    pool_timeout: float = 5.0

    siliconflow_api_key: str = ""
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    chat_model: str = "deepseek-ai/DeepSeek-V3"
    embedding_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_enabled: bool = True

    @property
    def conninfo(self) -> str:
        parts = [
            f"host={self.db_host}",
            f"port={self.db_port}",
            f"dbname={self.db_name}",
            f"user={self.db_user}",
        ]
        if self.db_password:
            parts.append(f"password={self.db_password}")
        return " ".join(parts)


@lru_cache
def get_settings() -> Settings:
    return Settings()