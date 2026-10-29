"""采集侧配置。

和 `app.config.Settings` 分开定义、共用同一个 `.env`：
在线服务不需要 GITHUB_TOKEN，采集器也不需要 SiliconFlow 的聊天配置。
合成一个类的话，任何一侧缺配置都会让另一侧起不来。

字段名大写后即为环境变量名（pydantic-settings 的隐式映射）。
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # —— GitHub ——
    github_token: str = ""
    github_base_url: str = "https://api.github.com"
    codeload_base_url: str = "https://codeload.github.com"

    # —— 采集目标（命令行可覆盖）——
    repo_owner: str = "spring-projects"
    repo_name: str = "spring-boot"
    repo_branch: str = "main"
    # 逗号分隔的 glob，相对仓库根目录；为空时用 knowledge_source.doc_path_pattern
    doc_pattern: str = ""

    # —— 采集行为 ——
    per_page: int = 100
    max_pages: int = 100
    request_interval: float = 0.2      # 两次请求之间的最小间隔（秒）
    max_retries: int = 3
    max_content_chars: int = 100_000   # 单条正文上限，防止撑爆单行 INSERT

    # —— 本地工作目录（下载的 tarball 解压到这里）——
    work_dir: str = "data/ingest"

    # —— 向量化 ——
    embed_batch_size: int = 32

    @property
    def doc_patterns(self) -> list[str]:
        return [p.strip() for p in self.doc_pattern.split(",") if p.strip()]


@lru_cache
def get_ingest_settings() -> IngestSettings:
    return IngestSettings()
