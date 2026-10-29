"""FastAPI 入口。

对应 Java：DevknowApplication + 控制器的装配。

演示页与接口同源托管：起一个进程，页面和 API 一起能用，不依赖外部目录。
静态资源是项目内的 static/（与 Java 版页面同源，改动需两边同步）。
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api import chat, conversations, stream
from app.db import close_pool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

# 项目内静态目录：相对 main.py 定位，整个文件夹拷走也能跑
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
DEMO_PAGE = "/agent-demo.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动不写库：语料由 app.ingest 离线同步，服务启动只读
    logger.info("devknow-py 启动")
    yield
    close_pool()
    logger.info("连接池已关闭")


def create_app() -> FastAPI:
    app = FastAPI(title="devknow-py", version="0.1.0", lifespan=lifespan)

    # 学习项目不做鉴权；生产环境这里要加认证与限流
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(chat.router, prefix="/api/agent", tags=["agent"])
    app.include_router(stream.router, prefix="/api/agent", tags=["agent"])
    app.include_router(conversations.router, prefix="/api/agent", tags=["conversation"])

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "UP"}

    @app.get("/")
    def index() -> RedirectResponse:
        return RedirectResponse(DEMO_PAGE)

    # 必须最后挂：mount 在 "/" 会吞掉所有未匹配路径
    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
        logger.info("演示页已挂载: %s", STATIC_DIR)
    else:
        logger.error("缺少静态目录，演示页不可用: %s", STATIC_DIR)

    return app


app = create_app()
