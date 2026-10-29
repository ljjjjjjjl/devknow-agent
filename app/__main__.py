"""一键启动：python -m app

页面与接口同一个进程同一个端口，所以只起这一条就够了。
等价于 uvicorn app.main:app。
"""
import os

import uvicorn


def main() -> None:
    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT", "8000"))
    uvicorn.run("app.main:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
