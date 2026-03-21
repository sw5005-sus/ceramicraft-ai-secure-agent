# --- 第一阶段：构建环境 ---
    FROM python:3.12-slim-bookworm AS builder

    # 安装 uv
    COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
    
    # 设置工作目录
    WORKDIR /app
    
    # 1. 缓存依赖安装（利用 Docker 层缓存）
    # 只有 pyproject.toml 或 uv.lock 变化时，才会重新下载依赖
    COPY pyproject.toml uv.lock README.md ./
    RUN --mount=type=cache,target=/root/.cache/uv \
        uv sync --frozen --no-install-project --no-dev
    
    # 2. 拷贝源码并正式安装项目
    COPY src/ ./src/
    RUN --mount=type=cache,target=/root/.cache/uv \
        uv sync --frozen --no-dev
    
    
    # --- 第二阶段：运行环境 ---
    FROM python:3.12-slim-bookworm
    
    # 设置环境变量
    ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1 \
        PATH="/app/.venv/bin:$PATH" \
        HOST=0.0.0.0 \
        PORT=8000
    
    WORKDIR /app
    
    # 1. 安全加固：创建非 root 用户
    RUN groupadd -r appgroup && useradd -r -g appgroup -s /sbin/nologin appuser
    
    # 2. 从构建阶段拷贝编译好的虚拟环境和源码
    # 这样最终镜像里不会包含 uv 工具，体积更小
    COPY --from=builder /app /app

    RUN chmod -R 555 /app && chown -R appuser:appgroup /app
    
    # 3. 切换到非 root 用户
    USER appuser
    
    # 暴露端口
    EXPOSE 8000
    
    # 启动命令：使用模块化方式启动 AI Secure Agent
    CMD ["python", "-m", "ceramicraft_ai_secure_agent.app"]