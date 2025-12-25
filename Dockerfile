# 使用官方的、与Smithery一致的Python基础镜像
FROM python:3.12-slim-bookworm

# 关键修复：安装C++编译器和相关的构建工具
# '&& rm -rf /var/lib/apt/lists/*' 用于保持镜像小巧
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 为了更好地利用Docker的缓存机制，先复制依赖管理文件
COPY pyproject.toml uv.lock* ./

# 安装uv并同步Python依赖
# 'pip install uv' 确保uv在镜像中可用
# 'uv sync' 会根据uv.lock或pyproject.toml安装所有包
RUN pip install uv && uv sync --no-dev

# 复制您项目的源代码
COPY ./src ./src

# 暴露端口，Smithery会使用它
EXPOSE 8080

# 最终启动服务器的命令
# 它会使用uv在虚拟环境中，通过smithery的CLI来运行您在pyproject.toml中指定的服务器入口
CMD ["uv", "run", "python", "-m", "smithery", "run", "biai_server.server:create_server"]