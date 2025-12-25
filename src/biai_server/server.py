#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smithery 标准入口：使用 @smithery.server() 装饰的工厂函数创建并返回 MCP 服务器实例。

入口函数：
    biai_server.server:create_server
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from smithery.decorators import smithery

from .enhanced_formatter import format_requirement_results, format_table_results

# FastMCP 导入兼容：
# - Smithery 推荐：mcp.server.fastmcp.FastMCP
# - 你旧项目里：fastmcp.FastMCP
try:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP  # type: ignore
except Exception:  # pragma: no cover
    from fastmcp import FastMCP  # type: ignore


logger = logging.getLogger(__name__)


def _tool_decorator(server: Any):
    """
    兼容不同 FastMCP 版本的 tool 注册方式：
    - 有的版本是 @server.tool()
    - 有的版本是 @server.tool
    """
    tool_attr = getattr(server, "tool")
    try:
        return tool_attr()
    except TypeError:
        return tool_attr


def _build_embeddings(openai_api_key: str, openai_base_url: str, embed_model: str) -> OpenAIEmbeddings:
    """构建 Embeddings 对象（支持 OpenRouter/OpenAI）"""
    embedding_kwargs: Dict[str, Any] = {
        "model": embed_model,
        "api_key": openai_api_key,
    }
    # OpenRouter/自定义网关场景需要 base_url
    if openai_base_url:
        embedding_kwargs["base_url"] = openai_base_url
    return OpenAIEmbeddings(**embedding_kwargs)


def _build_vectorstore(kb_name: str, kb_path: Path, embeddings: OpenAIEmbeddings) -> Optional[Chroma]:
    """加载指定的 Chroma 向量库；失败返回 None。"""
    try:
        if not kb_path.exists():
            logger.warning("知识库路径不存在: %s", kb_path)
            return None

        vectorstore = Chroma(
            collection_name=kb_name,
            embedding_function=embeddings,
            persist_directory=str(kb_path),
        )
        logger.info("成功加载知识库: %s from %s", kb_name, kb_path)
        return vectorstore
    except Exception as e:
        logger.error("加载知识库失败 %s: %s", kb_name, e, exc_info=True)
        return None


@smithery.server()
def create_server():
    """
    创建并返回 MCP Server 实例（Smithery 会调用这个函数）。

    环境变量（与旧版保持一致）：
    - openrouter_api_key 或 OPENAI_API_KEY
    - url_openrouter 或 OPENAI_BASE_URL
    - KB_PERSIST_DIR（可选，默认 ./chroma_db）
    - KB_EMBED_MODEL（可选，默认 openai/text-embedding-3-large）
    """
    # 日志：尽量不污染宿主，但保持可观测性
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    load_dotenv()

    openai_api_key = os.getenv("openrouter_api_key") or os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("url_openrouter") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    embed_model = os.getenv("KB_EMBED_MODEL", "openai/text-embedding-3-large")

    if not openai_api_key:
        raise RuntimeError("未设置 OPENAI_API_KEY 或 openrouter_api_key")

    # 关键修复：迁移到 src/ 后不要再用 __file__ 推断根目录
    chroma_db_dir = Path(os.getenv("KB_PERSIST_DIR", str(Path.cwd() / "chroma_db")))

    query_kb_name = "query_requirements_kb"
    table_kb_name = "meta_tables_kb"

    table_kb_path = chroma_db_dir / table_kb_name
    query_kb_path = chroma_db_dir / query_kb_name

    embeddings = _build_embeddings(openai_api_key, openai_base_url, embed_model)
    table_vectorstore = _build_vectorstore(table_kb_name, table_kb_path, embeddings)
    query_vectorstore = _build_vectorstore(query_kb_name, query_kb_path, embeddings)

    server = FastMCP(name="knowledge-base-mcp")
    tool = _tool_decorator(server)

    @tool
    def kb_search_tables(query: str, k: int = 5) -> str:
        """
        在数据表元数据知识库中检索表结构信息。

        Args:
            query: 查询文本，例如 "用户信息表" 或 "订单相关的表"
            k: 返回的结果数量，默认为 5，范围 1-20

        Returns:
            格式化的检索结果文本
        """
        k = max(1, min(20, k))
        logger.info("检索表结构: query=%s, k=%s", query, k)

        if table_vectorstore is None:
            error_msg = f"❌ 表结构知识库未初始化，请检查路径: {table_kb_path}"
            logger.error(error_msg)
            return error_msg

        try:
            results = table_vectorstore.similarity_search_with_score(query, k=k)
            if not results:
                return "未找到相关表结构信息。"
            return format_table_results(results, query)
        except Exception as exc:
            error_msg = f"❌ 表结构检索失败: {exc}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    @tool
    def kb_search_requirements(query: str, k: int = 5) -> str:
        """
        在业务需求查询知识库中检索历史查询需求。

        Args:
            query: 查询文本，例如 "放款金额统计" 或 "用户逾期分析"
            k: 返回的结果数量，默认为 5，范围 1-20

        Returns:
            格式化的检索结果文本
        """
        k = max(1, min(20, k))
        logger.info("检索业务需求: query=%s, k=%s", query, k)

        if query_vectorstore is None:
            error_msg = f"❌ 业务需求知识库未初始化，请检查路径: {query_kb_path}"
            logger.error(error_msg)
            return error_msg

        try:
            results = query_vectorstore.similarity_search_with_score(query, k=k)
            if not results:
                return "未找到相关业务需求。"
            return format_requirement_results(results, query)
        except Exception as exc:
            error_msg = f"❌ 业务需求检索失败: {exc}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    return server


