#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP 服务器：将数据库知识库查询功能暴露为 Cursor 可调用的工具。

启动示例：
    python knowledge_base_mcp_server.py
"""

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from mcp.server import InitializationOptions, NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.types import ServerCapabilities, TextContent, Tool

load_dotenv()

# 基础配置，可通过环境变量覆盖
OPENAI_API_KEY = os.getenv("openrouter_api_key") or os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("url_openrouter") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
PERSIST_DIRECTORY = Path(
    os.getenv("KB_PERSIST_DIR", Path(__file__).resolve().parent / "chroma_db")
)
COLLECTION_NAME = os.getenv("KB_COLLECTION_NAME", "database_knowledge")
EMBED_MODEL = os.getenv("KB_EMBED_MODEL", "text-embedding-3-large")

if not OPENAI_API_KEY:
    raise RuntimeError("未设置 OPENAI_API_KEY 或 openrouter_api_key")


def _build_vectorstore() -> Chroma:
    """加载 Chroma 向量库"""
    embedding_kwargs: Dict[str, Any] = {
        "model": EMBED_MODEL,
        "api_key": OPENAI_API_KEY,
    }
    if "openrouter" in OPENAI_BASE_URL.lower():
        embedding_kwargs["base_url"] = OPENAI_BASE_URL

    embeddings = OpenAIEmbeddings(**embedding_kwargs)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIRECTORY),
    )
    return vectorstore


VECTORSTORE = _build_vectorstore()
SERVER = Server("knowledge-base-mcp")


def _format_doc(doc: Any, index: int) -> str:
    """将 Chroma Document 转为可读文本"""
    meta = doc.metadata or {}
    doc_type = meta.get("type", "unknown")

    if doc_type == "business_query":
        title = f"查询 {meta.get('query_id', '')} - {meta.get('query_name', '')}".strip()
    elif doc_type == "table_schema":
        title = f"表结构 {meta.get('table_name', '')} - {meta.get('table_comment', '')}".strip()
    else:
        title = meta.get("source", "unknown")

    return f"【结果 {index}】{title}\n{doc.page_content}"


def _search(query: str, scope: str, k: int) -> str:
    """执行检索并格式化结果"""
    filter_arg: Optional[Dict[str, str]] = None
    if scope == "business":
        filter_arg = {"type": "business_query"}
    elif scope == "schema":
        filter_arg = {"type": "table_schema"}

    try:
        docs = VECTORSTORE.similarity_search(query, k=k, filter=filter_arg)
    except Exception as exc:  # 兜底捕获以返回给 MCP
        return f"检索失败: {exc}"

    if not docs:
        return "未找到相关内容。"

    parts = [_format_doc(doc, i + 1) for i, doc in enumerate(docs)]
    return "\n\n".join(parts)


@SERVER.list_tools()
async def list_tools() -> List[Tool]:
    """MCP: 列出可用工具"""
    return [
        Tool(
            name="kb_search",
            description="在数据库知识库中检索业务查询与表结构。"
            "scope 取值: business(仅业务查询)/schema(仅表结构)/all(默认)。"
            "k 为返回数量，1-10。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索问题或关键词"},
                    "scope": {
                        "type": "string",
                        "enum": ["business", "schema", "all"],
                        "default": "all",
                    },
                    "k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        )
    ]


@SERVER.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """MCP: 执行检索工具"""
    if name != "kb_search":
        raise ValueError(f"未知工具: {name}")

    query = str(arguments.get("query", "")).strip()
    if not query:
        return [TextContent(type="text", text="query 不能为空")]

    scope = str(arguments.get("scope", "all")).strip().lower()
    if scope not in {"business", "schema", "all"}:
        scope = "all"

    try:
        k = int(arguments.get("k", 3))
    except Exception:
        k = 3
    k = max(1, min(10, k))

    result_text = _search(query=query, scope=scope, k=k)
    return [TextContent(type="text", text=result_text)]


async def main() -> None:
    """入口：运行 MCP 服务器（stdio）"""
    async with stdio_server() as (read_stream, write_stream):
        init_options = SERVER.create_initialization_options(
            notification_options=NotificationOptions(),
            experimental_capabilities={},
        )
        await SERVER.run(
            read_stream,
            write_stream,
            initialization_options=init_options,  # 可按需传递客户端初始化参数
        )


if __name__ == "__main__":
    asyncio.run(main())

