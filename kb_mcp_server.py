#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP 服务器：将数据库知识库查询功能暴露为 Cursor 可调用的工具。

启动示例：
    python knowledge_base_mcp_server.py
"""
from fastmcp import FastMCP
import asyncio
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 导入增强版格式化器
from enhanced_formatter import format_table_results, format_requirement_results

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# 基础配置，可通过环境变量覆盖
OPENAI_API_KEY = os.getenv("openrouter_api_key") or os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("url_openrouter") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
CHROMA_DB_DIR = Path(
    os.getenv("KB_PERSIST_DIR", Path(__file__).resolve().parent / "chroma_db")
)
# 两个独立的知识库
QUERY_KB_NAME = "query_requirements_kb"
TABLE_KB_NAME = "meta_tables_kb"
EMBED_MODEL = os.getenv("KB_EMBED_MODEL", "openai/text-embedding-3-large")

if not OPENAI_API_KEY:
    raise RuntimeError("未设置 OPENAI_API_KEY 或 openrouter_api_key")


mcp = FastMCP("knowledge-base-mcp")


@mcp.tool
def kb_search_tables(query: str, k: int = 5) -> str:
    """
    在数据表元数据知识库中检索表结构信息。
    
    Args:
        query: 查询文本，例如 "用户信息表" 或 "订单相关的表"
        k: 返回的结果数量，默认为 5，范围 1-20
        
    Returns:
        格式化的检索结果文本
        
    示例:
        kb_search_tables("用户信息表", 3)
    """
    # 参数验证
    k = max(1, min(20, k))
    logger.info(f"检索表结构: query={query}, k={k}")
    return _search_tables(query, k)


@mcp.tool
def kb_search_requirements(query: str, k: int = 5) -> str:
    """
    在业务需求查询知识库中检索历史查询需求。
    
    Args:
        query: 查询文本，例如 "放款金额统计" 或 "用户逾期分析"
        k: 返回的结果数量，默认为 5，范围 1-20
        
    Returns:
        格式化的检索结果文本
        
    示例:
        kb_search_requirements("放款金额统计", 3)
    """
    # 参数验证
    k = max(1, min(20, k))
    logger.info(f"检索业务需求: query={query}, k={k}")
    return _search_requirements(query, k)


def _build_embeddings() -> OpenAIEmbeddings:
    """构建 Embeddings 对象"""
    embedding_kwargs: Dict[str, Any] = {
        "model": EMBED_MODEL,
        "api_key": OPENAI_API_KEY,
    }
    if "openrouter" in OPENAI_BASE_URL.lower():
        embedding_kwargs["base_url"] = OPENAI_BASE_URL
    
    return OpenAIEmbeddings(**embedding_kwargs)


def _build_vectorstore(kb_name: str, kb_path: Path) -> Optional[Chroma]:
    """
    加载指定的 Chroma 向量库
    
    Args:
        kb_name: 知识库集合名称
        kb_path: 知识库持久化路径
        
    Returns:
        Chroma 实例，失败时返回 None
    """

    try:
        if not kb_path.exists():
            logger.warning(f"知识库路径不存在: {kb_path}")
            return None
            
        embeddings = _build_embeddings()
        vectorstore = Chroma(
            collection_name=kb_name,
            embedding_function=embeddings,
            persist_directory=str(kb_path),
        )
        logger.info(f"成功加载知识库: {kb_name} from {kb_path}")
        return vectorstore
    except Exception as e:
        logger.error(f"加载知识库失败 {kb_name}: {e}")
        return None


# 初始化两个独立的知识库
TABLE_KB_PATH = CHROMA_DB_DIR / TABLE_KB_NAME
QUERY_KB_PATH = CHROMA_DB_DIR / QUERY_KB_NAME

TABLE_VECTORSTORE = _build_vectorstore(TABLE_KB_NAME, TABLE_KB_PATH)
QUERY_VECTORSTORE = _build_vectorstore(QUERY_KB_NAME, QUERY_KB_PATH)

# 旧的格式化函数已被移除，现在使用 enhanced_formatter 模块中的函数
# format_table_results() 和 format_requirement_results()
# 新格式化器返回完整的结构化数据：
# - 表查询：table_id, table_name, chunk_content, column_description
# - 需求查询：query_id, name, requirement


def _search_tables(query: str, k: int) -> str:
    """
    在表结构知识库中检索
    使用增强版格式化器返回完整数据：table_id, table_name, chunk_content, column_description
    
    Args:
        query: 查询文本
        k: 返回结果数量
        
    Returns:
        格式化的检索结果
    """
    if TABLE_VECTORSTORE is None:
        error_msg = f"❌ 表结构知识库未初始化，请检查路径: {TABLE_KB_PATH}"
        logger.error(error_msg)
        return error_msg
    
    try:
        # 使用 similarity_search_with_score 获取相似度分数
        results = TABLE_VECTORSTORE.similarity_search_with_score(query, k=k)
        
        if not results:
            return "未找到相关表结构信息。"
        
        # 使用增强版格式化器
        result_text = format_table_results(results, query)
        logger.info(f"表结构检索成功，返回 {len(results)} 条结果")
        return result_text
        
    except Exception as exc:
        error_msg = f"❌ 表结构检索失败: {exc}"
        logger.error(error_msg, exc_info=True)
        return error_msg


def _search_requirements(query: str, k: int) -> str:
    """
    在业务需求知识库中检索
    使用增强版格式化器返回完整数据：query_id, name, requirement
    
    Args:
        query: 查询文本
        k: 返回结果数量
        
    Returns:
        格式化的检索结果
    """
    if QUERY_VECTORSTORE is None:
        error_msg = f"❌ 业务需求知识库未初始化，请检查路径: {QUERY_KB_PATH}"
        logger.error(error_msg)
        return error_msg
    
    try:
        # 使用 similarity_search_with_score 获取相似度分数
        results = QUERY_VECTORSTORE.similarity_search_with_score(query, k=k)
        
        if not results:
            return "未找到相关业务需求。"
        
        # 使用增强版格式化器
        result_text = format_requirement_results(results, query)
        logger.info(f"业务需求检索成功，返回 {len(results)} 条结果")
        return result_text
        
    except Exception as exc:
        error_msg = f"❌ 业务需求检索失败: {exc}"
        logger.error(error_msg, exc_info=True)
        return error_msg

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("启动知识库 MCP 服务器")
    logger.info("=" * 60)
    logger.info(f"表结构知识库: {TABLE_KB_PATH}")
    logger.info(f"业务需求知识库: {QUERY_KB_PATH}")
    logger.info(f"Embedding 模型: {EMBED_MODEL}")
    
    # 检查知识库状态
    if TABLE_VECTORSTORE is None:
        logger.warning("⚠️  表结构知识库未成功加载")
    else:
        logger.info("✅ 表结构知识库已加载")
    
    if QUERY_VECTORSTORE is None:
        logger.warning("⚠️  业务需求知识库未成功加载")
    else:
        logger.info("✅ 业务需求知识库已加载")
    
    logger.info("=" * 60)
    logger.info("MCP 服务器运行中...")
    
    try:
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("\n收到中断信号，关闭服务器...")
    except Exception as e:
        logger.error(f"服务器运行错误: {e}", exc_info=True)
        raise  