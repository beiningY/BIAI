#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 MCP 工具集成
模拟调用 kb_search_tables 和 kb_search_requirements
"""

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from enhanced_formatter import format_table_results, format_requirement_results

load_dotenv()

print("=" * 80)
print("🧪 测试 MCP 工具集成")
print("=" * 80)

# 模拟 MCP 服务器的初始化
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("openrouter_api_key"),
    base_url=os.getenv("url_openrouter"),
    model="openai/text-embedding-3-large"
)

TABLE_VECTORSTORE = Chroma(
    collection_name="meta_tables_kb",
    embedding_function=embeddings,
    persist_directory="chroma_db/meta_tables_kb"
)

QUERY_VECTORSTORE = Chroma(
    collection_name="query_requirements_kb",
    embedding_function=embeddings,
    persist_directory="chroma_db/query_requirements_kb"
)

# 模拟 kb_search_tables 工具
def kb_search_tables(query: str, k: int = 5) -> str:
    """模拟 MCP 工具"""
    k = max(1, min(20, k))
    results = TABLE_VECTORSTORE.similarity_search_with_score(query, k=k)
    return format_table_results(results, query)

# 模拟 kb_search_requirements 工具
def kb_search_requirements(query: str, k: int = 5) -> str:
    """模拟 MCP 工具"""
    k = max(1, min(20, k))
    results = QUERY_VECTORSTORE.similarity_search_with_score(query, k=k)
    return format_requirement_results(results, query)


# ===== 测试1：表结构检索 =====
print("\n【测试1】kb_search_tables() 工具")
print("-" * 80)
print("🔧 调用: kb_search_tables('放款订单', k=2)\n")

result = kb_search_tables("放款订单", k=2)
print(result)


# ===== 测试2：业务需求检索 =====
print("\n\n【测试2】kb_search_requirements() 工具")
print("-" * 80)
print("🔧 调用: kb_search_requirements('统计放款金额', k=2)\n")

result = kb_search_requirements("统计放款金额", k=2)
print(result)


# ===== 验证返回数据 =====
print("\n\n【验证】返回数据完整性")
print("-" * 80)

# 表查询返回的数据
table_results = TABLE_VECTORSTORE.similarity_search_with_score("订单", k=1)
if table_results:
    doc, score = table_results[0]
    print("✅ 表查询返回的 metadata:")
    for key, value in doc.metadata.items():
        value_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
        print(f"   • {key}: {value_str}")

print()

# 需求查询返回的数据
query_results = QUERY_VECTORSTORE.similarity_search_with_score("放款", k=1)
if query_results:
    doc, score = query_results[0]
    print("✅ 需求查询返回的 metadata:")
    for key, value in doc.metadata.items():
        value_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
        print(f"   • {key}: {value_str}")

print("\n" + "=" * 80)
print("✅ MCP 工具集成测试完成！")
print("=" * 80)
print("\n💡 提示:")
print("   - 表查询返回：table_id, table_name, chunk_content, column_description")
print("   - 需求查询返回：query_id, name, requirement")
print("   - Agent 可以获得完整的结构化数据")

