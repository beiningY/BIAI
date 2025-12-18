#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试知识库查询功能
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# 配置
OPENAI_API_KEY = os.getenv("openrouter_api_key")
OPENAI_BASE_URL = os.getenv("url_openrouter")
CHROMA_DB_DIR = Path(__file__).parent / "chroma_db"
QUERY_KB_NAME = "query_requirements_kb"
TABLE_KB_NAME = "meta_tables_kb"
EMBED_MODEL = "openai/text-embedding-3-large"

if not OPENAI_API_KEY:
    print("❌ 错误: 未设置 openrouter_api_key")
    exit(1)


def load_vectorstore(kb_name: str):
    """加载向量数据库"""
    kb_path = CHROMA_DB_DIR / kb_name
    
    if not kb_path.exists():
        print(f"❌ 知识库路径不存在: {kb_path}")
        return None
    
    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        model=EMBED_MODEL
    )
    
    vectorstore = Chroma(
        collection_name=kb_name,
        embedding_function=embeddings,
        persist_directory=str(kb_path)
    )
    
    return vectorstore


def test_table_search():
    """测试表结构查询"""
    print("\n【测试 1】查询表结构知识库")
    print("=" * 60)
    
    vectorstore = load_vectorstore(TABLE_KB_NAME)
    if not vectorstore:
        print("❌ 无法加载表结构知识库")
        return False
    
    # 查询
    query = "用户信息表"
    print(f"查询: {query}")
    print("-" * 60)
    
    results = vectorstore.similarity_search_with_score(query, k=3)
    
    if not results:
        print("未找到结果")
        return False
    
    for i, (doc, distance) in enumerate(results, 1):
        similarity = 1 - distance
        table_name = doc.metadata.get('table_name', '未知表')
        print(f"\n【结果 {i}】表名: {table_name} [相似度: {similarity:.4f}]")
        print("-" * 60)
        print(doc.page_content[:200] + "...")
    
    return True


def test_requirement_search():
    """测试业务需求查询"""
    print("\n\n【测试 2】查询业务需求知识库")
    print("=" * 60)
    
    vectorstore = load_vectorstore(QUERY_KB_NAME)
    if not vectorstore:
        print("❌ 无法加载业务需求知识库")
        return False
    
    # 查询
    query = "放款金额统计"
    print(f"查询: {query}")
    print("-" * 60)
    
    results = vectorstore.similarity_search_with_score(query, k=3)
    
    if not results:
        print("未找到结果")
        return False
    
    for i, (doc, distance) in enumerate(results, 1):
        similarity = 1 - distance
        query_name = doc.metadata.get('name', '未知查询')
        print(f"\n【结果 {i}】查询: {query_name} [相似度: {similarity:.4f}]")
        print("-" * 60)
        print(doc.page_content[:200] + "...")
    
    return True


def main():
    """主函数"""
    print("=" * 60)
    print("知识库查询功能测试")
    print("=" * 60)
    
    # 测试表结构查询
    table_ok = test_table_search()
    
    # 测试业务需求查询
    requirement_ok = test_requirement_search()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    print(f"表结构查询: {'✅ 通过' if table_ok else '❌ 失败'}")
    print(f"业务需求查询: {'✅ 通过' if requirement_ok else '❌ 失败'}")
    
    if table_ok and requirement_ok:
        print("\n🎉 所有测试通过！")
    else:
        print("\n⚠️  部分测试失败")


if __name__ == "__main__":
    main()

