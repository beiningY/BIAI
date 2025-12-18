#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的知识库查询脚本
用于查询已构建的两个向量数据库
"""

import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 配置参数
CHROMA_DB_DIR = "chroma_db"
QUERY_KB_NAME = "query_requirements_kb"
TABLE_KB_NAME = "meta_tables_kb"

import dotenv
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("openrouter_api_key")
OPENAI_BASE_URL = os.getenv("url_openrouter")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")
if not OPENAI_BASE_URL:
    raise ValueError("OPENAI_BASE_URL is not set")

def load_knowledge_base(kb_name: str, persist_directory: str):
    """
    加载知识库
    
    Args:
        kb_name: 知识库名称
        persist_directory: 持久化目录
        
    Returns:
        Chroma 向量数据库实例
    """
    embeddings = OpenAIEmbeddings(  
        api_key=OPENAI_API_KEY,  # OpenRouter API 密钥  
        base_url=OPENAI_BASE_URL,      # OpenRouter API 端点  
        model="openai/text-embedding-3-large"        # 选择嵌入模型  
    )
    
    vectorstore = Chroma(
        collection_name=kb_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    
    return vectorstore


def query_requirements_kb(query: str, k: int = 10):
    """
    查询业务需求知识库
    
    Args:
        query: 查询文本
        k: 返回结果数量
    """
    print(f"\n🔍 查询业务需求知识库：{query}")
    print("=" * 60)
    
    kb_path = os.path.join(CHROMA_DB_DIR, QUERY_KB_NAME)
    vectorstore = load_knowledge_base(QUERY_KB_NAME, kb_path)
    
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n【结果 {i}】相似度: {1 - score:.4f}")
        print(f"ID: {doc.metadata.get('id')}")
        print(f"名称: {doc.metadata.get('name')}")
        print(f"业务需求: {doc.page_content}")
        print("-" * 60)
    
    return results


def query_tables_kb(query: str, k: int = 10):
    """
    查询数据表元数据知识库
    
    Args:
        query: 查询文本
        k: 返回结果数量
    """
    print(f"\n🔍 查询数据表元数据知识库：{query}")
    print("=" * 60)
    
    kb_path = os.path.join(CHROMA_DB_DIR, TABLE_KB_NAME)
    vectorstore = load_knowledge_base(TABLE_KB_NAME, kb_path)
    
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n【结果 {i}】相似度: {1 - score:.4f}")
        print(f"ID: {doc.metadata.get('id')}")
        print(f"表名: {doc.metadata.get('table_name')}")
        print(f"描述:\n{doc.page_content}")
        print("-" * 60)
    
    return results


def main():
    """主函数"""
    # 检查 OpenAI API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 错误：未设置 OPENAI_API_KEY 环境变量")
        return
    
    print("=" * 60)
    print("知识库查询工具")
    print("=" * 60)
    
    while True:
        print("\n请选择要查询的知识库：")
        print("1. 业务需求查询知识库")
        print("2. 数据表元数据知识库")
        print("3. 退出")
        
        choice = input("\n请输入选项 (1/2/3): ").strip()
        
        if choice == "1":
            query = input("\n请输入查询内容: ").strip()
            if query:
                try:
                    query_requirements_kb(query)
                except Exception as e:
                    print(f"❌ 查询失败: {e}")
        
        elif choice == "2":
            query = input("\n请输入查询内容: ").strip()
            if query:
                try:
                    query_tables_kb(query)
                except Exception as e:
                    print(f"❌ 查询失败: {e}")
        
        elif choice == "3":
            print("\n👋 再见！")
            break
        
        else:
            print("❌ 无效选项，请重新选择")


if __name__ == "__main__":
    main()

