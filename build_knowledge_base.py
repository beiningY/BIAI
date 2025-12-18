#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识库构建脚本
使用 LangChain 和 Chroma 构建两个向量数据库：
1. query_requirements_kb: 基于业务需求查询的知识库
2. meta_tables_kb: 基于数据表元数据的知识库
"""

import json
import os
from typing import List, Dict
from langchain_chroma import Chroma  
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings  
import dotenv
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("openrouter_api_key")
OPENAI_BASE_URL = os.getenv("url_openrouter")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")
if not OPENAI_BASE_URL:
    raise ValueError("OPENAI_BASE_URL is not set")

# 配置参数
DATA_DIR = "data"
CHROMA_DB_DIR = "chroma_db"
QUERY_REQUIREMENTS_FILE = "query_business_requirements.json"
META_TABLES_FILE = "singabi_meta_tables.json"

# 知识库名称
QUERY_KB_NAME = "query_requirements_kb"
TABLE_KB_NAME = "meta_tables_kb"


def load_json_file(file_path: str) -> dict:
    """加载 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_query_requirements_kb(data: dict) -> List[Document]:
    """
    构建业务需求查询知识库
    将每个 business_requirement 作为 chunk，id 和 name 作为元数据
    
    Args:
        data: query_business_requirements.json 的数据
        
    Returns:
        Document 列表
    """
    documents = []
    
    for key, value in data.items():
        # 创建 Document 对象
        doc = Document(
            page_content=value.get("business_requirement", ""),
            metadata={
                "id": key,
                "name": value.get("name", ""),
                "source": "query_business_requirements"
            }
        )
        documents.append(doc)
    
    print(f"✅ 业务需求知识库：加载了 {len(documents)} 个文档")
    return documents


def build_meta_tables_kb(data: dict) -> List[Document]:
    """
    构建数据表元数据知识库
    将每个 chunk_content 作为 chunk，id 和 table_name 作为元数据
    
    Args:
        data: singabi_meta_tables.json 的数据
        
    Returns:
        Document 列表
    """
    documents = []
    
    for key, value in data.items():
        # 创建 Document 对象
        doc = Document(
            page_content=value.get("chunk_content", ""),
            metadata={
                "id": value.get("id", key),
                "table_name": value.get("table_name", ""),
                "source": "singabi_meta_tables"
            }
        )
        documents.append(doc)
    
    print(f"✅ 数据表元数据知识库：加载了 {len(documents)} 个文档")
    return documents


def create_vector_store(
    documents: List[Document],
    collection_name: str,
    embeddings,
    persist_directory: str
) -> Chroma:
    """
    创建向量数据库
    
    Args:
        documents: 文档列表
        collection_name: 集合名称
        embeddings: 嵌入模型
        persist_directory: 持久化目录
        
    Returns:
        Chroma 向量数据库实例
    """
    print(f"🔄 正在创建向量数据库：{collection_name}")
    
    # 创建向量数据库
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    
    print(f"✅ 向量数据库创建完成：{collection_name}")
    print(f"   - 文档数量: {len(documents)}")
    print(f"   - 存储路径: {persist_directory}")
    
    return vectorstore


def main():
    """主函数"""
    print("=" * 60)
    print("开始构建知识库...")
    print("=" * 60)
    
    # 1. 检查 OpenAI API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 错误：未设置 OPENAI_API_KEY 环境变量")
        print("   请设置环境变量：export OPENAI_API_KEY='your-api-key'")
        return
    
    # 2. 初始化嵌入模型
    print("\n📦 初始化嵌入模型...")
    try:
        embeddings = OpenAIEmbeddings(  
            api_key=OPENAI_API_KEY,  # OpenRouter API 密钥  
            base_url=OPENAI_BASE_URL,      # OpenRouter API 端点  
            model="openai/text-embedding-3-large"        # 选择嵌入模型  
        )
        print("✅ 嵌入模型初始化成功")
    except Exception as e:
        print(f"❌ 嵌入模型初始化失败：{e}")
        return
    
    # 3. 创建存储目录
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    
    # 4. 构建业务需求查询知识库
    print("\n" + "=" * 60)
    print("【知识库 1】业务需求查询知识库")
    print("=" * 60)
    
    try:
        query_data_path = os.path.join(DATA_DIR, QUERY_REQUIREMENTS_FILE)
        query_data = load_json_file(query_data_path)
        query_documents = build_query_requirements_kb(query_data)
        
        query_kb_path = os.path.join(CHROMA_DB_DIR, QUERY_KB_NAME)
        query_vectorstore = create_vector_store(
            documents=query_documents,
            collection_name=QUERY_KB_NAME,
            embeddings=embeddings,
            persist_directory=query_kb_path
        )
        
        # 测试查询
        print("\n🔍 测试查询：查找与'放款金额'相关的内容")
        results = query_vectorstore.similarity_search("放款金额统计", k=3)
        for i, doc in enumerate(results, 1):
            print(f"\n   结果 {i}:")
            print(f"   - ID: {doc.metadata.get('id')}")
            print(f"   - Name: {doc.metadata.get('name')}")
            print(f"   - Content: {doc.page_content[:100]}...")
        
    except Exception as e:
        print(f"❌ 构建业务需求知识库失败：{e}")
        import traceback
        traceback.print_exc()
    
    # 5. 构建数据表元数据知识库
    print("\n" + "=" * 60)
    print("【知识库 2】数据表元数据知识库")
    print("=" * 60)
    
    try:
        table_data_path = os.path.join(DATA_DIR, META_TABLES_FILE)
        table_data = load_json_file(table_data_path)
        table_documents = build_meta_tables_kb(table_data)
        
        table_kb_path = os.path.join(CHROMA_DB_DIR, TABLE_KB_NAME)
        table_vectorstore = create_vector_store(
            documents=table_documents,
            collection_name=TABLE_KB_NAME,
            embeddings=embeddings,
            persist_directory=table_kb_path
        )
        
        # 测试查询
        print("\n🔍 测试查询：查找与'用户信息'相关的表")
        results = table_vectorstore.similarity_search("用户基本信息表", k=3)
        for i, doc in enumerate(results, 1):
            print(f"\n   结果 {i}:")
            print(f"   - ID: {doc.metadata.get('id')}")
            print(f"   - Table: {doc.metadata.get('table_name')}")
            print(f"   - Content: {doc.page_content[:150]}...")
        
    except Exception as e:
        print(f"❌ 构建数据表元数据知识库失败：{e}")
        import traceback
        traceback.print_exc()
    
    # 6. 完成
    print("\n" + "=" * 60)
    print("✅ 知识库构建完成！")
    print("=" * 60)
    print(f"\n📁 知识库存储位置：")
    print(f"   1. 业务需求查询知识库: {os.path.join(CHROMA_DB_DIR, QUERY_KB_NAME)}")
    print(f"   2. 数据表元数据知识库: {os.path.join(CHROMA_DB_DIR, TABLE_KB_NAME)}")
    print(f"\n💡 使用方法：")
    print(f"   可以使用 query_knowledge_base.py 脚本查询知识库")


if __name__ == "__main__":
    main()
