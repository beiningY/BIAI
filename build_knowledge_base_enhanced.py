#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整合三个数据源，构建完整的知识库
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

# 数据文件
QUERY_REQUIREMENTS_FILE = "query_business_requirements.json"
META_TABLES_FILE = "singabi_meta_tables.json"
DATA_DICTIONARY_FILE = "singabi_data_dictionary.json"

# 知识库名称
QUERY_KB_NAME = "query_requirements_kb"
TABLE_KB_NAME = "meta_tables_kb"


def load_json_file(file_path: str):
    """加载 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_query_requirements_kb(query_data: dict) -> List[Document]:
    """
    构建业务需求查询知识库
    
    整合内容：
    - query id
    - query name  
    - business_requirement (作为主要检索内容)
    
    Args:
        query_data: query_business_requirements.json 的数据
        
    Returns:
        Document 列表
    """
    documents = []
    
    for query_id, value in query_data.items():
        name = value.get("name", "")
        requirement = value.get("business_requirement", "")
        
        # 构建完整的内容用于检索
        # 包含 name 和 requirement，让向量检索更准确
        full_content = f"查询ID: {query_id}\n查询名称: {name}\n\n业务需求: {requirement}"
        
        # 创建 Document 对象
        doc = Document(
            page_content=full_content,
            metadata={
                "query_id": query_id,
                "name": name,
                "requirement": requirement,
                "source": "query_business_requirements"
            }
        )
        documents.append(doc)
    
    print(f"✅ 业务需求知识库：加载了 {len(documents)} 个文档")
    return documents


def build_meta_tables_kb(
    meta_data: dict, 
    dict_data: list
) -> List[Document]:
    """
    构建数据表元数据知识库
    
    整合三个数据源：
    1. singabi_meta_tables.json: id, table_name, chunk_content
    2. singabi_data_dictionary.json: table_id, table_name, column_description
    
    Args:
        meta_data: singabi_meta_tables.json 的数据
        dict_data: singabi_data_dictionary.json 的数据
        
    Returns:
        Document 列表
    """
    # 先构建 dictionary 的映射表
    dict_map = {}
    for item in dict_data:
        table_name = item.get('table_name', '')
        column_desc = item.get('column_description', '')
        if table_name and column_desc:
            dict_map[table_name] = column_desc
    
    print(f"📖 数据字典：加载了 {len(dict_map)} 个字段描述")
    
    documents = []
    
    for key, value in meta_data.items():
        table_id = value.get("id", key)
        table_name = value.get("table_name", "")
        chunk_content = value.get("chunk_content", "")
        
        # 从 dictionary 获取额外的字段描述
        column_description = dict_map.get(table_name, "")
        
        # 构建完整的内容用于检索
        # chunk_content 已经很完整，如果有额外的 column_description 就追加
        full_content = chunk_content
        if column_description:
            full_content += f"\n\n补充字段说明:\n{column_description}"
        
        # 创建 Document 对象
        doc = Document(
            page_content=full_content,
            metadata={
                "table_id": table_id,
                "table_name": table_name,
                "chunk_content": chunk_content,
                "column_description": column_description,
                "source": "singabi_meta_tables and singabi_data_dictionary"
            }
        )
        documents.append(doc)
    
    print(f"✅ 数据表元数据知识库：加载了 {len(documents)} 个文档")
    return documents


def create_vector_store(
    documents: List[Document],
    collection_name: str,
    persist_directory: str
) -> Chroma:
    """
    创建向量数据库
    
    Args:
        documents: Document 列表
        collection_name: 集合名称
        persist_directory: 持久化目录
        
    Returns:
        Chroma 向量数据库实例
    """
    print(f"\n🔨 正在创建向量数据库: {collection_name}")
    print(f"   文档数量: {len(documents)}")
    print(f"   持久化目录: {persist_directory}")
    
    # 创建 Embeddings
    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        model="openai/text-embedding-3-large"
    )
    
    # 创建向量数据库
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    
    print(f"✅ 向量数据库创建成功！")
    return vectorstore


def main():
    """主函数"""
    print("=" * 60)
    print("🚀 开始构建增强版知识库")
    print("=" * 60)
    
    # 1. 加载数据文件
    print("\n📂 加载数据文件...")
    
    query_data_path = os.path.join(DATA_DIR, QUERY_REQUIREMENTS_FILE)
    meta_data_path = os.path.join(DATA_DIR, META_TABLES_FILE)
    dict_data_path = os.path.join(DATA_DIR, DATA_DICTIONARY_FILE)
    
    query_data = load_json_file(query_data_path)
    meta_data = load_json_file(meta_data_path)
    dict_data = load_json_file(dict_data_path)
    
    print(f"   ✅ {QUERY_REQUIREMENTS_FILE}: {len(query_data)} 条")
    print(f"   ✅ {META_TABLES_FILE}: {len(meta_data)} 条")
    print(f"   ✅ {DATA_DICTIONARY_FILE}: {len(dict_data)} 条")
    
    # 2. 构建业务需求知识库
    print("\n📝 构建业务需求知识库...")
    query_documents = build_query_requirements_kb(query_data)
    
    query_kb_dir = os.path.join(CHROMA_DB_DIR, QUERY_KB_NAME)
    query_vectorstore = create_vector_store(
        documents=query_documents,
        collection_name=QUERY_KB_NAME,
        persist_directory=query_kb_dir
    )
    
    # 3. 构建数据表元数据知识库（整合 meta_tables 和 dictionary）
    print("\n📊 构建数据表元数据知识库...")
    table_documents = build_meta_tables_kb(meta_data, dict_data)
    
    table_kb_dir = os.path.join(CHROMA_DB_DIR, TABLE_KB_NAME)
    table_vectorstore = create_vector_store(
        documents=table_documents,
        collection_name=TABLE_KB_NAME,
        persist_directory=table_kb_dir
    )
    
    # 4. 完成
    print("\n" + "=" * 60)
    print("🎉 知识库构建完成！")
    print("=" * 60)
    print(f"\n📍 知识库位置:")
    print(f"   1. 业务需求知识库: {query_kb_dir}")
    print(f"   2. 数据表元数据知识库: {table_kb_dir}")
    print(f"\n📊 统计信息:")
    print(f"   业务需求文档数: {len(query_documents)}")
    print(f"   数据表文档数: {len(table_documents)}")
    print(f"   总文档数: {len(query_documents) + len(table_documents)}")
    print("\n✨ 可以开始使用 MCP 服务器了！")


if __name__ == "__main__":
    main()

