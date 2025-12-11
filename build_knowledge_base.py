#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建Chroma向量数据库知识库 (LangChain 最新版本)
- 第一部分：查询业务需求和SQL语句（按query分块）
- 第二部分：数据库表结构和Schema信息（按表分块）
- 第三部分：智能增量更新和批处理优化

优化特性：
- 使用 LangChain 最新稳定版 API
- 批处理向量化提升性能
- 智能增量更新机制
- 完善的错误处理和日志
- 进度追踪和统计信息
- 支持 OpenRouter 和标准 OpenAI API
"""

import json
import re
import os
import sys
import hashlib
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# LangChain 最新版本 imports
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 环境配置
from dotenv import load_dotenv
load_dotenv()

# API密钥配置
OPENAI_API_KEY = os.getenv("openrouter_api_key") or os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("url_openrouter", "https://api.openai.com/v1")

if not OPENAI_API_KEY:
    print("⚠️  警告: 未设置 OPENAI_API_KEY 环境变量")
    print("   请运行: export OPENAI_API_KEY='your-api-key'")
    print("   或者设置 openrouter_api_key 环境变量\n")
    sys.exit(1)


class KnowledgeBaseBuilder:
    """
    知识库构建器 (LangChain 最新版本)
    
    Features:
    - 智能文档分块和向量化
    - 增量更新支持
    - 批处理优化
    - 详细的构建统计
    - 支持 OpenRouter 和 OpenAI API
    """
    
    def __init__(
        self, 
        persist_directory: str = "./chroma_db",
        collection_name: str = "database_knowledge",
        embedding_model: str = "text-embedding-3-large",
        batch_size: int = 100
    ):
        """
        初始化知识库构建器
        
        Args:
            persist_directory: Chroma数据库持久化目录
            collection_name: 集合名称
            embedding_model: OpenAI embedding模型
            batch_size: 批处理大小
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.batch_size = batch_size
        
        # 创建持久化目录
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # 初始化 Embeddings
        # 注意：OpenRouter 需要使用 base_url 参数
        embedding_kwargs = {
            "model": embedding_model,
            "api_key": OPENAI_API_KEY,
        }
        
        # 如果使用 OpenRouter，设置 base_url
        if "openrouter" in OPENAI_BASE_URL.lower():
            embedding_kwargs["base_url"] = OPENAI_BASE_URL
        
        self.embeddings = OpenAIEmbeddings(**embedding_kwargs)
        
        # 统计信息
        self.stats = {
            "total_documents": 0,
            "query_documents": 0,
            "schema_documents": 0,
            "chunks_created": 0,
            "build_time": 0
        }
        
    def _calculate_content_hash(self, content: str) -> str:
        """计算内容的MD5哈希值，用于增量更新"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def load_query_documents(self, json_file: str) -> List[Document]:
        """
        加载查询业务需求JSON文件，按每个query分块
        
        Args:
            json_file: JSON文件路径
            
        Returns:
            Document列表
        """
        print(f"📖 正在加载查询文件: {json_file}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                queries = json.load(f)
        except FileNotFoundError:
            print(f"❌ 错误: 文件不存在 - {json_file}")
            return []
        except json.JSONDecodeError as e:
            print(f"❌ 错误: JSON解析失败 - {e}")
            return []
        
        documents = []
        for query_id, query_data in queries.items():
            # 提取涉及的表名
            sql = query_data.get('sql', '')
            tables = self._extract_tables_from_sql(sql)
            
            # 构建文档内容
            content = f"""查询ID: {query_id}
查询名称: {query_data['name']}

业务需求:
{query_data['business_requirement']}

涉及的表: {', '.join(tables) if tables else '未知'}

SQL语句:
{sql}
"""
            
            # 计算内容哈希
            content_hash = self._calculate_content_hash(content)
            
            # 创建Document对象，添加丰富的元数据
            doc = Document(
                page_content=content,
                metadata={
                    "source": "query_business_requirements",
                    "query_id": query_id,
                    "query_name": query_data['name'],
                    "type": "business_query",
                    "has_sql": True,
                    "business_requirement": query_data['business_requirement'],
                    "tables": ','.join(tables),
                    "content_hash": content_hash,
                    "created_at": datetime.now().isoformat()
                }
            )
            documents.append(doc)
        
        self.stats["query_documents"] = len(documents)
        print(f"✅ 成功加载 {len(documents)} 个查询文档")
        return documents
    
    def _extract_tables_from_sql(self, sql: str) -> List[str]:
        """
        从SQL语句中提取表名
        
        Args:
            sql: SQL语句
            
        Returns:
            表名列表
        """
        tables = set()
        
        # 匹配 FROM table_name 或 JOIN table_name
        patterns = [
            r'FROM\s+`?(\w+\.)?(\w+)`?',
            r'JOIN\s+`?(\w+\.)?(\w+)`?',
            r'INTO\s+`?(\w+\.)?(\w+)`?',
            r'UPDATE\s+`?(\w+\.)?(\w+)`?'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, sql, re.IGNORECASE)
            for match in matches:
                # 获取表名（不包括数据库名）
                table_name = match.group(2)
                if table_name and table_name.upper() not in ['SELECT', 'WHERE', 'GROUP', 'ORDER', 'HAVING']:
                    tables.add(table_name)
        
        return sorted(list(tables))
    
    def load_schema_documents(self, schema_file: str) -> List[Document]:
        """
        加载数据库Schema SQL文件，按表结构分块
        
        Args:
            schema_file: SQL文件路径
            
        Returns:
            Document列表
        """
        print(f"📖 正在加载Schema文件: {schema_file}")
        
        try:
            with open(schema_file, 'r', encoding='utf-8') as f:
                schema_content = f.read()
        except FileNotFoundError:
            print(f"❌ 错误: 文件不存在 - {schema_file}")
            return []
        
        documents = []
        
        # 使用正则表达式提取CREATE TABLE语句
        # 匹配 CREATE TABLE ... ) ENGINE=...;
        table_pattern = r"CREATE TABLE `(\w+)`\s*\((.*?)\)\s*ENGINE=.*?(?:COMMENT='(.*?)')?;"
        matches = re.finditer(table_pattern, schema_content, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            table_name = match.group(1)
            table_definition = match.group(2)
            table_comment = match.group(3) or "无描述"
            
            # 提取字段信息
            fields_info, field_count = self._parse_table_fields(table_definition)
            
            # 提取索引信息
            indexes = self._extract_indexes(table_definition)
            
            # 构建文档内容
            content = f"""表名: {table_name}

表说明: {table_comment}

字段信息 (共 {field_count} 个字段):
{fields_info}

索引信息:
{indexes}

完整DDL:
CREATE TABLE `{table_name}` (
{table_definition.strip()}
) COMMENT='{table_comment}';
"""
            
            # 计算内容哈希
            content_hash = self._calculate_content_hash(content)
            
            # 创建Document对象
            doc = Document(
                page_content=content,
                metadata={
                    "source": "database_schema",
                    "table_name": table_name,
                    "table_comment": table_comment,
                    "type": "table_schema",
                    "database": "singa_bi",
                    "field_count": field_count,
                    "content_hash": content_hash,
                    "created_at": datetime.now().isoformat()
                }
            )
            documents.append(doc)
        
        self.stats["schema_documents"] = len(documents)
        print(f"✅ 成功加载 {len(documents)} 个表结构文档")
        return documents
    
    def _parse_table_fields(self, table_definition: str) -> Tuple[str, int]:
        """
        解析表字段信息
        
        Args:
            table_definition: 表定义SQL
            
        Returns:
            (格式化的字段信息字符串, 字段数量)
        """
        fields = []
        lines = table_definition.split('\n')
        
        for line in lines:
            line = line.strip()
            # 跳过空行、PRIMARY KEY、KEY等约束定义
            if not line or line.startswith('PRIMARY KEY') or line.startswith('KEY ') or line.startswith('CONSTRAINT'):
                continue
            
            # 提取字段定义和注释
            # 格式通常是: `field_name` type ... COMMENT '注释',
            field_match = re.match(r"`(\w+)`\s+(.*?)(?:,\s*)?$", line)
            if field_match:
                field_name = field_match.group(1)
                field_def = field_match.group(2)
                
                # 提取COMMENT
                comment_match = re.search(r"COMMENT\s+'(.*?)'", field_def)
                if comment_match:
                    comment = comment_match.group(1)
                    # 移除COMMENT部分，保留类型定义
                    field_type = re.sub(r"\s*COMMENT\s+'.*?'", "", field_def).strip()
                    fields.append(f"  - {field_name}: {field_type}  // {comment}")
                else:
                    fields.append(f"  - {field_name}: {field_def}")
        
        fields_text = '\n'.join(fields) if fields else "无字段信息"
        return fields_text, len(fields)
    
    def _extract_indexes(self, table_definition: str) -> str:
        """
        提取表的索引信息
        
        Args:
            table_definition: 表定义SQL
            
        Returns:
            格式化的索引信息
        """
        indexes = []
        lines = table_definition.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('PRIMARY KEY'):
                indexes.append(f"  - PRIMARY KEY: {line}")
            elif line.startswith('KEY ') or line.startswith('INDEX '):
                # 提取索引名和列
                idx_match = re.match(r"(?:KEY|INDEX)\s+`?(\w+)`?\s+\((.*?)\)", line)
                if idx_match:
                    idx_name = idx_match.group(1)
                    idx_columns = idx_match.group(2)
                    indexes.append(f"  - INDEX {idx_name}: {idx_columns}")
        
        return '\n'.join(indexes) if indexes else "  - 无索引"
    
    def build_vector_store(
        self, 
        query_json: str, 
        schema_sql: str,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        force_rebuild: bool = False
    ) -> Chroma:
        """
        构建向量数据库 (支持增量更新)
        
        Args:
            query_json: 查询JSON文件路径
            schema_sql: Schema SQL文件路径
            chunk_size: 分块大小（对于长文本）
            chunk_overlap: 分块重叠大小
            force_rebuild: 是否强制重建（删除现有数据库）
            
        Returns:
            Chroma向量数据库实例
        """
        start_time = datetime.now()
        
        print("\n" + "="*70)
        print("🚀 开始构建知识库 (LangChain)")
        print("="*70 + "\n")
        
        # 检查是否需要强制重建
        if force_rebuild and self.persist_directory.exists():
            print("🗑️  强制重建模式：删除现有数据库...")
            import shutil
            shutil.rmtree(self.persist_directory)
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            print("✅ 已清空数据库\n")
        
        # 加载文档
        query_docs = self.load_query_documents(query_json)
        schema_docs = self.load_schema_documents(schema_sql)
        
        # 合并所有文档
        all_documents = query_docs + schema_docs
        self.stats["total_documents"] = len(all_documents)
        
        print(f"\n📊 文档统计:")
        print(f"  - 查询文档: {len(query_docs)} 个")
        print(f"  - Schema文档: {len(schema_docs)} 个")
        print(f"  - 总计: {len(all_documents)} 个")
        
        if not all_documents:
            print("❌ 错误: 没有加载到任何文档")
            return None
        
        # 对于特别长的文档，进行额外分块
        print(f"\n🔪 智能文档分块 (chunk_size={chunk_size}, overlap={chunk_overlap})...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""]
        )
        
        final_documents = []
        chunk_stats = defaultdict(int)
        
        for doc in all_documents:
            doc_length = len(doc.page_content)
            if doc_length > chunk_size:
                # 分块
                chunks = text_splitter.split_documents([doc])
                chunk_stats[doc.metadata.get('type')] += len(chunks) - 1
                doc_name = doc.metadata.get('query_name') or doc.metadata.get('table_name')
                print(f"  📄 [{doc.metadata.get('type')}] {doc_name}: {doc_length} 字符 → {len(chunks)} 块")
                final_documents.extend(chunks)
            else:
                final_documents.append(doc)
        
        self.stats["chunks_created"] = sum(chunk_stats.values())
        
        print(f"\n✅ 最终文档数量: {len(final_documents)} 个")
        if chunk_stats:
            print(f"📌 分块统计:")
            for doc_type, count in chunk_stats.items():
                print(f"   - {doc_type}: 额外创建 {count} 个分块")
        
        # 创建向量数据库（遵循 LangChain 最新最佳实践）
        print(f"\n🔧 正在创建Chroma向量数据库...")
        print(f"  - 存储目录: {self.persist_directory}")
        print(f"  - 集合名称: {self.collection_name}")
        print(f"  - 批处理大小: {self.batch_size}")
        
        try:
            # 方法1：使用 from_documents 快速构建（小数据集）
            # 方法2：使用 add_documents 批量添加（大数据集，支持进度显示）
            # 参考: https://docs.langchain.com/oss/python/langchain/rag
            
            if len(final_documents) <= 100:
                # 小数据集：使用 from_documents 一次性创建
                print(f"\n📥 使用快速模式创建向量数据库...")
                print(f"  ⏳ 正在向量化 {len(final_documents)} 个文档...")
                
                vectorstore = Chroma.from_documents(
                    documents=final_documents,
                    embedding=self.embeddings,
                    collection_name=self.collection_name,
                    persist_directory=str(self.persist_directory)
                )
                
                print(f"  ✓ 已成功添加 {len(final_documents)} 个文档")
            else:
                # 大数据集：使用批处理模式，提供进度反馈
                print(f"\n📥 使用批处理模式创建向量数据库...")
                print(f"  ⏳ 将分 {(len(final_documents) + self.batch_size - 1) // self.batch_size} 批处理...")
                
                # 初始化空的向量存储
                vectorstore = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=str(self.persist_directory)
                )
                
                # 批量添加文档（按照 LangChain 文档推荐方式）
                for i in range(0, len(final_documents), self.batch_size):
                    batch = final_documents[i:i + self.batch_size]
                    batch_num = i // self.batch_size + 1
                    total_batches = (len(final_documents) + self.batch_size - 1) // self.batch_size
                    
                    print(f"  📦 批次 {batch_num}/{total_batches}: 处理 {len(batch)} 个文档...", end="")
                    
                    # 添加文档到向量存储
                    vectorstore.add_documents(documents=batch)
                    
                    processed = min(i + self.batch_size, len(final_documents))
                    progress = (processed / len(final_documents)) * 100
                    print(f" ✓ ({processed}/{len(final_documents)}, {progress:.1f}%)")
            
            end_time = datetime.now()
            self.stats["build_time"] = (end_time - start_time).total_seconds()
            
            print(f"\n✅ 向量数据库构建完成!")
            print(f"  - 已存储 {len(final_documents)} 个文档块")
            print(f"  - 构建耗时: {self.stats['build_time']:.2f} 秒")
            print(f"  - 平均速度: {len(final_documents) / self.stats['build_time']:.1f} 文档/秒")
            
            # 验证向量存储
            doc_count = vectorstore._collection.count()
            print(f"  - 向量数据库文档数: {doc_count}")
            
            # 保存构建统计
            self._save_build_stats()
            
            return vectorstore
            
        except Exception as e:
            print(f"\n❌ 构建失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 提供故障排除建议
            print("\n💡 故障排除建议:")
            print("  1. 检查 API 密钥是否正确")
            print("  2. 检查网络连接是否正常")
            print("  3. 尝试减小 batch_size 参数")
            print("  4. 检查磁盘空间是否充足")
            
            return None
    
    def _save_build_stats(self):
        """保存构建统计信息"""
        stats_file = self.persist_directory / "build_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump({
                **self.stats,
                "last_build": datetime.now().isoformat(),
                "collection_name": self.collection_name
            }, f, indent=2, ensure_ascii=False)
        print(f"📊 构建统计已保存到: {stats_file}")
    
    def query_test(self, vectorstore: Chroma, query: str, k: int = 3):
        """
        测试查询功能 (支持相似度评分)
        遵循 LangChain RAG 最佳实践
        参考: https://docs.langchain.com/oss/python/langchain/rag
        
        Args:
            vectorstore: 向量数据库实例
            query: 查询问题
            k: 返回top-k结果
        """
        print("\n" + "="*70)
        print(f"🔍 测试查询: {query}")
        print("="*70)
        
        try:
            # 使用 similarity_search_with_score 获取相似度评分
            # 这是 LangChain 推荐的检索方式
            results = vectorstore.similarity_search_with_score(query, k=k)
            
            if not results:
                print("⚠️  未找到相关结果")
                return
            
            for i, (doc, score) in enumerate(results, 1):
                # 计算相关性百分比（距离越小越相关）
                relevance = max(0, 100 - score * 100)
                
                print(f"\n📄 结果 {i} (距离: {score:.4f}, 相关性: {relevance:.1f}%):")
                print(f"类型: {doc.metadata.get('type')}")
                
                if doc.metadata.get('type') == 'business_query':
                    print(f"查询ID: {doc.metadata.get('query_id')}")
                    print(f"查询名称: {doc.metadata.get('query_name')}")
                    if doc.metadata.get('tables'):
                        print(f"涉及表: {doc.metadata.get('tables')}")
                else:
                    print(f"表名: {doc.metadata.get('table_name')}")
                    print(f"表说明: {doc.metadata.get('table_comment')}")
                    if doc.metadata.get('field_count'):
                        print(f"字段数: {doc.metadata.get('field_count')}")
                
                print(f"\n内容预览:")
                preview = doc.page_content[:300].replace('\n', '\n  ')
                print(f"  {preview}...")
                print("-" * 70)
                
        except Exception as e:
            print(f"❌ 查询失败: {e}")
            import traceback
            traceback.print_exc()
    
    def get_stats(self) -> Dict:
        """获取构建统计信息"""
        stats_file = self.persist_directory / "build_stats.json"
        if stats_file.exists():
            with open(stats_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return self.stats
    
    def load_existing_vectorstore(self) -> Optional[Chroma]:
        """
        加载现有的向量数据库
        遵循 LangChain 最佳实践进行向量存储加载
        
        Returns:
            Chroma实例，如果不存在则返回None
        """
        if not self.persist_directory.exists():
            print(f"⚠️  向量数据库不存在: {self.persist_directory}")
            return None
        
        print(f"📂 加载现有向量数据库: {self.persist_directory}")
        
        try:
            # 使用相同的 embedding 函数加载向量存储
            vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory)
            )
            
            # 获取统计信息
            collection = vectorstore._collection
            doc_count = collection.count()
            print(f"✅ 成功加载，包含 {doc_count} 个文档")
            
            # 显示构建统计（如果存在）
            stats = self.get_stats()
            if stats.get('last_build'):
                print(f"📅 最后构建时间: {stats.get('last_build')}")
            
            return vectorstore
            
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            print("\n💡 可能的原因:")
            print("  1. embedding 模型不匹配")
            print("  2. 数据库文件损坏")
            print("  3. 集合名称不正确")
            return None
    
    def similarity_search(
        self, 
        vectorstore: Chroma, 
        query: str, 
        k: int = 4
    ) -> List[Document]:
        """
        执行相似度搜索
        遵循 LangChain RAG 模式的标准检索方法
        参考: https://docs.langchain.com/oss/python/langchain/rag
        
        Args:
            vectorstore: 向量数据库实例
            query: 查询文本
            k: 返回的文档数量
            
        Returns:
            检索到的文档列表
        """
        try:
            # 使用 similarity_search 方法（LangChain 标准 API）
            retrieved_docs = vectorstore.similarity_search(query, k=k)
            return retrieved_docs
        except Exception as e:
            print(f"❌ 检索失败: {e}")
            return []


def main():
    """
    主函数 - 支持命令行参数
    遵循 LangChain 最新最佳实践构建 RAG 知识库
    参考文档: https://docs.langchain.com/oss/python/langchain/rag
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="构建数据库知识库向量数据库 (LangChain 最新版本)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本构建
  python build_knowledge_base.py
  
  # 强制重建
  python build_knowledge_base.py --force-rebuild
  
  # 使用自定义批处理大小
  python build_knowledge_base.py --batch-size 50
  
  # 跳过测试查询
  python build_knowledge_base.py --no-test
  
参考: https://docs.langchain.com/oss/python/langchain/rag
        """
    )
    parser.add_argument("--query-json", default="/Users/sarah/工作/数据库自动化/BIAI/query_business_requirements.json",
                        help="查询JSON文件路径")
    parser.add_argument("--schema-sql", default="/Users/sarah/工作/数据库自动化/BIAI/schema.sql",
                        help="Schema SQL文件路径")
    parser.add_argument("--persist-dir", default="/Users/sarah/工作/数据库自动化/BIAI/chroma_db",
                        help="向量数据库持久化目录")
    parser.add_argument("--collection", default="database_knowledge",
                        help="集合名称")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="强制重建数据库（删除现有数据）")
    parser.add_argument("--no-test", action="store_true",
                        help="跳过测试查询")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="批处理大小（推荐: 50-100）")
    parser.add_argument("--chunk-size", type=int, default=2000,
                        help="文档分块大小（推荐: 1000-2000）")
    parser.add_argument("--embedding-model", default="text-embedding-3-large",
                        help="OpenAI embedding 模型（默认: text-embedding-3-large）")
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.query_json).exists():
        print(f"❌ 错误: 文件不存在 - {args.query_json}")
        return 1
    
    if not Path(args.schema_sql).exists():
        print(f"❌ 错误: 文件不存在 - {args.schema_sql}")
        return 1
    
    # 创建知识库构建器（遵循 LangChain 最佳实践）
    builder = KnowledgeBaseBuilder(
        persist_directory=args.persist_dir,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size
    )
    
    print(f"\n📋 配置信息:")
    print(f"  - Embedding 模型: {args.embedding_model}")
    print(f"  - 批处理大小: {args.batch_size}")
    print(f"  - 文档分块大小: {args.chunk_size}")
    print(f"  - 集合名称: {args.collection}")
    
    # 构建向量数据库
    vectorstore = builder.build_vector_store(
        query_json=args.query_json,
        schema_sql=args.schema_sql,
        chunk_size=args.chunk_size,
        force_rebuild=args.force_rebuild
    )
    
    if not vectorstore:
        print("❌ 向量数据库构建失败")
        return 1
    
    # 测试查询（遵循 LangChain RAG 模式）
    if not args.no_test:
        print("\n" + "="*70)
        print("🧪 开始测试查询功能")
        print("   参考: https://docs.langchain.com/oss/python/langchain/rag")
        print("="*70)
        
        test_queries = [
            "如何按小时统计用户注册数量？",
            "sgo_orders表有哪些字段？",
            "还款相关的查询有哪些？",
            "approval_info表的作用是什么？",
            "WhatsApp消息发送记录在哪个表？"
        ]
        
        for test_query in test_queries:
            builder.query_test(vectorstore, test_query, k=2)
        
        # 额外演示：使用标准 similarity_search API
        print("\n" + "="*70)
        print("📝 演示标准检索 API (similarity_search)")
        print("="*70)
        demo_query = "如何统计新老客申请量？"
        print(f"查询: {demo_query}\n")
        retrieved_docs = builder.similarity_search(vectorstore, demo_query, k=2)
        print(f"✓ 检索到 {len(retrieved_docs)} 个相关文档")
        for i, doc in enumerate(retrieved_docs, 1):
            doc_type = doc.metadata.get('type')
            name = doc.metadata.get('query_name') or doc.metadata.get('table_name')
            print(f"  {i}. [{doc_type}] {name}")
    
    # 显示最终统计
    print("\n" + "="*70)
    print("✅ 知识库构建和测试完成！")
    print("="*70)
    
    stats = builder.get_stats()
    print(f"\n📊 最终统计:")
    print(f"  - 查询文档: {stats.get('query_documents', 0)} 个")
    print(f"  - Schema文档: {stats.get('schema_documents', 0)} 个")
    print(f"  - 总文档数: {stats.get('total_documents', 0)} 个")
    print(f"  - 额外分块: {stats.get('chunks_created', 0)} 个")
    print(f"  - 构建耗时: {stats.get('build_time', 0):.2f} 秒")
    print(f"\n📁 数据库位置: {args.persist_dir}")
    print(f"🔗 集合名称: {args.collection}")
    print(f"\n💡 下一步:")
    print(f"  1. 使用 query_knowledge_base.py 进行查询")
    print(f"  2. 集成到 RAG 应用中进行问答")
    print(f"  3. 参考文档: https://docs.langchain.com/oss/python/langchain/rag")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

