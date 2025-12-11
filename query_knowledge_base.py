#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用Chroma向量数据库进行查询的示例脚本
"""

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List, Dict
import os
import dotenv
dotenv.load_dotenv()
# 尝试从环境变量读取配置
# API密钥配置
OPENAI_API_KEY = os.getenv("openrouter_api_key") or os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("url_openrouter", "https://api.openai.com/v1")
if not OPENAI_API_KEY:
    print("⚠️  警告: 未设置 OPENAI_API_KEY 环境变量")
    print("   请运行: export OPENAI_API_KEY='your-api-key'")
    print("   或者在代码中设置 os.environ['OPENAI_API_KEY'] = 'your-api-key'\n")
class KnowledgeBaseQuery:
    """知识库查询器"""
    
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "database_knowledge",embedding_model: str = "text-embedding-3-large"):
        """
        初始化查询器
        
        Args:
            persist_directory: Chroma数据库目录
            collection_name: 集合名称
        """
        print(f"📖 加载知识库: {persist_directory}")
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
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        print("✅ 知识库加载成功!\n")
    
    def search_business_queries(self, question: str, k: int = 5) -> List[Dict]:
        """
        搜索业务查询相关内容
        
        Args:
            question: 用户问题
            k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        # 使用过滤器只搜索业务查询类型
        results = self.vectorstore.similarity_search(
            question,
            k=k,
            filter={"type": "business_query"}
        )
        
        return self._format_results(results)
    
    def search_table_schemas(self, question: str, k: int = 5) -> List[Dict]:
        """
        搜索表结构相关内容
        
        Args:
            question: 用户问题
            k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        # 使用过滤器只搜索表结构类型
        results = self.vectorstore.similarity_search(
            question,
            k=k,
            filter={"type": "table_schema"}
        )
        
        return self._format_results(results)
    
    def search_all(self, question: str, k: int = 5) -> List[Dict]:
        """
        搜索所有类型的内容
        
        Args:
            question: 用户问题
            k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        results = self.vectorstore.similarity_search(question, k=k)
        return self._format_results(results)
    
    def search_with_score(self, question: str, k: int = 5) -> List[tuple]:
        """
        搜索并返回相似度分数
        
        Args:
            question: 用户问题
            k: 返回结果数量
            
        Returns:
            (Document, score) 元组列表
        """
        return self.vectorstore.similarity_search_with_score(question, k=k)
    
    def _format_results(self, results: List) -> List[Dict]:
        """
        格式化搜索结果
        
        Args:
            results: Document列表
            
        Returns:
            格式化的字典列表
        """
        formatted = []
        for doc in results:
            formatted.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        return formatted
    
    def pretty_print_results(self, results: List[Dict]):
        """
        美化打印搜索结果
        
        Args:
            results: 格式化的结果列表
        """
        for i, result in enumerate(results, 1):
            print(f"\n{'='*70}")
            print(f"📄 结果 {i}")
            print(f"{'='*70}")
            
            metadata = result['metadata']
            print(f"类型: {metadata.get('type')}")
            
            if metadata.get('type') == 'business_query':
                print(f"查询ID: {metadata.get('query_id')}")
                print(f"查询名称: {metadata.get('query_name')}")
            elif metadata.get('type') == 'table_schema':
                print(f"表名: {metadata.get('table_name')}")
                print(f"表说明: {metadata.get('table_comment')}")
            
            print(f"\n内容:\n{result['content']}")
        
        print(f"\n{'='*70}\n")


def main():
    """主函数 - 使用示例"""
    
    # 初始化查询器
    query_engine = KnowledgeBaseQuery(
        persist_directory="/Users/sarah/工作/数据库自动化/BIAI/chroma_db",
        collection_name="database_knowledge",
        embedding_model="text-embedding-3-large"
    )
    
    # 示例1: 搜索业务查询
    print("🔍 示例1: 搜索注册相关的业务查询")
    print("-" * 70)
    results = query_engine.search_business_queries("用户注册量统计", k=2)
    query_engine.pretty_print_results(results)
    
    # 示例2: 搜索表结构
    print("🔍 示例2: 搜索订单表结构")
    print("-" * 70)
    results = query_engine.search_table_schemas("订单表有哪些字段", k=2)
    query_engine.pretty_print_results(results)
    
    # 示例3: 混合搜索
    print("🔍 示例3: 混合搜索还款相关内容")
    print("-" * 70)
    results = query_engine.search_all("还款金额统计和还款表结构", k=3)
    query_engine.pretty_print_results(results)
    
    # 示例4: 带分数的搜索
    print("🔍 示例4: 搜索风控审批相关（带相似度分数）")
    print("-" * 70)
    results_with_scores = query_engine.search_with_score("风控审批流程", k=3)
    for i, (doc, score) in enumerate(results_with_scores, 1):
        print(f"\n结果 {i} (相似度分数: {score:.4f}):")
        print(f"类型: {doc.metadata.get('type')}")
        if doc.metadata.get('type') == 'business_query':
            print(f"查询名称: {doc.metadata.get('query_name')}")
        else:
            print(f"表名: {doc.metadata.get('table_name')}")
        print(f"内容预览: {doc.page_content[:200]}...")
        print("-" * 70)
    
    print("\n✅ 示例查询完成!")


if __name__ == "__main__":
    main()

