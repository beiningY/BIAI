#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的 RAG 智能体 - 使用 LangChain create_agent
基于 LangChain 最新最佳实践
参考: https://docs.langchain.com/oss/python/langchain/rag

功能：
- 使用 create_agent 创建智能体
- 通过工具（tool）检索向量数据库
- 回答关于数据库业务需求和表结构的问题
"""

import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv()

# LangChain 核心组件

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# API 配置
OPENAI_API_KEY = os.getenv("openrouter_api_key") or os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("url_openrouter", "https://api.openai.com/v1")

if not OPENAI_API_KEY:
    print("❌ 错误: 未设置 OPENAI_API_KEY 环境变量")
    sys.exit(1)


class SimpleRAGAgent:
    """
    简单的 RAG 智能体
    使用 LangChain create_agent 和向量数据库
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "database_knowledge",
        model_name: str = "anthropic/claude-opus-4.5",
        temperature: float = 0
    ):
        """
        初始化 RAG 智能体
        
        Args:
            persist_directory: 向量数据库目录
            collection_name: 集合名称
            model_name: LLM 模型名称
            temperature: 温度参数（0=确定性，1=创造性）
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        
        print("🚀 初始化 RAG 智能体...")
        
        # 1. 初始化 Embeddings
        print("  📊 加载 Embedding 模型...")
        embedding_kwargs = {
            "model": "text-embedding-3-large",
            "api_key": OPENAI_API_KEY,
        }
        if "openrouter" in OPENAI_BASE_URL.lower():
            embedding_kwargs["base_url"] = OPENAI_BASE_URL
        
        self.embeddings = OpenAIEmbeddings(**embedding_kwargs)
        
        # 2. 加载向量数据库
        print(f"  💾 加载向量数据库: {persist_directory}")
        self.vectorstore = self._load_vectorstore()
        
        if not self.vectorstore:
            raise ValueError("无法加载向量数据库，请先运行 build_knowledge_base.py")
        
        # 3. 初始化 LLM
        print(f"  🤖 初始化 LLM: {model_name}")
        llm_kwargs = {
            "model": model_name,
            "temperature": temperature,
            "api_key": OPENAI_API_KEY,
        }
        if "openrouter" in OPENAI_BASE_URL.lower():
            llm_kwargs["base_url"] = OPENAI_BASE_URL
        
        self.llm = ChatOpenAI(**llm_kwargs)
        
        # 4. 创建检索工具
        print("  🛠️  创建检索工具...")
        self.tools = [self._create_retrieval_tool()]
        
        # 5. 创建智能体
        print(" 创建智能体...")
        self.agent = self._create_agent()
        
        print(" RAG 智能体初始化完成！\n")
    
    def _load_vectorstore(self) -> Chroma:
        """加载向量数据库"""
        try:
            vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory)
            )
            doc_count = vectorstore._collection.count()
            print(f"  成功加载 {doc_count} 个文档")
            return vectorstore
        except Exception as e:
            print(f"  加载失败: {e}")
            return None
    
    def _create_retrieval_tool(self):
        """
        创建检索工具
        参考: https://docs.langchain.com/oss/python/langchain/rag
        """
        @tool
        def retrieve_database_knowledge(query: str) -> str:
            """
            从数据库知识库中检索相关信息。
            
            这个工具可以回答关于：
            - 数据库表结构和字段信息
            - SQL查询和业务需求
            - 表之间的关系
            
            Args:
                query: 要检索的问题或关键词
                
            Returns:
                检索到的相关信息
            """
            try:
                # 执行相似度搜索
                retrieved_docs = self.vectorstore.similarity_search(query, k=3)
                
                if not retrieved_docs:
                    return "未找到相关信息。"
                
                # 格式化检索结果
                result_parts = []
                for i, doc in enumerate(retrieved_docs, 1):
                    metadata = doc.metadata
                    doc_type = metadata.get('type', '未知类型')
                    
                    if doc_type == 'business_query':
                        title = f"查询 #{metadata.get('query_id')}: {metadata.get('query_name')}"
                        tables = metadata.get('tables', '')
                        if tables:
                            title += f" (涉及表: {tables})"
                    else:
                        table_name = metadata.get('table_name', '未知表')
                        table_comment = metadata.get('table_comment', '')
                        title = f"表结构: {table_name} - {table_comment}"
                    
                    result_parts.append(f"【检索结果 {i}】{title}\n{doc.page_content}\n")
                
                return "\n".join(result_parts)
                
            except Exception as e:
                return f"检索出错: {str(e)}"
        
        return retrieve_database_knowledge
    
    def _create_agent(self):
        """
        创建智能体（使用 LangChain v1.x 方式）
        """
        # 创建智能体
        agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt="""你是一个专业的数据库助手，负责回答关于数据库表结构和SQL查询的问题。

你的职责：
1. 使用检索工具 (retrieve_database_knowledge) 查找相关信息
2. 基于检索到的信息准确回答用户问题
3. 如果信息不足，可以多次使用工具检索
4. 用清晰、专业的中文回答

注意事项：
- 对于表结构问题，详细说明字段含义
- 对于SQL问题，解释业务需求和实现逻辑
- 如果检索不到信息，诚实告知用户"""
        )

        return agent
    
    def chat(self, question: str) -> str:
        """
        与智能体对话
        
        Args:
            question: 用户问题
            
        Returns:
            智能体回答
        """
        try:
            print(f"开始调用智能体")
            response = self.agent.invoke({"messages": [{"role": "user", "content": question}]})
            last_message = response["messages"][-1]  
            answer = last_message.content
            return answer
        except Exception as e:
            return f"发生错误: {str(e)}"
    
    def stream_chat(self, question: str):
        """
        流式对话（打印中间步骤）
        
        Args:
            question: 用户问题
        """
        print(f"\n{'='*70}")
        print(f"💬 用户问题: {question}")
        print(f"{'='*70}\n")
        
        try:
            for step in self.agent.stream({"input": question}):
                # 打印中间步骤
                if "actions" in step:
                    for action in step["actions"]:
                        print(f"🔧 调用工具: {action.tool}")
                        print(f"📝 工具输入: {action.tool_input}")
                        print()
                
                if "steps" in step:
                    for step_detail in step["steps"]:
                        print(f"📊 工具输出:")
                        output = step_detail.observation
                        # 限制输出长度
                        if len(output) > 500:
                            output = output[:500] + "...\n[输出已截断]"
                        print(output)
                        print()
                
                if "output" in step:
                    print(f"\n{'='*70}")
                    print(f"🤖 智能体回答:")
                    print(f"{'='*70}")
                    print(step["output"])
                    print()
                    
        except Exception as e:
            print(f"❌ 错误: {e}")


def main():
    """主函数 - 演示智能体使用"""
    
    print("\n" + "="*70)
    print("RAG 智能体")
    print("   基于 LangChain create_agent")
    print("="*70 + "\n")
    
    # 初始化智能体
    try:
        agent = SimpleRAGAgent(
            persist_directory="./chroma_db",
            collection_name="database_knowledge",
            model_name="openai/gpt-4o-mini",  # OpenRouter
            temperature=0
        )
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        print("\n💡 请先运行 build_knowledge_base.py 构建知识库")
        return 1
    
    # 测试问题列表
    test_questions = [
        "如何查询2025.12.11发生还款且手机号验证通过的借款用户，截止到昨天的待还金额以及还款金额，还款率"
    ]
    
    print("🧪 开始测试智能体...\n")
    
    # 依次测试每个问题
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'#'*70}")
        print(f"# 测试 {i}/{len(test_questions)}")
        print(f"问题: {question}")
        print(f"{'#'*70}")
        
        # 使用流式对话展示过程
        response = agent.chat(question)
        print(f"回答: {response}")



if __name__ == "__main__":
    sys.exit(main())

