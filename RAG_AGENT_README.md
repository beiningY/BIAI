# 🤖 简单 RAG 智能体使用指南

基于 LangChain `create_agent` 的最简单 RAG 智能体实现。

参考文档: [LangChain RAG Tutorial](https://docs.langchain.com/oss/python/langchain/rag)

## 📋 特性

✅ **简单易用** - 不到 200 行代码实现完整 RAG 智能体  
✅ **标准实现** - 严格遵循 LangChain 最新最佳实践  
✅ **工具检索** - 使用 `@tool` 装饰器创建检索工具  
✅ **智能对话** - 支持多轮对话和上下文理解  
✅ **流式输出** - 可视化智能体思考和工具调用过程  

## 🚀 快速开始

### 1. 确保已构建知识库

```bash
# 如果还没有构建知识库，先运行
python build_knowledge_base.py
```

### 2. 运行智能体

```bash
python simple_rag_agent.py
```

### 3. 与智能体对话

程序会先运行几个测试问题，然后进入交互模式：

```
💬 进入交互模式（输入 'quit' 或 'exit' 退出）
======================================================================

👤 你的问题: sgo_orders表有什么作用？

🔧 调用工具: retrieve_database_knowledge
📝 工具输入: sgo_orders表结构

📊 工具输出:
[显示检索到的相关信息]

======================================================================
🤖 智能体回答:
======================================================================
sgo_orders 是贷款订单核心表，主要用于记录所有贷款申请、审核、放款、还款信息...
```

## 📖 工作原理

### 架构图

```
用户问题
   ↓
智能体 (Agent)
   ↓
检索工具 (retrieve_database_knowledge)
   ↓
向量数据库 (Chroma)
   ↓
检索结果
   ↓
LLM 生成回答
   ↓
返回用户
```

### 核心组件

#### 1. 检索工具 (Tool)

```python
@tool
def retrieve_database_knowledge(query: str) -> str:
    """从数据库知识库中检索相关信息"""
    retrieved_docs = vectorstore.similarity_search(query, k=3)
    # 格式化并返回结果
    return formatted_results
```

#### 2. 智能体 (Agent)

```python
agent = create_tool_calling_agent(
    llm=llm,              # 语言模型
    tools=tools,          # 可用工具列表
    prompt=prompt         # 系统提示词
)
```

#### 3. 执行器 (AgentExecutor)

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,         # 显示详细过程
    max_iterations=5      # 最大迭代次数
)
```

## 🎯 示例问题

### 表结构相关

```
- sgo_orders 表有哪些字段？
- approval_info 表的作用是什么？
- WhatsApp 消息记录在哪个表？
- users 表和 sgo_orders 表有什么关系？
```

### 业务查询相关

```
- 如何按小时统计用户注册量？
- 如何查询新老客申请量？
- 还款金额统计的查询是怎么写的？
- 如何统计 WhatsApp 消息发送量？
```

### 综合问题

```
- 要统计放款金额需要用到哪些表和字段？
- 风控审批流程涉及哪些表？
- 消息发送成本如何计算？
```

## ⚙️ 配置选项

### 修改模型

```python
agent = SimpleRAGAgent(
    model_name="openai/gpt-4o",      # 使用更强大的模型
    temperature=0.7                   # 增加创造性
)
```

### 修改检索数量

在 `_create_retrieval_tool` 方法中修改：

```python
retrieved_docs = self.vectorstore.similarity_search(query, k=5)  # 检索5个文档
```

### 修改系统提示词

在 `_create_agent` 方法中修改 prompt 的 system 消息。

## 🔍 调试技巧

### 1. 查看详细日志

智能体默认开启 `verbose=True`，会显示：
- 工具调用过程
- 工具输入输出
- 思考步骤

### 2. 检查检索结果

在工具函数中添加打印：

```python
@tool
def retrieve_database_knowledge(query: str) -> str:
    retrieved_docs = self.vectorstore.similarity_search(query, k=3)
    print(f"检索到 {len(retrieved_docs)} 个文档")  # 调试输出
    # ...
```

### 3. 测试单个工具

```python
# 直接测试检索工具
agent = SimpleRAGAgent()
result = agent.tools[0].invoke({"query": "sgo_orders表"})
print(result)
```

## 📚 进阶功能

### 1. 添加对话历史

```python
# 修改 chat 方法支持历史记录
def chat_with_history(self, question: str, chat_history: list = None):
    response = self.agent.invoke({
        "input": question,
        "chat_history": chat_history or []
    })
    return response["output"]
```

### 2. 添加更多工具

```python
@tool
def query_database_directly(sql: str) -> str:
    """直接执行SQL查询"""
    # 实现SQL执行逻辑
    pass

# 添加到工具列表
self.tools = [
    self._create_retrieval_tool(),
    query_database_directly
]
```

### 3. 结构化输出

```python
from langchain_core.pydantic_v1 import BaseModel, Field

class DatabaseAnswer(BaseModel):
    answer: str = Field(description="回答内容")
    tables: list[str] = Field(description="涉及的表")
    confidence: float = Field(description="置信度")

# 使用结构化输出
llm_with_structure = self.llm.with_structured_output(DatabaseAnswer)
```

## 🆚 与传统方法对比

### RAG 智能体 (本方案)

✅ 自动决策是否检索  
✅ 支持多次检索  
✅ 可以处理复杂查询  
✅ 灵活扩展工具  

### 简单 RAG 链

✅ 更快（单次 LLM 调用）  
✅ 更可控  
⚠️ 总是执行检索  
⚠️ 无法多次检索  

## 🎓 学习资源

- [LangChain RAG 官方教程](https://docs.langchain.com/oss/python/langchain/rag)
- [LangChain Agents 文档](https://docs.langchain.com/oss/python/langchain/agents)
- [Tools 使用指南](https://docs.langchain.com/oss/python/langchain/tools)

## ❓ 常见问题

### Q: 智能体不调用工具？

A: 检查：
1. 工具描述是否清晰
2. 问题是否需要检索
3. 尝试更强大的模型

### Q: 检索结果不准确？

A: 优化方向：
1. 调整检索数量 `k`
2. 改进文档分块策略
3. 使用更好的 embedding 模型

### Q: 如何支持中文？

A: 代码已支持中文：
- 系统提示词使用中文
- 工具描述使用中文
- 支持中文问答

## 📝 下一步

1. ✅ 已完成：基础 RAG 智能体
2. 🚀 可以尝试：添加更多工具
3. 🎯 高级功能：集成到 Web 应用
4. 📊 优化方向：添加评估指标

---

💡 **提示**: 这是最简单的实现。生产环境建议使用 LangGraph 实现更复杂的工作流。

