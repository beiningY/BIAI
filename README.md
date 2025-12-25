# 数据库知识库构建系统

基于 LangChain 和 Chroma 的向量数据库知识库系统，用于存储和查询数据库相关的业务需求和Schema信息。

## 📋 功能特性

### 两部分知识库内容

1. **业务查询知识库** (`query_business_requirements.json`)
   - 按每个查询（query）分块存储
   - 包含查询ID、名称、业务需求描述和SQL语句
   - 支持根据业务需求搜索对应的SQL实现

2. **数据库Schema知识库** (`schema.sql`)
   - 按每个数据库表分块存储
   - 包含表名、字段定义、数据类型、注释说明
   - 支持根据业务需求查找相关表结构

## 🚀 快速开始

### 1. 安装依赖

```bash
# 推荐：使用 uv（读取 pyproject.toml）
pip install uv
uv sync
```

### 2. 配置环境变量

设置 OpenAI API Key（用于生成向量嵌入）：

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. 构建知识库

运行构建脚本，将数据导入Chroma向量数据库：

```bash
python build_knowledge_base_enhanced.py
```

这将：
- 读取 `query_business_requirements.json` 文件中的所有查询
- 解析 `schema.sql` 文件中的所有表结构
- 为每个查询和表创建独立的文档块
- 生成向量嵌入并存储到 `./chroma_db` 目录

输出示例：
```
==================================================================
🚀 开始构建知识库
==================================================================

📖 正在加载查询文件: query_business_requirements.json
✅ 成功加载 11 个查询文档

📖 正在加载Schema文件: schema.sql
✅ 成功加载 8 个表结构文档

📊 文档统计:
  - 查询文档: 11 个
  - Schema文档: 8 个
  - 总计: 19 个

✅ 向量数据库构建完成!
```

### 4. 查询知识库

运行查询示例脚本：

```bash
python query_knowledge_base.py
```

## 📖 使用方法

### 基础查询

```python
from query_knowledge_base import KnowledgeBaseQuery

# 初始化查询器
query_engine = KnowledgeBaseQuery(
    persist_directory="./chroma_db",
    collection_name="database_knowledge"
)

# 搜索业务查询
results = query_engine.search_business_queries("用户注册量统计", k=3)

# 搜索表结构
results = query_engine.search_table_schemas("订单表有什么字段", k=3)

# 混合搜索
results = query_engine.search_all("还款相关的查询和表", k=5)
```

### 高级查询（带相似度分数）

```python
# 获取相似度分数
results_with_scores = query_engine.search_with_score("风控审批", k=5)

for doc, score in results_with_scores:
    print(f"相似度: {score:.4f}")
    print(f"内容: {doc.page_content[:100]}...")
```

### 使用过滤器

```python
# 只搜索业务查询类型
results = query_engine.vectorstore.similarity_search(
    "注册统计",
    k=5,
    filter={"type": "business_query"}
)

# 只搜索特定表
results = query_engine.vectorstore.similarity_search(
    "订单字段",
    k=5,
    filter={"table_name": "sgo_orders"}
)
```

## 📊 数据结构

### 业务查询文档结构

```python
{
    "page_content": """
        查询ID: 408
        查询名称: Register by hour
        
        业务需求:
        按小时统计用户注册数量...
        
        SQL语句:
        WITH dtl AS ...
    """,
    "metadata": {
        "source": "query_business_requirements",
        "query_id": "408",
        "query_name": "Register by hour",
        "type": "business_query",
        "has_sql": True,
        "business_requirement": "按小时统计用户注册数量..."
    }
}
```

### Schema文档结构

```python
{
    "page_content": """
        表名: users
        
        表说明: 用户信息表，存储注册用户的基本信息和验证状态
        
        字段信息:
          - id: int unsigned NOT NULL AUTO_INCREMENT  // 用户唯一ID
          - mobile_phone: varchar(255)  // 手机号码
          ...
        
        完整DDL:
        CREATE TABLE `users` ...
    """,
    "metadata": {
        "source": "database_schema",
        "table_name": "users",
        "table_comment": "用户信息表，存储注册用户的基本信息和验证状态",
        "type": "table_schema",
        "database": "singa_bi"
    }
}
```

## 🔧 配置选项

### build_knowledge_base.py

```python
builder = KnowledgeBaseBuilder(
    persist_directory="./chroma_db",  # 数据库存储目录
    collection_name="database_knowledge"  # 集合名称
)

vectorstore = builder.build_vector_store(
    query_json="query_business_requirements.json",
    schema_sql="schema.sql",
    chunk_size=2000,  # 文档分块大小
    chunk_overlap=200  # 分块重叠大小
)
```

## 📁 文件说明

```
数据库自动化/
├── build_knowledge_base_enhanced.py  # 知识库构建脚本（增强版）
├── query_kb_simple.py                # 知识库查询示例
├── pyproject.toml                    # 依赖与项目元数据（替代 requirements.txt）
├── smithery.yaml                     # Smithery 运行时声明
├── src/biai_server/server.py         # MCP Server（create_server 工厂函数）
├── query_business_requirements.json  # 业务查询数据
├── schema.sql                   # 数据库Schema
└── chroma_db/                   # Chroma向量数据库（自动生成）
```

## 🧩 Smithery 部署入口

- **server factory**: `biai_server.server:create_server`
- **本地运行（兼容入口）**:

```bash
uv run python main.py
```

## 🎯 典型应用场景

1. **根据业务需求查找SQL**
   - 输入：业务需求描述
   - 输出：相关的SQL查询和实现

2. **根据业务场景查找相关表**
   - 输入：业务场景（如"用户注册"、"订单管理"）
   - 输出：相关的数据库表结构和字段说明

3. **SQL开发辅助**
   - 查找相似的查询实现作为参考
   - 了解表结构和字段含义

4. **数据分析支持**
   - 快速找到特定指标的计算方法
   - 理解数据来源和定义

## ⚙️ 高级功能

### 自定义嵌入模型

如果想使用其他嵌入模型（如本地模型），可以修改：

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

self.embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
```

### 增量更新

如果需要添加新的查询或表，重新运行构建脚本即可：

```bash
python build_knowledge_base.py
```

### 性能优化

对于大规模数据，可以调整：
- `chunk_size`: 减小以获得更精细的检索
- `k`: 增大以获得更多候选结果
- 使用 `max_marginal_relevance_search` 提高结果多样性

```python
results = vectorstore.max_marginal_relevance_search(
    query="用户注册",
    k=5,
    fetch_k=20  # 先获取20个候选，再选择5个多样性高的
)
```

## 🐛 故障排除

### 1. OpenAI API错误

确保设置了正确的API Key：
```bash
echo $OPENAI_API_KEY
```

### 2. 编码问题

确保所有文件使用UTF-8编码

### 3. 依赖版本冲突

如遇到版本冲突，可以使用虚拟环境：
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 📝 注意事项

1. **OpenAI API成本**: 构建知识库会调用OpenAI API生成嵌入，会产生少量费用
2. **数据隐私**: 如有敏感数据，建议使用本地嵌入模型
3. **增量更新**: 每次构建会覆盖之前的数据库
4. **查询质量**: 问题描述越清晰准确，检索结果越好

## 📄 License

MIT License

