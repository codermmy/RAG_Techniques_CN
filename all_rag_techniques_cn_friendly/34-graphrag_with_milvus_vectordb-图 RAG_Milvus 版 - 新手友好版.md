# 🌟 新手入门：Graph RAG（图 RAG）系统 - Milvus 版

> **💡 给新手的说明**
> - **难度等级**：⭐⭐⭐⭐⭐（高级）
> - **预计时间**：90-120 分钟
> - **前置知识**：了解 RAG 基础、向量数据库概念，有图数据库基础更佳
> - **学习目标**：理解 Graph RAG 的核心思想，掌握使用 Milvus 构建支持多跳查询的 RAG 系统

---

## 📖 核心概念理解

### 什么是 Graph RAG？

**Graph RAG**（图检索增强生成）是将**知识图谱**与**RAG**结合的技术。它不仅检索相关文档，还能理解文档中实体之间的关系，从而回答需要多步推理的复杂问题。

### 🍕 通俗理解：社交网络比喻

想象你要找"你老师的儿子的同学"：

1. **传统 RAG**：
   - 搜索包含"老师"、"儿子"、"同学"关键词的文档
   - 可能找到相关信息，但无法系统性地找出答案

2. **Graph RAG**：
   - 找到"你"→"老师"（师生关系）
   - 找到"老师"→"儿子"（父子关系）
   - 找到"儿子"→"同学"（同学关系）
   - 沿着关系路径找到答案！

### 📊 传统 RAG vs Graph RAG

| 特性 | 传统 RAG | Graph RAG |
|------|---------|-----------|
| **擅长问题** | 单跳查询 | 多跳查询 |
| **示例问题** | "谁发明了电话？" | "发明电话的人的老师是谁？" |
| **数据结构** | 文本块 | 实体 + 关系图 |
| **检索方式** | 相似度搜索 | 图遍历 + 相似度 |

### 🔑 核心组件解释

| 组件 | 作用 | 生活比喻 |
|------|------|----------|
| **实体** | 图中的人/事/物节点 | 社交网络中的用户 |
| **关系** | 连接实体的边 | 用户之间的关注/好友关系 |
| **Milvus** | 向量数据库，存储和检索 | 超大的关系数据库 |
| **三元组** | [主语，谓语，宾语] 结构 | "A 是 B 的父亲" |
| **多跳查询** | 需要多次关系遍历的查询 | "你爷爷的老师的儿子" |

### 🎯 Graph RAG 解决的问题

**传统 RAG 难以回答的问题**：
- "Euler 老师的儿子做出了什么贡献？"（需要 2 跳）
- "发现 DNA 结构的人在哪所大学工作？"（需要 2 跳）
- "写《时间简史》的物理学家获得过什么奖？"（需要 2 跳）

**Graph RAG 可以回答**：
- ✅ 通过实体关系遍历，系统性地找到答案
- ✅ 理解间接关系和隐含连接

---

## 🏗️ Graph RAG 架构

### 四阶段方法

```
┌─────────────────────────────────────────────────────────┐
│                    Graph RAG 流程                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. 离线数据准备                                        │
│     文本 → 提取实体和关系 → 存储到 Milvus               │
│                                                         │
│  2. 查询时检索                                          │
│     查询 → 向量搜索 → 找到相似实体/关系                 │
│                                                         │
│  3. 子图扩展                                            │
│     检索到的实体 → 扩展邻居 → 发现间接关系              │
│                                                         │
│  4. LLM 重排序                                          │
│     候选关系 → LLM 筛选 → 最终答案                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 数据模型

我们将数据分为三类：

1. **实体（Entities）**：人、地点、概念等
   - 例如："牛顿"、"剑桥大学"、"万有引力"

2. **关系（Relations）**：完整的三元组
   - 例如：["牛顿", "就读于", "剑桥大学"]
   - 例如：["牛顿", "发现了", "万有引力"]

3. **段落（Passages）**：原始文本
   - 提供上下文和详细信息

---

## 🛠️ 第一步：环境准备

### 💻 完整代码

```python
# ============================================
# 安装所需的包
# ============================================
!pip install --upgrade --quiet pymilvus numpy scipy langchain langchain-core langchain-openai tqdm

# 如果你使用 Google Colab，安装后需要重启运行时
```

> **💡 代码解释**
> - `pymilvus`：Milvus 向量数据库的 Python 客户端
> - `numpy`、`scipy`：数值计算库
> - `langchain`：LLM 应用框架
> - `tqdm`：进度条显示

> **⚠️ 新手注意**
> - Milvus 可以通过 [Zilliz Cloud](https://cloud.zilliz.com/) 免费试用
> - 也可以在本地部署 Milvus（需要 Docker）

---

## 🔑 第二步：配置 API 和连接 Milvus

### 💻 完整代码

```python
# ============================================
# 导入必要的库
# ============================================
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
from pymilvus import MilvusClient
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from tqdm import tqdm

# ============================================
# 配置 API 密钥和 Milvus 连接
# ============================================
import os

# OpenAI API 密钥
os.environ["OPENAI_API_KEY"] = "sk-your-key-here"

# Milvus 连接（使用 Zilliz Cloud）
milvus_client = MilvusClient(
    uri="YOUR_ZILLIZ_PUBLIC_ENDPOINT",  # Zilliz Cloud 端点
    token="YOUR_ZILLIZ_TOKEN"           # Zilliz Cloud Token
)

# 初始化 LLM 和 Embedding 模型
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,  # 温度为 0，输出更稳定
)
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)
```

> **💡 代码解释**
> - `MilvusClient` 是连接 Milvus 的客户端
> - `uri` 和 `token` 从 Zilliz Cloud 控制台获取
> - `gpt-4o` 用于复杂的关系推理

> **⚠️ 新手注意**
> - 需要在 [Zilliz Cloud](https://cloud.zilliz.com/) 注册账号
> - 注册后创建集群，获取 endpoint 和 token
> - 免费套餐足够学习使用

---

## 📊 第三步：理解数据结构（三元组）

### 📖 什么是三元组？

**三元组**是知识图谱的基本单位，格式为 `[主语，谓语，宾语]`：

```
["Jakob Bernoulli", "was the older brother of", "Johann Bernoulli"]
   ↓              ↓                        ↓
 主语 (Subject)   谓语 (Predicate)         宾语 (Object)
```

### 💻 示例数据

```python
# 示例：伯努利家族和 Euler 的关系
nano_dataset = [
    {
        "passage": "Jakob Bernoulli (1654–1705): 他是伯努利家族中最早在数学界崭露头角的成员之一。他对微积分做出了重大贡献，特别是概率论的发展。他以伯努利数和伯努利定理闻名。他是 Johann Bernoulli 的哥哥，两人有着复杂的关系，既有合作也有竞争。",
        "triplets": [
            ["Jakob Bernoulli", "made significant contributions to", "calculus"],
            ["Jakob Bernoulli", "is known for", "the Bernoulli numbers"],
            ["Jakob Bernoulli", "was the older brother of", "Johann Bernoulli"],
        ]
    },
    {
        "passage": "Johann Bernoulli (1667–1748): 他是 Jakob 的弟弟，同样在数学领域取得了卓越成就。他在微积分和力学方面有重要贡献。他是 Leonhard Euler 的老师之一。",
        "triplets": [
            ["Johann Bernoulli", "was the younger brother of", "Jakob Bernoulli"],
            ["Johann Bernoulli", "was a teacher of", "Leonhard Euler"],
        ]
    },
    {
        "passage": "Leonhard Euler (1707–1783): 他是历史上最多产的数学家之一。他在 Johann Bernoulli 的指导下学习。他的儿子 Johann Euler 也是一位数学家。",
        "triplets": [
            ["Leonhard Euler", "was a student of", "Johann Bernoulli"],
            ["Leonhard Euler", "had a son named", "Johann Euler"],
        ]
    },
]
```

> **💡 关键理解**
> - 每个 passage 是一段文本
> - triplets 是从文本中提取的关系
> - 通过这些关系，我们可以回答"Euler 的老师是谁？"这类问题

---

## 🗄️ 第四步：创建 Milvus 集合

### 📖 这是什么？

Milvus 使用**集合（Collection）**来组织数据。我们需要三个集合：
1. **entities**：存储实体
2. **relations**：存储关系（三元组）
3. **passages**：存储原始文本

### 💻 完整代码

```python
# ============================================
# 创建三个 Milvus 集合
# ============================================

# 集合配置
collection_configs = {
    "entities": {"dimension": 512},      # 实体向量维度
    "relations": {"dimension": 512},      # 关系向量维度
    "passages": {"dimension": 512},       # 段落向量维度
}

# 创建集合
for name, config in collection_configs.items():
    if not milvus_client.has_collection(name):
        milvus_client.create_collection(
            collection_name=name,
            dimension=config["dimension"],
        )
        print(f"✓ 集合 '{name}' 创建完成")
    else:
        print(f"⚠ 集合 '{name}' 已存在")
```

> **💡 代码解释**
> - `dimension=512` 对应 `text-embedding-3-small` 的输出维度
> - 每个集合独立存储不同类型的信息
> - 集合类似数据库中的"表"

---

## 🔄 第五步：数据索引（离线处理）

### 📖 这是什么？

将三元组和段落转换为向量，存储到 Milvus 中。

### 💻 完整代码

```python
# ============================================
# 数据索引函数
# ============================================
def index_dataset(dataset, milvus_client):
    """
    将数据集中的实体、关系和段落索引到 Milvus 中
    
    参数：
        dataset: 包含 passage 和 triplets 的数据集
        milvus_client: Milvus 客户端
    """
    entity_set = set()  # 用集合去重
    relation_data = []  # 关系数据列表
    passage_data = []   # 段落数据列表
    
    # ========== 提取实体和关系 ==========
    for item in tqdm(dataset, desc="处理文档"):
        passage = item["passage"]
        triplets = item["triplets"]
        
        # 提取三元组中的实体
        for subj, pred, obj in triplets:
            entity_set.add(subj)  # 主语是实体
            entity_set.add(obj)   # 宾语是实体
            relation_data.append({
                "subject": subj,
                "predicate": pred,
                "object": obj,
            })
        
        # 保存段落
        passage_data.append({"text": passage})
    
    # ========== 生成向量并存储 ==========
    entities = list(entity_set)
    
    # 实体向量
    entity_embeddings = embedding_model.embed_documents(entities)
    for i, entity in enumerate(entities):
        milvus_client.insert(
            collection_name="entities",
            data=[{"vector": entity_embeddings[i], "text": entity}]
        )
    
    # 关系向量（将三元组转为文本再生成向量）
    relation_texts = [f"{r['subject']} {r['predicate']} {r['object']}" for r in relation_data]
    relation_embeddings = embedding_model.embed_documents(relation_texts)
    for i, rel in enumerate(relation_data):
        milvus_client.insert(
            collection_name="relations",
            data=[{
                "vector": relation_embeddings[i],
                "text": relation_texts[i],
                "subject": rel["subject"],
                "object": rel["object"]
            }]
        )
    
    # 段落向量
    passage_texts = [p["text"] for p in passage_data]
    passage_embeddings = embedding_model.embed_documents(passage_texts)
    for i, passage in enumerate(passage_data):
        milvus_client.insert(
            collection_name="passages",
            data=[{"vector": passage_embeddings[i], "text": passage["text"]}]
        )
    
    print(f"✓ 索引完成：{len(entities)} 个实体，{len(relation_data)} 个关系，{len(passage_data)} 个段落")
```

---

## 🔍 第六步：查询处理

### 📖 这是什么？

将用户查询转换为向量，检索相关的实体和关系。

### 💻 完整代码

```python
# ============================================
# 查询处理函数
# ============================================
def retrieve_initial(query, milvus_client, top_k=5):
    """
    初始检索：找到与查询相似的实体和关系
    
    参数：
        query: 用户查询
        milvus_client: Milvus 客户端
        top_k: 返回的最相似结果数量
    
    返回：
        similar_entities, similar_relations
    """
    # 生成查询向量
    query_embedding = embedding_model.embed_query(query)
    
    # 检索相似实体
    entity_results = milvus_client.search(
        collection_name="entities",
        data=[query_embedding],
        limit=top_k,
    )
    similar_entities = [hit["entity"]["text"] for hit in entity_results[0]]
    
    # 检索相似关系
    relation_results = milvus_client.search(
        collection_name="relations",
        data=[query_embedding],
        limit=top_k * 2,  # 关系可以多一些
    )
    similar_relations = [
        (hit["entity"]["subject"], hit["entity"]["object"]) 
        for hit in relation_results[0]
    ]
    
    return similar_entities, similar_relations
```

---

## 🔗 第七步：子图扩展（核心）

### 📖 这是什么？

找到初始实体/关系后，扩展它们的"邻居"，发现间接连接。这是 Graph RAG 的核心！

### 💻 完整代码

```python
# ============================================
# 子图扩展函数
# ============================================
def expand_subgraph(entities, relations, milvus_client, num_hops=1):
    """
    扩展子图：找到实体的邻居和关系的连接
    
    参数：
        entities: 初始实体列表
        relations: 初始关系列表
        milvus_client: Milvus 客户端
        num_hops: 扩展跳数（1=直接邻居，2=邻居的邻居）
    
    返回：
        扩展后的关系路径列表
    """
    # ========== 获取所有关系 ==========
    all_relations = milvus_client.query(
        collection_name="relations",
        output_fields=["subject", "predicate", "object"],
    )
    
    # 构建邻接表（图的表示方式）
    graph = defaultdict(list)
    for rel in all_relations:
        subj = rel["entity"]["subject"]
        pred = rel["entity"]["predicate"]
        obj = rel["entity"]["object"]
        # 双向图（关系可以反向）
        graph[subj].append((pred, obj))
        graph[obj].append((f"reverse of {pred}", subj))
    
    # ========== BFS 扩展 ==========
    extended_paths = []
    queue = [(entity, 0) for entity in entities]  # (当前实体，当前跳数)
    visited = set(entities)
    
    while queue:
        current, hops = queue.pop(0)
        
        if hops > num_hops:
            continue
        
        # 获取当前实体的邻居
        neighbors = graph.get(current, [])
        for predicate, neighbor in neighbors:
            if neighbor not in visited or hops == 0:
                extended_paths.append((current, predicate, neighbor))
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, hops + 1))
    
    return extended_paths
```

> **💡 图扩展原理**
> ```
> 初始：找到 "Euler"
>   ↓
> 1 跳扩展：Euler → 老师 → Johann Bernoulli
>   ↓
> 2 跳扩展：Johann Bernoulli → 哥哥 → Jakob Bernoulli
> 
> 最终路径：Euler ← 老师 ← Johann ← 哥哥 ← Jakob
> ```

---

## 🎯 第八步：LLM 重排序

### 📖 这是什么？

使用 LLM 智能筛选最相关的关系，生成最终答案。

### 💻 完整代码

```python
# ============================================
# LLM 重排序函数
# ============================================
def rerank_with_llm(query, candidate_paths, llm):
    """
    使用 LLM 重排序候选关系路径
    
    参数：
        query: 用户查询
        candidate_paths: 候选关系路径列表
        llm: 语言模型
    
    返回：
        最相关的路径和最终答案
    """
    # 构建提示词
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个知识图谱推理专家。请分析以下关系路径，找出最能回答用户查询的路径。"),
        ("human", """
查询：{query}

候选关系路径：
{paths}

请：
1. 分析每条路径与查询的相关性
2. 选择最相关的路径
3. 基于该路径生成答案

最相关路径：
答案：""")
    ])
    
    # 格式化路径
    paths_text = "\n".join([
        f"{i+1}. {subj} --{pred}--> {obj}" 
        for i, (subj, pred, obj) in enumerate(candidate_paths)
    ])
    
    # 调用 LLM
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "query": query,
        "paths": paths_text
    })
    
    return result
```

---

## 🚀 第九步：完整的 Graph RAG 流程

### 💻 完整代码

```python
# ============================================
# 完整的 Graph RAG 流程
# ============================================
def graph_rag_query(query, milvus_client, llm):
    """
    完整的 Graph RAG 查询流程
    
    参数：
        query: 用户查询
        milvus_client: Milvus 客户端
        llm: 语言模型
    
    返回：
        最终答案
    """
    print(f"\n🔍 处理查询：{query}")
    
    # 步骤 1：初始检索
    print("📊 步骤 1：初始检索...")
    entities, relations = retrieve_initial(query, milvus_client)
    print(f"   找到 {len(entities)} 个实体，{len(relations)} 个关系")
    
    # 步骤 2：子图扩展
    print("🔗 步骤 2：子图扩展...")
    expanded_paths = expand_subgraph(entities, relations, milvus_client, num_hops=2)
    print(f"   扩展得到 {len(expanded_paths)} 条路径")
    
    # 步骤 3：LLM 重排序
    print("🤖 步骤 3：LLM 重排序...")
    result = rerank_with_llm(query, expanded_paths, llm)
    
    print("\n✅ 答案生成完成")
    return result


# ============================================
# 测试查询
# ============================================
query = "Who were the teachers and family members of Leonhard Euler?"
answer = graph_rag_query(query, milvus_client, llm)
print(f"\n问题：{query}")
print(f"答案：{answer}")
```

---

## ⚠️ 常见问题与调试

### Q1: Milvus 和 Zilliz Cloud 有什么区别？

**解释**：
- **Milvus**：开源向量数据库，可以自己部署
- **Zilliz Cloud**：基于 Milvus 的全托管云服务

**建议**：
- 学习/测试：用 Zilliz Cloud 免费套餐
- 生产环境：根据需求选择自建或云服务

### Q2: 三元组如何提取？

**方法**：
```python
# 方法 1：使用 LLM 提取
def extract_triplets_with_llm(text, llm):
    prompt = "从以下文本中提取所有三元组关系...
    return llm.invoke(prompt).content

# 方法 2：使用 NER + 关系抽取模型
from transformers import pipeline
ner = pipeline("ner")
# ...
```

### Q3: 多跳查询为什么慢？

**原因**：
- 每次扩展都需要查询数据库
- LLM 重排序需要推理时间

**优化**：
- 缓存常用路径
- 限制最大跳数（建议 2-3 跳）
- 使用异步查询

---

## 📚 总结

### 核心要点回顾

1. **Graph RAG 的价值**：
   - 回答多跳查询（"A 的老师的儿子"）
   - 发现隐含关系
   - 系统性推理

2. **关键技术**：
   - 三元组表示：[主语，谓语，宾语]
   - 子图扩展：BFS/DFS 遍历
   - LLM 重排序：智能筛选

3. **Milvus 的作用**：
   - 存储实体、关系、段落
   - 高效向量相似度搜索
   - 可扩展的分布式架构

### 进阶方向

1. **自动三元组提取**：用 LLM 自动从文本抽取关系
2. **动态图更新**：增量更新知识图谱
3. **混合检索**：结合向量检索和图遍历

---

## 🔗 相关资源

- [Milvus 官方文档](https://milvus.io/docs)
- [Zilliz Cloud 免费试用](https://cloud.zilliz.com/signup)
- [知识图谱入门指南](https://www.w3.org/standards/semanticweb/)
