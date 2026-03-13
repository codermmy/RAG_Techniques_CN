# 🔬 RAG 技术深度研究报告

> 从原理到实践：全面解析检索增强生成系统的核心技术、实现细节与优化策略

**版本**：Deep Research Edition
**最后更新**：2026 年 3 月
**基于项目**：[RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques)

---

## 📋 目录

1. [引言：RAG 技术演进历程](#引言-rag 技术演进历程)
2. [基础架构层：深入理解 RAG 核心组件](#基础架构层深入理解-rag 核心组件)
3. [索引优化层：数据表示与组织的艺术](#索引优化层数据表示与组织的艺术)
4. [查询优化层：从用户意图到精确检索](#查询优化层从用户意图到精确检索)
5. [检索增强层：多策略融合与重排序](#检索增强层多策略融合与重排序)
6. [后处理层：上下文精炼与生成优化](#后处理层上下文精炼与生成优化)
7. [高级架构层：自反思与代理式系统](#高级架构层自反思与代理式系统)
8. [评估与监控：构建可靠 RAG 系统](#评估与监控构建可靠-rag 系统)
9. [实战调优指南](#实战调优指南)
10. [前沿趋势与未来方向](#前沿趋势与未来方向)

---

# 引言：RAG 技术演进历程

## RAG 发展的三个阶段

### 第一阶段：基础 RAG（2020-2022）

2020 年，Facebook AI Research 发表了开创性论文《Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks》，正式提出了 RAG 范式。这一阶段的核心特征是：

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  文档分块   │ →  │  向量检索   │ →  │  LLM 生成   │
│  Chunking   │    │  Retrieval  │    │  Generation │
└─────────────┘    └─────────────┘    └─────────────┘
```

**技术特点：**
- 简单的固定长度分块
- 单一向量相似度检索
- 直接使用检索结果进行生成

**局限性：**
- 检索精度受限于嵌入模型质量
- 分块策略粗糙，容易切断语义
- 无法处理复杂查询和多跳推理

### 第二阶段：增强 RAG（2022-2024）

随着嵌入模型和 LLM 能力的提升，研究者开始在各个环节进行优化：

**索引层增强：**
- 语义分块（Semantic Chunking）
- 父子文档结构（Parent-Child）
- 假设性问题索引（Hypothetical Questions）

**查询层增强：**
- 查询重写（Query Rewriting）
- HyDE 假设文档嵌入
- 多路检索融合（Fusion Retrieval）

**检索层增强：**
- 混合检索（Hybrid Search）
- 重排序（Reranking）
- 上下文压缩

### 第三阶段：智能 RAG（2024-至今）

引入反思、纠错和代理决策能力：

- **Self-RAG**：自我评估检索必要性和答案质量
- **CRAG**：动态纠正检索策略
- **Agentic RAG**：自主规划多步检索任务
- **Graph RAG**：融合知识图谱的结构化推理

---

# 基础架构层：深入理解 RAG 核心组件

## 1.1 基础 RAG 架构深度解析

### 完整数据流

```
原始文档
   ↓
┌──────────────────────────────────────────────┐
│              索引管道 (Indexing Pipeline)     │
├──────────────────────────────────────────────┤
│  文档加载 → 文本清洗 → 分块 → 向量化 → 存储  │
└──────────────────────────────────────────────┘
   ↓
向量数据库 (FAISS/Milvus/Weaviate/Pinecone)
   ↓
┌──────────────────────────────────────────────┐
│              查询管道 (Query Pipeline)        │
├──────────────────────────────────────────────┤
│  查询处理 → 向量化 → 检索 → 重排序 → 生成    │
└──────────────────────────────────────────────┘
   ↓
最终答案
```

### 核心组件技术细节

#### 1. 文档加载器（Document Loaders）

```python
from langchain.document_loaders import (
    PyPDFLoader,        # PDF 文档
    TextLoader,        # 纯文本
    UnstructuredWordDocumentLoader,  # Word 文档
    UnstructuredHTMLLoader,          # HTML 网页
    CSVLoader,                       # CSV 文件
    JSONLoader,                      # JSON 数据
    DirectoryLoader    # 批量加载目录
)

# 实际使用示例
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# 每个 Document 对象包含
# - page_content: 文本内容
# - metadata: 元数据（页码、来源等）
```

#### 2. 文本分块器（Text Splitters）

**RecursiveCharacterTextSplitter 工作原理：**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # 每块最大字符数
    chunk_overlap=200,      # 块间重叠字符数
    length_function=len,    # 长度计算函数
    separators=["\n\n", "\n", "。", ".", "！", "!", "？", "?", " ", ""]  # 分隔符优先级
)

# 分块过程演示
"""
输入：一个长文档

步骤 1: 尝试用 "\n\n" (段落) 分割
       如果分割后每块都 < chunk_size → 完成
       如果有块仍 > chunk_size → 进入下一步

步骤 2: 尝试用 "\n" (换行) 分割大块
       ... 递归继续 ...

步骤 7: 最后用 "" (字符级) 强制分割
       确保每块都满足大小要求
"""
```

**分块大小选择策略：**

| 文档类型 | 推荐 chunk_size | 推荐 overlap | 原因 |
|----------|-----------------|--------------|------|
| 学术论文 | 800-1200 | 150-250 | 段落结构清晰，需要完整论证单元 |
| 法律合同 | 600-1000 | 100-200 | 条款独立性强，需要精确边界 |
| 技术文档 | 1000-1500 | 200-300 | 代码示例可能需要更大上下文 |
| 对话记录 | 500-800 | 100-150 | 对话轮次较短 |
| 新闻文章 | 800-1200 | 150-200 | 单篇文章通常聚焦一个主题 |

#### 3. 嵌入模型（Embedding Models）

**主流嵌入模型对比：**

| 模型 | 维度 | 最大序列长度 | 特点 | 适用场景 |
|------|------|-------------|------|----------|
| OpenAI text-embedding-3-large | 3072 | 8191 | 语义理解强，多语言 | 通用场景 |
| OpenAI text-embedding-3-small | 1536 | 8191 | 性价比高 | 成本敏感场景 |
| BGE-large-zh | 1024 | 512 | 中文优化 | 中文文档 |
| mE5-large | 1024 | 512 | 多语言，开源 | 私有部署 |
| Cohere embed-v3 | 1024 | 2048 | 支持多任务 | 多模态场景 |

**嵌入模型选择决策树：**

```
是否需要中文优化？
├─ 是 → BGE-large-zh / mE5-large-multilingual
└─ 否 →
    ├─ 预算充足 → OpenAI text-embedding-3-large
    ├─ 成本敏感 → OpenAI text-embedding-3-small
    └─ 需要私有部署 → mE5-large / BGE-large
```

#### 4. 向量数据库（Vector Databases）

**FAISS（最常用）：**

```python
from langchain.vectorstores import FAISS

# 创建
vectorstore = FAISS.from_documents(documents, embeddings)

# 保存
vectorstore.save_local("faiss_index")

# 加载
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# 相似度搜索
docs = vectorstore.similarity_search(query, k=4)

# 带分数的相似度搜索
docs_with_scores = vectorstore.similarity_search_with_score(query, k=4)

# 最大边际相关性（MMR，去重）
docs_mmr = vectorstore.max_marginal_relevance_search(query, k=4, fetch_k=20)
```

**向量数据库对比：**

| 数据库 | 类型 | 规模 | 特点 | 适用场景 |
|--------|------|------|------|----------|
| FAISS | 内存库 | 百万级 | 快速，简单 | 原型开发，小规模 |
| Chroma | 嵌入式 | 百万级 | 内置持久化 | 中小规模应用 |
| Milvus | 分布式 | 十亿级 | 企业级功能 | 生产环境大规模 |
| Weaviate | 分布式 | 十亿级 | 内置 GraphQL | 需要复杂查询 |
| Pinecone | 托管服务 | 十亿级 | 免运维 | 快速上线 |

---

## 1.2 RAG 评估体系

### 评估指标详解

#### 1. 忠实度（Faithfulness）

**定义：** 生成的答案是否完全基于检索到的上下文，有无幻觉。

**计算方法（Ragas）：**
```python
from ragas.metrics import faithfulness
from ragas import evaluate

# faithfulness 计算逻辑
def calculate_faithfulness(answer, contexts):
    """
    1. 从 answer 中提取所有陈述句
    2. 对每个陈述，判断是否能从 contexts 中推导出来
    3. faithfulness = 可推导的陈述数 / 总陈述数
    """
    pass

# 示例
answer = "苹果公司成立于 1976 年，创始人是乔布斯。"
contexts = ["苹果公司由史蒂夫·乔布斯和史蒂夫·沃兹尼亚克于 1976 年创立。"]
# faithfulness = 1.0（两个陈述都可从上下文推导）
```

#### 2. 答案相关性（Answer Relevance）

**定义：** 答案是否直接回应了用户问题。

```python
from ragas.metrics import answer_relevancy

# 计算逻辑
def calculate_answer_relevancy(question, answer):
    """
    1. 从 answer 反向生成可能的問題
    2. 计算生成的问题与原始问题的相似度
    3. 相似度越高，答案越相关
    """
    pass
```

#### 3. 上下文召回率（Context Recall）

**定义：** 检索的上下文是否包含回答问题所需的关键信息。

```python
from ragas.metrics import context_recall

# 计算逻辑
def calculate_context_recall(question, answer, contexts):
    """
    1. 识别回答正确答案所需的关键信息
    2. 检查这些关键信息是否出现在 contexts 中
    3. context_recall = 找到的关键信息数 / 总关键信息数
    """
    pass
```

### 完整评估流程

```python
from ragas import evaluate
from datasets import Dataset
import pandas as pd

# 准备评估数据
data = {
    "question": ["问题 1", "问题 2", "问题 3"],
    "answer": ["答案 1", "答案 2", "答案 3"],
    "contexts": [
        ["上下文 1-1", "上下文 1-2"],
        ["上下文 2-1", "上下文 2-2"],
        ["上下文 3-1", "上下文 3-2"]
    ],
    "ground_truth": ["标准答案 1", "标准答案 2", "标准答案 3"]  # 可选
}

# 创建 Dataset
dataset = Dataset.from_dict(data)

# 执行评估
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

results = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
)

# 查看结果
df = results.to_pandas()
print(df.describe())
```

---

# 索引优化层：数据表示与组织的艺术

## 2.1 语义分块（Semantic Chunking）深度解析

### 原理详解

语义分块的核心思想是**在语义边界处自然分割**，而不是按照固定字符数机械切分。

**算法流程：**

```
原始文档
   ↓
按句子拆分 → [S1, S2, S3, ..., Sn]
   ↓
计算相邻句子嵌入相似度
   ↓
sim(S1,S2), sim(S2,S3), sim(S3,S4), ..., sim(Sn-1,Sn)
   ↓
计算分割阈值
   - 百分位法：取第 X 百分位
   - 标准差法：mean + X * std
   - 四分位法：Q3 + 1.5 * IQR
   ↓
在相似度低于阈值处切分
   ↓
[句子 1-3] ||| [句子 4-7] ||| [句子 8-12] ...
```

### 完整实现代码

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

# 初始化
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# 创建语义分块器
semantic_splitter = SemanticChunker(
    embedding_model,
    breakpoint_threshold_type='percentile',  # 分割阈值类型
    breakpoint_threshold_amount=90,          # 第 90 百分位
    number_of_chunks=None,                   # 不限制分块数量
    min_chunk_size=500,                      # 最小分块大小
    sentence_split_regex=r'(?<=[.?!])\s+'    # 句子分割正则
)

# 执行分块
with open("document.txt", "r", encoding="utf-8") as f:
    content = f.read()

documents = semantic_splitter.create_documents([content])

print(f"原始字符数：{len(content)}")
print(f"分块数量：{len(documents)}")
print(f"平均每块字符数：{len(content) / len(documents):.0f}")
```

### 阈值类型对比

| 阈值类型 | 参数含义 | 优点 | 缺点 | 适用场景 |
|----------|---------|------|------|----------|
| `percentile` | 百分位 (0-100) | 直观，可预测分块数 | 对不同文档泛化性差 | 文档风格一致 |
| `standard_deviation` | 标准差倍数 | 自适应文档特性 | 参数调优复杂 | 文档变化大 |
| `interquartile` | 四分位距 | 对异常值鲁棒 | 计算复杂 | 长尾分布文档 |

### 实战建议

```python
# 针对不同文档类型的推荐配置

# 1. 技术文档（结构清晰，术语多）
semantic_splitter_tech = SemanticChunker(
    embedding_model,
    breakpoint_threshold_type='percentile',
    breakpoint_threshold_amount=85,  # 较低阈值，更多分块
    min_chunk_size=300
)

# 2. 文学作品（语义连贯，叙述流畅）
semantic_splitter_lit = SemanticChunker(
    embedding_model,
    breakpoint_threshold_type='percentile',
    breakpoint_threshold_amount=95,  # 较高阈值，更少分块
    min_chunk_size=600
)

# 3. 法律文档（精确，条款独立）
semantic_splitter_legal = SemanticChunker(
    embedding_model,
    breakpoint_threshold_type='standard_deviation',
    breakpoint_threshold_amount=2.0,  # 2 倍标准差
    min_chunk_size=400
)
```

---

## 2.2 命题分块（Proposition Chunking）深度解析

### 什么是命题？

**命题**是一个能够独立表达完整事实的最小语义单元。例如：

```
原文："苹果公司于 1976 年由史蒂夫·乔布斯和史蒂夫·沃兹尼亚克在加利福尼亚州创立，
      总部位于库比蒂诺，现任 CEO 是蒂姆·库克。"

提取的命题：
- 命题 1：苹果公司的成立时间是 1976 年
- 命题 2：苹果公司的创始人包括史蒂夫·乔布斯
- 命题 3：苹果公司的创始人包括史蒂夫·沃兹尼亚克
- 命题 4：苹果公司的创立地点是加利福尼亚州
- 命题 5：苹果公司总部位于库比蒂诺
- 命题 6：苹果公司的现任 CEO 是蒂姆·库克
```

### 完整实现流程

```python
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

# 1. 定义命题的数据结构
class Proposition(BaseModel):
    text: str = Field(description="命题的文本表述")
    confidence: float = Field(description="命题的可信度，0-1 之间")
    entity_mentions: List[str] = Field(description="命题中提到的实体")

class PropositionsOutput(BaseModel):
    propositions: List[Proposition]

# 2. 创建提取提示
proposition_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个信息提取专家。你的任务是从给定的文本中提取出所有独立的'命题'。

命题的定义：
- 能够独立表达一个完整事实
- 不包含代词，所有实体都要明确指代
- 简洁、准确、无歧义

请按 JSON 格式输出，包含命题文本、可信度和提到的实体。"""),
    ("human", "文本：{text}")
])

# 3. 创建 LLM 链
llm = ChatOpenAI(model="gpt-4o", temperature=0)
proposition_chain = proposition_prompt | llm.with_structured_output(PropositionsOutput)

# 4. 执行提取
def extract_propositions(text: str) -> List[Proposition]:
    result = proposition_chain.invoke({"text": text})
    return result.propositions

# 5. 质量评分（可选）
class PropositionQuality(BaseModel):
    clarity: float  # 清晰度
    completeness: float  # 完整性
    independence: float  # 独立性
    overall_score: float  # 综合评分

quality_prompt = ChatPromptTemplate.from_messages([
    ("system", "评估以下命题的质量，各项评分 0-1 分。"),
    ("human", "命题：{proposition}")
])
quality_chain = quality_prompt | llm.with_structured_output(PropositionQuality)

def score_proposition(proposition: str) -> PropositionQuality:
    return quality_chain.invoke({"proposition": proposition})
```

### 命题分块 vs 传统分块

| 维度 | 传统分块 | 命题分块 |
|------|----------|----------|
| 语义完整性 | 可能断裂 | 完整独立 |
| 检索精度 | 中等 | 高 |
| 事实核查 | 困难 | 容易 |
| 计算成本 | 低 | 高（需要 LLM） |
| 适用场景 | 通用 | 知识密集型、事实验证 |

### 应用场景

1. **知识图谱构建**：命题天然适合转换为 RDF 三元组
2. **事实验证系统**：每个命题可独立验证
3. **法律合规审查**：精确追踪每个事实陈述
4. **医疗问答**：确保医疗建议的准确性

---

## 2.3 RAPTOR：递归摘要树检索

### 架构深度解析

RAPTOR（Recursive Abstractive Processing and Thematic Organization for Retrieval）的核心是构建一个**多层次的摘要树**。

```
Level 3 (根节点)
└── [全球气候变化的综合评估报告摘要]
    ├── Level 2
    │   ├── [气候变化原因摘要]
    │   │   ├── Level 1
    │   │   │   ├── [温室气体排放摘要]
    │   │   │   │   └── Level 0 (叶子节点)
    │   │   │   │       ├── [化石燃料燃烧释放 CO2...]
    │   │   │   │       ├── [工业生产过程中的排放...]
    │   │   │   │       └── [...]
    │   │   │   ├── [森林砍伐影响摘要]
    │   │   │   │   └── Level 0
    │   │   │   │       ├── [亚马逊雨林面积减少...]
    │   │   │   │       └── [...]
    │   │   │   └── [...]
    │   │   └── [自然因素摘要]
    │   └── [气候变化影响摘要]
    └── [应对策略摘要]
```

### 完整实现代码

```python
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.mixture import GaussianMixture
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.docstore.document import Document

# ========== 1. 树构建模块 ==========

def build_raptor_tree(
    texts: List[str],
    max_levels: int = 3,
    embeddings: OpenAIEmbeddings = None,
    llm: ChatOpenAI = None
) -> Dict[int, pd.DataFrame]:
    """
    构建 RAPTOR 树结构

    Args:
        texts: 原始文本列表
        max_levels: 最大层数
        embeddings: 嵌入模型
        llm: 语言模型

    Returns:
        每层的 DataFrame 字典
    """
    results = {}
    current_texts = texts
    current_metadata = [{"level": 0, "origin": "original", "parent_id": None} for _ in texts]

    for level in range(1, max_levels + 1):
        print(f"正在处理第 {level} 层...")

        # 1. 嵌入当前层文本
        embedding_matrix = np.array(embeddings.embed_documents(current_texts))

        # 2. 聚类
        n_clusters = min(10, len(current_texts) // 2)
        if n_clusters < 2:
            break

        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        cluster_labels = gmm.fit_predict(embedding_matrix)

        # 3. 存储当前层
        df = pd.DataFrame({
            'text': current_texts,
            'embedding': list(embedding_matrix),
            'cluster': cluster_labels,
            'metadata': current_metadata
        })
        results[level - 1] = df

        # 4. 生成下一层（摘要层）
        summaries = []
        new_metadata = []

        for cluster_id in df['cluster'].unique():
            cluster_docs = df[df['cluster'] == cluster_id]
            cluster_texts = cluster_docs['text'].tolist()
            cluster_meta = cluster_docs['metadata'].tolist()

            # 生成摘要
            summary = summarize_texts(cluster_texts, llm)
            summaries.append(summary)

            # 记录父子关系
            new_metadata.append({
                "level": level,
                "origin": f"summary_cluster_{cluster_id}_level_{level-1}",
                "child_ids": [m.get('id') for m in cluster_meta],
                "id": f"summary_{level}_{cluster_id}"
            })

        current_texts = summaries
        current_metadata = new_metadata

        # 如果只剩一个摘要，停止
        if len(current_texts) <= 1:
            results[level] = pd.DataFrame({
                'text': current_texts,
                'embedding': list(embeddings.embed_documents(current_texts)),
                'cluster': [0],
                'metadata': current_metadata
            })
            print(f"在第 {level} 层收敛到单个摘要")
            break

    return results


def summarize_texts(texts: List[str], llm: ChatOpenAI) -> str:
    """生成文本摘要"""
    prompt = ChatPromptTemplate.from_template(
        "请简洁地总结以下文本，保留关键信息：\n\n{text}"
    )
    chain = prompt | llm
    return chain.invoke({"text": texts}).content


# ========== 2. 检索模块 ==========

def build_vectorstore(tree_results: Dict[int, pd.DataFrame], embeddings) -> FAISS:
    """从树的所有层构建向量库"""
    all_texts = []
    all_embeddings = []
    all_metadatas = []

    for level, df in tree_results.items():
        all_texts.extend([str(t) for t in df['text'].tolist()])
        all_embeddings.extend([e.tolist() if isinstance(e, np.ndarray) else e
                               for e in df['embedding'].tolist()])
        all_metadatas.extend(df['metadata'].tolist())

    documents = [
        Document(page_content=str(text), metadata=meta)
        for text, meta in zip(all_texts, all_metadatas)
    ]

    return FAISS.from_documents(documents, embeddings)


def hierarchical_retrieval(
    query: str,
    vectorstore: FAISS,
    embeddings: OpenAIEmbeddings,
    k: int = 3
) -> List[Document]:
    """
    层级检索：从顶层开始，逐步向下
    """
    query_embedding = embeddings.embed_query(query)

    def retrieve_level(level: int, parent_ids: List[str] = None) -> List[Document]:
        # 构建过滤条件
        def filter_func(meta):
            level_match = meta.get('level') == level
            if parent_ids:
                return level_match and meta.get('id') in parent_ids
            return level_match

        docs = vectorstore.similarity_search_by_vector_with_relevance_scores(
            query_embedding, k=k, filter=filter_func
        )

        if not docs or level == 0:
            return [doc for doc, _ in docs]

        # 获取子节点 ID
        child_ids = []
        for doc, _ in docs:
            child_ids.extend(doc.metadata.get('child_ids', []))

        # 递归检索子节点
        child_docs = retrieve_level(level - 1, child_ids)
        return [doc for doc, _ in docs] + child_docs

    max_level = max(doc.metadata.get('level', 0)
                    for doc in vectorstore.docstore.values())
    return retrieve_level(max_level)


# ========== 3. 使用示例 ==========

if __name__ == "__main__":
    # 初始化
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # 准备文档
    texts = ["文档 1 内容...", "文档 2 内容...", ...]

    # 构建树
    tree_results = build_raptor_tree(
        texts=texts,
        max_levels=3,
        embeddings=embeddings,
        llm=llm
    )

    # 构建向量库
    vectorstore = build_vectorstore(tree_results, embeddings)

    # 执行查询
    query = "气候变化的主要原因是什么？"
    results = hierarchical_retrieval(query, vectorstore, embeddings, k=5)

    # 生成答案
    context = "\n\n".join([doc.page_content for doc in results])
    answer_prompt = ChatPromptTemplate.from_template(
        "基于以下上下文回答问题：\n\n上下文：{context}\n\n问题：{question}\n\n答案："
    )
    answer = answer_prompt | llm
    print(answer.invoke({"context": context, "question": query}))
```

### 参数调优指南

| 参数 | 含义 | 推荐值 | 影响 |
|------|------|--------|------|
| `max_levels` | 树的最大层数 | 3-5 | 层数越多，摘要越抽象，但可能丢失细节 |
| `n_clusters` | 每层聚类数 | 5-15 | 聚类数影响摘要的粒度 |
| `summary_prompt` | 摘要提示 | 根据领域定制 | 影响摘要质量和焦点 |

### 适用场景分析

| 场景 | 适合度 | 原因 |
|------|--------|------|
| 长文档综合查询 | ⭐⭐⭐⭐⭐ | 支持宏观到微观的多层次检索 |
| 跨文档模式发现 | ⭐⭐⭐⭐⭐ | 摘要层可揭示隐藏关联 |
| 精确事实验证 | ⭐⭐⭐ | 摘要可能丢失细节，建议用叶子节点 |
| 实时问答 | ⭐⭐ | 树构建耗时，不适合频繁更新 |

---

# 查询优化层：从用户意图到精确检索

## 3.1 查询重写（Query Rewriting）完整指南

### 为什么需要查询重写？

用户查询通常存在以下问题：
1. **指代不明**：「它的创始人是谁？」（缺少上下文）
2. **表述模糊**：「那个怎么弄？」
3. **缺少关键信息**：「Python 教程」（需要什么级别的？）
4. **语法不规范**：口语化、错别字等

### 查询重写策略矩阵

```
┌─────────────────────────────────────────────────────────────┐
│                    查询重写策略矩阵                          │
├──────────────┬──────────────────────────────────────────────┤
│   策略类型   │              具体方法                        │
├──────────────┼──────────────────────────────────────────────┤
│ 指代消解     │ 用对话历史中的实体替换代词                    │
│ 查询扩展     │ 添加同义词、相关术语                          │
│ 查询简化     │ 移除冗余词汇，保留核心意图                    │
│ 结构化改写   │ 转换为更适合检索的格式                        │
│ 多语言转换   │ 将查询转换为目标文档的主要语言                │
└──────────────┴──────────────────────────────────────────────┘
```

### 完整实现代码

```python
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ========== 策略 1: 指代消解 ==========

class ResolvedQuery(BaseModel):
    original_query: str = Field(description="原始查询")
    resolved_query: str = Field(description="消除指代后的查询")
    resolved_entities: List[str] = Field(description="被解析的实体列表")

resolution_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个查询理解专家。请识别并解析查询中的指代词（如'它'、'这个'、'那个'等）。
结合对话历史，将指代词替换为具体的实体名称。

如果查询中没有指代词，直接返回原查询。"""),
    ("human", """对话历史：
{conversation_history}

当前查询：{query}

请输出解析后的查询。""")
])

resolution_chain = resolution_prompt | llm.with_structured_output(ResolvedQuery)


# ========== 策略 2: 查询扩展 ==========

class ExpandedQuery(BaseModel):
    original_query: str
    expanded_query: str
    added_terms: List[str] = Field(description="添加的相关术语")

expansion_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个信息检索专家。请扩展用户查询，添加相关的同义词、上下位词和相关概念。

目标：提高检索召回率，同时保持查询的核心意图。

要求：
- 添加 2-5 个相关术语
- 使用 OR 连接同义词
- 保持查询简洁"""),
    ("human", "查询：{query}\n领域：{domain}")
])

expansion_chain = expansion_prompt | llm.with_structured_output(ExpandedQuery)


# ========== 策略 3: 查询简化 ==========

class SimplifiedQuery(BaseModel):
    original_query: str
    simplified_query: str
    removed_words: List[str] = Field(description="移除的冗余词汇")

simplification_prompt = ChatPromptTemplate.from_messages([
    ("system", """简化以下查询，移除冗余词汇，保留核心意图。

要求：
- 移除口语化表达
- 删除无意义的修饰词
- 保留关键实体和意图"""),
    ("human", "查询：{query}")
]))

simplification_chain = simplification_prompt | llm.with_structured_output(SimplifiedQuery)


# ========== 策略 4: 多策略组合 ==========

class RewrittenQuery(BaseModel):
    original_query: str
    rewritten_query: str
    strategies_applied: List[str]
    confidence: float

combined_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个查询重写专家。请综合运用以下策略重写查询：

1. 指代消解：将代词替换为具体实体
2. 查询扩展：添加同义词和相关术语
3. 查询简化：移除冗余表达
4. 结构化：使查询更适合检索

输入：
- 当前查询
- 对话历史（可选）
- 领域信息（可选）

输出重写后的查询和应用的策略。"""),
    ("human", """查询：{query}
对话历史：{history}
领域：{domain}""")
]))

combined_chain = combined_prompt | llm.with_structured_output(RewrittenQuery)


# ========== 使用示例 ==========

def rewrite_query(
    query: str,
    conversation_history: str = "",
    domain: str = "通用",
    strategy: str = "combined"
) -> str:
    """
    重写查询以提升检索效果

    Args:
        query: 原始查询
        conversation_history: 对话历史
        domain: 领域信息
        strategy: 使用的策略 (resolution/expansion/simplification/combined)

    Returns:
        重写后的查询
    """
    if strategy == "resolution":
        result = resolution_chain.invoke({
            "query": query,
            "conversation_history": conversation_history
        })
        return result.resolved_query

    elif strategy == "expansion":
        result = expansion_chain.invoke({
            "query": query,
            "domain": domain
        })
        return result.expanded_query

    elif strategy == "simplification":
        result = simplification_chain.invoke({"query": query})
        return result.simplified_query

    else:  # combined
        result = combined_chain.invoke({
            "query": query,
            "history": conversation_history,
            "domain": domain
        })
        return result.rewritten_query


# 测试
if __name__ == "__main__":
    # 测试指代消解
    query1 = "它的创始人是谁？"
    history1 = """用户：苹果公司是什么时候成立的？
助手：苹果公司于 1976 年成立。"""
    print(f"指代消解：{rewrite_query(query1, history1, strategy='resolution')}")
    # 输出："苹果公司的创始人是谁？"

    # 测试查询扩展
    query2 = "机器学习"
    print(f"查询扩展：{rewrite_query(query2, domain='人工智能', strategy='expansion')}")
    # 输出："机器学习 OR 深度学习 OR 神经网络 OR 监督学习 OR 无监督学习"

    # 测试查询简化
    query3 = "我想知道一下那个 Python 的东西就是怎么安装 pip 库"
    print(f"查询简化：{rewrite_query(query3, strategy='simplification')}")
    # 输出："如何安装 Python pip 库"
```

---

## 3.2 HyDE（假设文档嵌入）深度解析

### 核心思想

HyDE（Hypothetical Document Embedding）的核心洞察是：**查询的嵌入空间与文档的嵌入空间存在鸿沟**。

```
传统方法的问题:

查询空间                          文档空间
"气候变化的原因"    ←────鸿沟──→   "温室气体排放导致全球变暖..."
(简短、问题式)                    (冗长、陈述式)


HyDE 的解决方案:

查询 → LLM 生成假设答案 → "气候变化的主要原因是..."    ≈    "温室气体排放导致全球变暖..."
                                ↓                            ↓
                         相同的嵌入空间                    相同的嵌入空间
                                ↓                            ↓
                         向量相似度高 → 检索准确
```

### 为什么 HyDE 有效？

1. **语义空间对齐**：假设答案与真实文档在语义空间上更接近
2. **查询意图显式化**：LLM 生成的假设答案隐含了查询的意图
3. **零样本适应**：无需训练，直接使用

### 完整实现代码

```python
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS

# ========== 1. HyDE 核心组件 ==========

class HyDERetriever:
    """HyDE 检索器"""

    def __init__(
        self,
        vectorstore: FAISS,
        llm: ChatOpenAI,
        embeddings: OpenAIEmbeddings,
        k: int = 5
    ):
        self.vectorstore = vectorstore
        self.llm = llm
        self.embeddings = embeddings
        self.k = k

        # HyDE 提示模板
        self.hyde_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个信息助手。请根据问题生成一个详细的答案段落。

要求：
- 答案应该详细、具体，像一个真实的文档片段
- 即使你不确定答案，也要生成一个合理的假设
- 使用陈述句，不要使用疑问句"""),
            ("human", "问题：{question}")
        ])
        self.hyde_chain = self.hyde_prompt | llm

    def generate_hypothetical_document(self, query: str) -> str:
        """生成假设文档"""
        result = self.hyde_chain.invoke({"question": query})
        return result.content

    def retrieve(self, query: str) -> List[Document]:
        """
        使用 HyDE 进行检索

        流程:
        1. 生成假设文档
        2. 将假设文档嵌入为向量
        3. 用假设文档向量检索真实文档
        """
        # 生成假设文档
        hypothetical_doc = self.generate_hypothetical_document(query)
        print(f"假设文档：{hypothetical_doc[:200]}...")

        # 用假设文档检索（注意：不是用原始查询！）
        docs = self.vectorstore.similarity_search(hypothetical_doc, k=self.k)

        return docs


# ========== 2. 进阶：多假设 HyDE ==========

class MultiHyDERetriever(HyDERetriever):
    """多假设 HyDE 检索器"""

    def __init__(self, *args, num_hypotheses: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_hypotheses = num_hypotheses

        # 多样化假设提示
        self.diverse_hyde_prompt = ChatPromptTemplate.from_messages([
            ("system", """请为以下问题生成 {num} 个不同角度的假设答案。

每个答案应该：
- 从不同角度切入（如原因、影响、解决方案等）
- 详细具体，像真实文档
- 彼此之间有差异性"""),
            ("human", "问题：{question}")
        ])
        self.diverse_hyde_chain = self.diverse_hyde_prompt | llm

    def retrieve(self, query: str) -> List[Document]:
        # 生成多个假设文档
        result = self.diverse_hyde_chain.invoke({
            "question": query,
            "num": self.num_hypotheses
        })
        hypotheses = result.content.split('\n\n')

        # 从每个假设检索
        all_docs = []
        for hypothesis in hypotheses[:self.num_hypotheses]:
            docs = self.vectorstore.similarity_search(hypothesis.strip(), k=self.k // self.num_hypotheses)
            all_docs.extend(docs)

        # 去重
        unique_docs = []
        seen_contents = set()
        for doc in all_docs:
            if doc.page_content not in seen_contents:
                unique_docs.append(doc)
                seen_contents.add(doc.page_content)

        return unique_docs[:self.k]


# ========== 3. 使用示例 ==========

if __name__ == "__main__":
    # 初始化
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)  # 较高温度以增加多样性
    vectorstore = FAISS.from_texts(
        ["文档 1...", "文档 2...", ...],  # 实际文档
        embeddings
    )

    # 基础 HyDE
    hyde_retriever = HyDERetriever(vectorstore, llm, embeddings, k=5)
    query = "气候变化的主要原因是什么？"
    docs = hyde_retriever.retrieve(query)

    # 多假设 HyDE
    multi_hyde_retriever = MultiHyDERetriever(
        vectorstore, llm, embeddings, k=5, num_hypotheses=3
    )
    docs_multi = multi_hyde_retriever.retrieve(query)

    print(f"基础 HyDE 检索到 {len(docs)} 个文档")
    print(f"多假设 HyDE 检索到 {len(docs_multi)} 个文档")
```

### HyDE 变体对比

| 变体 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| 基础 HyDE | 生成单个假设文档 | 简单快速 | 可能遗漏其他角度 |
| 多假设 HyDE | 生成多个不同角度的假设 | 覆盖更广 | 需要更多 LLM 调用 |
| 结构化 HyDE | 按特定格式生成假设（如列表、表格） | 适合结构化文档 | 需要定制提示 |
| 迭代 HyDE | 基于初步检索结果再生成假设 | 逐步精确 | 延迟较高 |

### 何时使用 HyDE？

| 场景 | 推荐度 | 原因 |
|------|--------|------|
| 开放性问题 | ⭐⭐⭐⭐⭐ | 假设答案能有效桥接语义鸿沟 |
| 事实性查询 | ⭐⭐⭐ | 直接检索可能更准确 |
| 复杂推理问题 | ⭐⭐⭐⭐ | 假设答案可展示推理链 |
| 专业领域问题 | ⭐⭐⭐ | 需要领域特定的假设生成提示 |

---

## 3.3 语义路由（Semantic Router）

### 什么是语义路由？

语义路由是一种**轻量级的查询分类机制**，用于将查询导向最合适的处理管道。

```
                    用户查询
                       ↓
              ┌────────────────┐
              │   语义路由器   │
              │  (轻量分类器)  │
              └────────────────┘
                       ↓
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │ 闲聊管道 │   │ 检索管道 │   │ 工具管道 │
   │ (跳过 RAG)│   │ (标准 RAG)│   │ (Agent) │
   └─────────┘   └─────────┘   └─────────┘
```

### 为什么需要语义路由？

1. **降低成本**：不是所有查询都需要检索
2. **提升效果**：不同问题类型需要不同处理策略
3. **提高效率**：避免不必要的计算

### 完整实现代码

```python
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from enum import Enum
from typing import List, Optional
import numpy as np

# ========== 1. 定义路由类型 ==========

class RouteType(str, Enum):
    CHITCHAT = "chitchat"        # 闲聊，不需要检索
    FACTUAL = "factual"          # 事实查询，需要检索
    REASONING = "reasoning"      # 推理问题，需要多步检索
    TECHNICAL = "technical"      # 技术问题，需要专业检索
    ACTION = "action"            # 需要执行操作（如调用 API）
    AMBIGUOUS = "ambiguous"      # 模糊查询，需要澄清


class RouterOutput(BaseModel):
    route_type: RouteType
    confidence: float = Field(description="置信度，0-1")
    reasoning: str = Field(description="路由决策理由")
    suggested_pipeline: str = Field(description="建议使用的处理管道")


# ========== 2. 实现语义路由器 ==========

class SemanticRouter:
    """语义路由器"""

    def __init__(self, llm: ChatOpenAI = None):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)

        self.router_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个语义路由器，负责将用户查询分类到最合适的处理管道。

可用的路由类型：
- chitchat: 闲聊、问候、简单对话（如"你好"、"谢谢"）
- factual: 事实性查询，需要外部知识（如"法国的首都是哪里？"）
- reasoning: 需要逻辑推理或多步思考（如"比较 A 和 B 的优缺点"）
- technical: 专业技术问题（如代码、配置、故障排除）
- action: 需要执行操作（如"帮我搜索"、"发送邮件"）
- ambiguous: 模糊不清、需要澄清的查询

请分析查询并输出：
1. 路由类型
2. 置信度
3. 决策理由
4. 建议的处理管道"""),
            ("human", "查询：{query}")
        ])

        self.router_chain = self.router_prompt | llm.with_structured_output(RouterOutput)

    def route(self, query: str) -> RouterOutput:
        """对查询进行路由"""
        return self.router_chain.invoke({"query": query})


# ========== 3. 基于嵌入的快速路由（无需 LLM）==========

class EmbeddingRouter:
    """基于嵌入的快速路由器"""

    def __init__(self, embeddings, route_examples: dict):
        """
        Args:
            embeddings: 嵌入模型
            route_examples: 每类路由的示例查询
                {
                    "chitchat": ["你好", "再见", "谢谢"],
                    "factual": ["什么是...", "为什么...", ...],
                    ...
                }
        """
        self.embeddings = embeddings
        self.route_examples = route_examples

        # 预计算每类路由的"原型"嵌入（示例的平均向量）
        self.route_prototypes = {}
        for route_type, examples in route_examples.items():
            example_embeddings = embeddings.embed_documents(examples)
            self.route_prototypes[route_type] = np.mean(example_embeddings, axis=0)

    def route(self, query: str, top_k: int = 1) -> List[tuple]:
        """
        将查询路由到最匹配的 k 个路由类型

        Returns:
            [(route_type, similarity_score), ...]
        """
        query_embedding = self.embeddings.embed_query(query)

        # 计算与每个原型的相似度
        similarities = {}
        for route_type, prototype in self.route_prototypes.items():
            sim = np.dot(query_embedding, prototype) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(prototype)
            )
            similarities[route_type] = sim

        # 返回最匹配的 k 个
        sorted_routes = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_routes[:top_k]


# ========== 4. 路由处理器 ==========

class RoutedQueryProcessor:
    """根据路由处理查询"""

    def __init__(self, router, retrieval_pipeline, chitchat_pipeline, action_pipeline):
        self.router = router
        self.retrieval_pipeline = retrieval_pipeline
        self.chitchat_pipeline = chitchat_pipeline
        self.action_pipeline = action_pipeline

    def process(self, query: str) -> str:
        """处理查询并返回结果"""
        route_decision = self.router.route(query)

        if route_decision.route_type == RouteType.CHITCHAT:
            return self.chitchat_pipeline.respond(query)

        elif route_decision.route_type in [RouteType.FACTUAL, RouteType.TECHNICAL]:
            return self.retrieval_pipeline.query(query)

        elif route_decision.route_type == RouteType.REASONING:
            return self.retrieval_pipeline.query_with_reasoning(query)

        elif route_decision.route_type == RouteType.ACTION:
            return self.action_pipeline.execute(query)

        else:  # AMBIGUOUS
            return "您的问题不太清楚，能否提供更多细节？"


# ========== 5. 使用示例 ==========

if __name__ == "__main__":
    # LLM 路由器
    llm_router = SemanticRouter()

    queries = [
        "你好啊！",
        "气候变化的主要原因是什么？",
        "帮我比较一下太阳能和风能的优缺点",
        "如何在 Python 中安装 pip？",
        "帮我搜索一下最近的新闻"
    ]

    for query in queries:
        result = llm_router.route(query)
        print(f"\n查询：{query}")
        print(f"路由：{result.route_type} (置信度：{result.confidence:.2f})")
        print(f"理由：{result.reasoning}")

    # 嵌入路由器（更快，但精度略低）
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()

    route_examples = {
        "chitchat": ["你好", "再见", "谢谢", "早上好"],
        "factual": ["什么是", "为什么", "什么时候", "在哪里"],
        "technical": ["如何安装", "代码怎么写", "报错怎么办"],
        "action": ["帮我搜索", "发送邮件", "设置提醒"]
    }

    emb_router = EmbeddingRouter(embeddings, route_examples)

    for query in queries:
        result = emb_router.route(query)
        print(f"\n查询：{query}")
        print(f"最匹配路由：{result[0][0]} (相似度：{result[0][1]:.3f})")
```

### 路由器选择指南

| 路由器类型 | 延迟 | 精度 | 成本 | 适用场景 |
|------------|------|------|------|----------|
| LLM 路由器 | 高（~500ms） | 高 | 高 | 复杂查询、多类别 |
| 嵌入路由器 | 低（~50ms） | 中 | 低 | 简单分类、实时场景 |
| 规则路由器 | 极低（~1ms） | 低 | 无 | 明确的关键词匹配 |

---

# 检索增强层：多策略融合与重排序

## 4.1 融合检索（Fusion Retrieval）深度解析

### 核心思想

融合检索结合了**向量检索（语义相似度）**和**BM25（关键词匹配）**的优势：

```
┌─────────────────────────────────────────────────────────┐
│                  融合检索架构                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│                    用户查询                              │
│                       ↓                                  │
│          ┌────────────┴────────────┐                    │
│          ↓                         ↓                    │
│   ┌─────────────┐          ┌─────────────┐             │
│   │  向量检索   │          │   BM25 检索  │             │
│   │ (语义相似)  │          │ (关键词匹配) │             │
│   └─────────────┘          └─────────────┘             │
│          ↓                         ↓                    │
│   文档 A: 0.9                 文档 C: 0.85              │
│   文档 B: 0.8                 文档 A: 0.7               │
│   文档 C: 0.7                 文档 B: 0.6               │
│          ↓                         ↓                    │
│          ┌────────────┴────────────┐                    │
│          │    分数归一化 + 融合     │                    │
│          └────────────┬────────────┘                    │
│                       ↓                                  │
│          文档 A: 0.5×0.9 + 0.5×0.7 = 0.8                │
│          文档 B: 0.5×0.8 + 0.5×0.6 = 0.7                │
│          文档 C: 0.5×0.7 + 0.5×0.85 = 0.775             │
│                       ↓                                  │
│              最终排序：A > C > B                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 为什么需要融合检索？

| 检索类型 | 擅长场景 | 不擅长场景 |
|----------|----------|------------|
| 向量检索 | 语义相似、同义词、概念匹配 | 专有名词、精确匹配、新词 |
| BM25 | 术语、人名、代码、新词 | 语义等价、改写、同义词 |

**融合检索 = 两者的优势互补**

### 完整实现代码

```python
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi
from typing import List, Tuple
import numpy as np

class FusionRetriever:
    """融合检索器"""

    def __init__(
        self,
        documents: List[Document],
        embeddings: OpenAIEmbeddings,
        k: int = 5,
        alpha: float = 0.5
    ):
        """
        Args:
            documents: 文档列表
            embeddings: 嵌入模型
            k: 返回的文档数量
            alpha: 向量检索权重 (0-1, 1-alpha 为 BM25 权重)
        """
        self.documents = documents
        self.embeddings = embeddings
        self.k = k
        self.alpha = alpha

        # 1. 构建向量库
        self.vectorstore = FAISS.from_documents(documents, embeddings)

        # 2. 构建 BM25 索引
        self.bm25 = self._create_bm25_index(documents)

    def _create_bm25_index(self, documents: List[Document]) -> BM25Okapi:
        """创建 BM25 索引"""
        # 简单的分词（中文可能需要更复杂的分词器如 jieba）
        tokenized_docs = [doc.page_content.split() for doc in documents]
        return BM25Okapi(tokenized_docs)

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Min-Max 归一化分数到 0-1 范围"""
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score - min_score < 1e-8:
            return np.ones_like(scores)
        return (scores - min_score) / (max_score - min_score)

    def retrieve(self, query: str) -> List[Document]:
        """
        执行融合检索

        步骤：
        1. 获取所有文档
        2. BM25 检索并归一化分数
        3. 向量检索并归一化分数
        4. 加权融合
        5. 排序返回 Top-K
        """
        # 获取所有文档
        all_docs = list(self.documents)
        n_docs = len(all_docs)

        # BM25 检索
        bm25_scores_raw = self.bm25.get_scores(query.split())
        bm25_scores = self._normalize_scores(bm25_scores_raw)

        # 向量检索
        vector_results = self.vectorstore.similarity_search_with_score(
            query, k=n_docs
        )
        vector_scores_raw = np.array([score for _, score in vector_results])
        # 注意：向量距离越小越相似，需要转换为分数
        vector_scores = self._normalize_scores(1 - vector_scores_raw)

        # 融合分数
        combined_scores = (
            self.alpha * vector_scores +
            (1 - self.alpha) * bm25_scores
        )

        # 排序
        sorted_indices = np.argsort(combined_scores)[::-1]

        # 返回 Top-K 文档
        return [all_docs[i] for i in sorted_indices[:self.k]]

    def retrieve_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """返回带分数的文档"""
        all_docs = list(self.documents)
        n_docs = len(all_docs)

        bm25_scores_raw = self.bm25.get_scores(query.split())
        bm25_scores = self._normalize_scores(bm25_scores_raw)

        vector_results = self.vectorstore.similarity_search_with_score(
            query, k=n_docs
        )
        vector_scores_raw = np.array([score for _, score in vector_results])
        vector_scores = self._normalize_scores(1 - vector_scores_raw)

        combined_scores = (
            self.alpha * vector_scores +
            (1 - self.alpha) * bm25_scores
        )

        sorted_indices = np.argsort(combined_scores)[::-1]

        return [
            (all_docs[i], combined_scores[i])
            for i in sorted_indices[:self.k]
        ]


# ========== 进阶：RRF 融合 ==========

class RRFFusionRetriever:
    """使用 RRF（Reciprocal Rank Fusion）的融合检索器"""

    def __init__(
        self,
        documents: List[Document],
        embeddings: OpenAIEmbeddings,
        k: int = 5,
        rrf_k: int = 60
    ):
        """
        Args:
            documents: 文档列表
            embeddings: 嵌入模型
            k: 返回的文档数量
            rrf_k: RRF 参数，控制排名倒数的平滑度
        """
        self.documents = documents
        self.embeddings = embeddings
        self.k = k
        self.rrf_k = rrf_k

        self.vectorstore = FAISS.from_documents(documents, embeddings)
        self.bm25 = self._create_bm25_index(documents)

    def _create_bm25_index(self, documents: List[Document]) -> BM25Okapi:
        tokenized_docs = [doc.page_content.split() for doc in documents]
        return BM25Okapi(tokenized_docs)

    def _rrf_score(self, rank: int) -> float:
        """计算 RRF 分数"""
        return 1 / (self.rrf_k + rank)

    def retrieve(self, query: str) -> List[Document]:
        # 获取 BM25 排名
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_ranks = np.argsort(bm25_scores)[::-1]  # 分数越高排名越前

        # 获取向量检索排名
        n_docs = len(self.documents)
        vector_results = self.vectorstore.similarity_search_with_score(
            query, k=n_docs
        )
        vector_docs = [doc for doc, _ in vector_results]
        vector_rank_map = {id(doc): i for i, doc in enumerate(vector_docs)}

        # 计算 RRF 分数
        rrf_scores = {}
        for i, doc in enumerate(self.documents):
            doc_id = id(doc)
            score = 0

            # BM25 排名贡献
            bm25_rank = np.where(bm25_ranks == i)[0][0]
            score += self._rrf_score(bm25_rank)

            # 向量排名贡献
            if doc_id in vector_rank_map:
                vector_rank = vector_rank_map[doc_id]
                score += self._rrf_score(vector_rank)

            rrf_scores[i] = score

        # 排序
        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        return [self.documents[i] for i in sorted_indices[:self.k]]


# ========== 使用示例 ==========

if __name__ == "__main__":
    from langchain_openai import OpenAIEmbeddings

    # 准备文档
    documents = [
        Document(page_content="机器学习是人工智能的一个分支"),
        Document(page_content="深度学习使用神经网络进行模型训练"),
        Document(page_content="Python 是流行的编程语言"),
        Document(page_content="自然语言处理涉及语言理解和生成"),
        # ... 更多文档
    ]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 加权融合检索
    fusion_retriever = FusionRetriever(
        documents=documents,
        embeddings=embeddings,
        k=3,
        alpha=0.5  # 向量和 BM25 各占 50%
    )

    query = "机器学习和深度学习有什么关系？"
    results = fusion_retriever.retrieve(query)

    print("查询:", query)
    print("\n检索结果:")
    for i, doc in enumerate(results):
        print(f"{i+1}. {doc.page_content[:50]}...")

    # RRF 融合检索
    rrf_retriever = RRFFusionRetriever(
        documents=documents,
        embeddings=embeddings,
        k=3,
        rrf_k=60
    )

    results_rrf = rrf_retriever.retrieve(query)

    print("\nRRF 检索结果:")
    for i, doc in enumerate(results_rrf):
        print(f"{i+1}. {doc.page_content[:50]}...")
```

### 参数调优指南

#### alpha 参数选择

| alpha 值 | 向量权重 | BM25 权重 | 适用场景 |
|----------|----------|-----------|----------|
| 0.3 | 30% | 70% | 术语密集、需要精确匹配（如法律、医学） |
| 0.5 | 50% | 50% | 通用场景，平衡语义和关键词 |
| 0.7 | 70% | 30% | 语义理解更重要（如文学、社科） |

#### RRF k 参数选择

| rrf_k 值 | 特点 | 适用场景 |
|----------|------|----------|
| 30-40 | 排名影响大 | 想要显著区分排名 |
| 60 | 默认值，平衡 | 通用场景 |
| 80-100 | 排名影响小 | 想要更平滑的分数分布 |

---

## 4.2 重排序（Reranking）深度解析

### 重排序的必要性

```
初始检索（快速但粗糙）              重排序（慢但精确）
┌────────────────────────┐         ┌────────────────────────┐
│ 查询 → 向量检索 → Top-50│   →    │ Top-50 → Cross-Encoder │
│ (Bi-Encoder, 毫秒级)    │         │ → Top-5 (精确排序)      │
└────────────────────────┘         └────────────────────────┘
```

### Bi-Encoder vs Cross-Encoder

| 特性 | Bi-Encoder | Cross-Encoder |
|------|------------|---------------|
| 编码方式 | 查询和文档独立编码 | 查询和文档一起编码 |
| 计算方式 | 向量点积/余弦相似度 | 深度交互注意力 |
| 速度 | 快（可预计算文档） | 慢（需要实时计算） |
| 精度 | 中等 | 高 |
| 适用场景 | 初始检索 | 重排序 |

### 完整实现代码

```python
from langchain.docstore.document import Document
from langchain.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from sentence_transformers import CrossEncoder
from typing import List, Tuple, Any
import numpy as np

# ========== 方法 1: LLM 重排序 ==========

class RelevanceScore(BaseModel):
    score: float = Field(description="相关性分数，0-10 分")

class LLMBasedReranker:
    """基于 LLM 的重排序器"""

    def __init__(self, llm: ChatOpenAI = None, top_k: int = 5):
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=100)
        self.top_k = top_k

        self.rerank_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个信息检索专家。请评估以下文档与查询的相关性。

评分标准：
- 10: 完美匹配，直接回答问题
- 8-9: 高度相关，包含关键信息
- 6-7: 中等相关，部分信息有用
- 4-5: 略微相关，有少量有用信息
- 0-3: 不相关

只输出分数（0-10 的数字）。"""),
            ("human", """查询：{query}
文档：{document}

相关性分数：""")
        ])

        self.rerank_chain = self.rerank_prompt | llm.with_structured_output(RelevanceScore)

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        重排序文档

        流程：
        1. 对每个文档评分
        2. 按分数排序
        3. 返回 Top-K
        """
        scored_docs = []

        for doc in documents:
            result = self.rerank_chain.invoke({
                "query": query,
                "document": doc.page_content[:2000]  # 限制长度
            })
            score = result.score
            scored_docs.append((doc, score))

        # 按分数降序排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 返回 Top-K 文档
        return [doc for doc, _ in scored_docs[:self.top_k]]


# ========== 方法 2: Cross-Encoder 重排序 ==========

class CrossEncoderReranker:
    """基于 Cross-Encoder 的重排序器"""

    def __init__(
        self,
        model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        top_k: int = 5
    ):
        """
        Args:
            model_name: Cross-Encoder 模型名称
            top_k: 返回的文档数量
        """
        self.model = CrossEncoder(model_name)
        self.top_k = top_k

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        使用 Cross-Encoder 重排序

        流程：
        1. 构建查询 - 文档对
        2. 批量预测相关性分数
        3. 排序并返回 Top-K
        """
        # 构建查询 - 文档对
        pairs = [[query, doc.page_content] for doc in documents]

        # 预测分数
        scores = self.model.predict(pairs)

        # 排序
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 返回 Top-K
        return [doc for doc, _ in scored_docs[:self.top_k]]

    def rerank_with_scores(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """返回带分数的文档"""
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)

        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs[:self.top_k]


# ========== 方法 3: 混合重排序（两阶段） ==========

class HybridReranker:
    """混合重排序器：先用 Cross-Encoder 粗排，再用 LLM 精排"""

    def __init__(
        self,
        cross_encoder_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        llm: ChatOpenAI = None,
        ce_top_k: int = 20,
        final_top_k: int = 5
    ):
        self.ce_reranker = CrossEncoderReranker(
            model_name=cross_encoder_model,
            top_k=ce_top_k
        )
        self.llm_reranker = LLMBasedReranker(llm=llm, top_k=final_top_k)
        self.final_top_k = final_top_k

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        两阶段重排序：
        1. Cross-Encoder 从 N 个文档中选出 Top-20
        2. LLM 从 Top-20 中选出 Top-5
        """
        # 阶段 1: Cross-Encoder 粗排
        ce_results = self.ce_reranker.rerank(query, documents)

        # 阶段 2: LLM 精排
        final_results = self.llm_reranker.rerank(query, ce_results)

        return final_results


# ========== 自定义 Retriever 集成 ==========

class RerankingRetriever(BaseRetriever):
    """带重排序的检索器（可直接替换标准检索器）"""

    base_vectorstore: Any = Field(description="基础向量库")
    reranker: Any = Field(description="重排序器")
    initial_k: int = Field(description="初始检索数量")
    final_k: int = Field(description="重排序后返回数量")

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        # 初始检索
        initial_docs = self.base_vectorstore.similarity_search(query, k=self.initial_k)

        # 重排序
        reranked_docs = self.reranker.rerank(query, initial_docs)

        return reranked_docs[:self.final_k]


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 准备文档
    documents = [
        Document(page_content="机器学习是 AI 的分支..."),
        Document(page_content="深度学习使用神经网络..."),
        # ... 更多文档
    ]

    query = "机器学习和深度学习有什么区别？"

    # 方法 1: LLM 重排序
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_reranker = LLMBasedReranker(llm=llm, top_k=3)
    reranked_llm = llm_reranker.rerank(query, documents)

    print("LLM 重排序结果:")
    for i, doc in enumerate(reranked_llm):
        print(f"{i+1}. {doc.page_content[:50]}...")

    # 方法 2: Cross-Encoder 重排序
    ce_reranker = CrossEncoderReranker(
        model_name='cross-encoder/ms-marco-MiniLM-L-6-v2',
        top_k=3
    )
    reranked_ce = ce_reranker.rerank(query, documents)

    print("\nCross-Encoder 重排序结果:")
    for i, doc in enumerate(reranked_ce):
        print(f"{i+1}. {doc.page_content[:50]}...")

    # 方法 3: 混合重排序
    hybrid_reranker = HybridReranker(
        ce_top_k=20,
        final_top_k=3
    )
    reranked_hybrid = hybrid_reranker.rerank(query, documents)

    print("\n混合重排序结果:")
    for i, doc in enumerate(reranked_hybrid):
        print(f"{i+1}. {doc.page_content[:50]}...")

    # 集成到检索器
    from langchain.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    reranking_retriever = RerankingRetriever(
        base_vectorstore=vectorstore,
        reranker=ce_reranker,
        initial_k=20,
        final_k=5
    )

    results = reranking_retriever.get_relevant_documents(query)
```

### 重排序模型推荐

| 模型 | 语言 | 大小 | 速度 | 精度 | 适用场景 |
|------|------|------|------|------|----------|
| ms-marco-MiniLM-L-6-v2 | 英文 | 小 | 快 | 好 | 通用英文 |
| bge-reranker-base | 多语言 | 中 | 中 | 很好 | 中英混合 |
| bge-reranker-large | 多语言 | 大 | 慢 | 优秀 | 高精度需求 |
| Cohere Rerank | 多语言 | API | API | 优秀 | 不想本地部署 |

---

# 高级架构层：自反思与代理式系统

## 5.1 Self-RAG：自反思 RAG

### 核心思想

Self-RAG 引入**反思标记（Reflection Tokens）**，让模型学会自我评估：

```
┌─────────────────────────────────────────────────────────────┐
│                    Self-RAG 决策流程                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  用户查询                                                    │
│     ↓                                                       │
│  ┌─────────────────┐                                        │
│  │ 检索必要性判断   │                                        │
│  │ "我需要检索吗？" │                                        │
│  └───────┬─────────┘                                        │
│          ↓                                                  │
│     Yes / No                                                │
│     ↓                                                        │
│  如果需要检索：                                              │
│  ┌─────────────────┐                                        │
│  │   文档检索      │                                        │
│  └───────┬─────────┘                                        │
│          ↓                                                  │
│  ┌─────────────────┐                                        │
│  │ 相关性评估      │                                        │
│  │ "这文档相关吗？" │                                        │
│  └───────┬─────────┘                                        │
│          ↓                                                  │
│     Relevant / Irrelevant                                   │
│     ↓                                                        │
│  ┌─────────────────┐                                        │
│  │   答案生成      │                                        │
│  └───────┬─────────┘                                        │
│          ↓                                                  │
│  ┌─────────────────┐                                        │
│  │ 支持度评估      │                                        │
│  │ "答案有依据吗？" │                                        │
│  └───────┬─────────┘                                        │
│          ↓                                                  │
│  Fully Supported / Partially / No Support                   │
│          ↓                                                  │
│  ┌─────────────────┐                                        │
│  │ 效用评估        │                                        │
│  │ "答案有用吗？"   │                                        │
│  └───────┬─────────┘                                        │
│          ↓                                                  │
│  最终答案（选择最佳支持度和效用的）                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 完整实现代码

```python
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Tuple, Optional
import numpy as np

# ========== 1. 定义响应类型 ==========

class RetrievalDecision(BaseModel):
    response: str = Field(..., description="Yes 或 No")

class RelevanceDecision(BaseModel):
    response: str = Field(..., description="Relevant 或 Irrelevant")

class GenerationResponse(BaseModel):
    response: str = Field(..., description="生成的答案")

class SupportDecision(BaseModel):
    response: str = Field(..., description="Fully supported, Partially supported, 或 No support")

class UtilityDecision(BaseModel):
    response: int = Field(..., description="1-5 的效用评分")


# ========== 2. Self-RAG 实现 ==========

class SelfRAG:
    """Self-RAG 自反思检索增强生成系统"""

    def __init__(
        self,
        vectorstore: FAISS,
        llm: ChatOpenAI = None,
        top_k: int = 3
    ):
        self.vectorstore = vectorstore
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.top_k = top_k

        # 初始化所有决策链
        self._init_chains()

    def _init_chains(self):
        """初始化所有决策链"""

        # 1. 检索必要性判断
        retrieval_prompt = ChatPromptTemplate.from_messages([
            ("system", "给定以下查询，判断是否需要检索外部知识来回答。只输出'Yes'或'No'。"),
            ("human", "查询：{query}")
        ])
        self.retrieval_chain = retrieval_prompt | self.llm.with_structured_output(RetrievalDecision)

        # 2. 相关性评估
        relevance_prompt = ChatPromptTemplate.from_messages([
            ("system", "判断以下文档是否与查询相关。只输出'Relevant'或'Irrelevant'。"),
            ("human", "查询：{query}\n文档：{context}")
        ])
        self.relevance_chain = relevance_prompt | self.llm.with_structured_output(RelevanceDecision)

        # 3. 答案生成
        generation_prompt = ChatPromptTemplate.from_messages([
            ("system", "基于以下上下文回答问题。如果上下文不足，请说明。"),
            ("human", "查询：{query}\n上下文：{context}\n答案：")
        ])
        self.generation_chain = generation_prompt | self.llm.with_structured_output(GenerationResponse)

        # 4. 支持度评估
        support_prompt = ChatPromptTemplate.from_messages([
            ("system", "判断以下答案是否被上下文支持。输出'Fully supported'、'Partially supported'或'No support'。"),
            ("human", "答案：{response}\n上下文：{context}")
        ])
        self.support_chain = support_prompt | self.llm.with_structured_output(SupportDecision)

        # 5. 效用评估
        utility_prompt = ChatPromptTemplate.from_messages([
            ("system", "评估以下答案对查询的效用，1-5 分（5 分最高）。"),
            ("human", "查询：{query}\n答案：{response}")
        ])
        self.utility_chain = utility_prompt | self.llm.with_structured_output(UtilityDecision)

    def query(self, query: str, verbose: bool = False) -> str:
        """
        执行 Self-RAG 查询

        流程：
        1. 判断是否需要检索
        2. 如果需要，检索并评估相关性
        3. 生成答案
        4. 评估支持度和效用
        5. 选择最佳答案
        """
        if verbose:
            print(f"\n处理查询：{query}")

        # 步骤 1: 判断是否需要检索
        if verbose:
            print("步骤 1: 判断检索必要性...")

        retrieval_result = self.retrieval_chain.invoke({"query": query})
        need_retrieval = retrieval_result.response.strip().lower() == "yes"

        if verbose:
            print(f"检索必要性：{retrieval_result.response}")

        if not need_retrieval:
            # 直接生成
            if verbose:
                print("无需检索，直接生成...")
            result = self.generation_chain.invoke({"query": query, "context": "无需外部知识"})
            return result.response

        # 步骤 2: 检索文档
        if verbose:
            print("步骤 2: 检索文档...")

        docs = self.vectorstore.similarity_search(query, k=self.top_k)
        contexts = [doc.page_content for doc in docs]

        # 步骤 3: 评估相关性
        if verbose:
            print("步骤 3: 评估相关性...")

        relevant_contexts = []
        for i, context in enumerate(contexts):
            relevance_result = self.relevance_chain.invoke({
                "query": query,
                "context": context
            })
            is_relevant = relevance_result.response.strip().lower() == "relevant"

            if verbose:
                print(f"文档{i+1}相关性：{relevance_result.response}")

            if is_relevant:
                relevant_contexts.append(context)

        if not relevant_contexts:
            if verbose:
                print("无相关文档，直接生成...")
            return self.generation_chain.invoke({"query": query, "context": "无相关文档"}).response

        # 步骤 4: 为每个相关上下文生成答案并评估
        if verbose:
            print("步骤 4: 生成答案并评估...")

        responses = []
        for i, context in enumerate(relevant_contexts):
            if verbose:
                print(f"处理上下文{i+1}...")

            # 生成答案
            gen_result = self.generation_chain.invoke({
                "query": query,
                "context": context
            })
            answer = gen_result.response

            # 评估支持度
            support_result = self.support_chain.invoke({
                "response": answer,
                "context": context
            })
            support = support_result.response.strip().lower()

            # 评估效用
            utility_result = self.utility_chain.invoke({
                "query": query,
                "response": answer
            })
            utility = utility_result.response

            if verbose:
                print(f"支持度：{support}, 效用：{utility}")

            responses.append((answer, support, utility))

        # 步骤 5: 选择最佳答案
        if verbose:
            print("步骤 5: 选择最佳答案...")

        # 评分：Fully supported=2, Partially=1, No=0
        support_scores = {
            "fully supported": 2,
            "partially supported": 1,
            "no support": 0
        }

        best_response = max(
            responses,
            key=lambda x: (support_scores.get(x[1], 0), x[2])
        )

        if verbose:
            print(f"最佳答案支持度：{best_response[1]}, 效用：{best_response[2]}")

        return best_response[0]


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 初始化
    from langchain_openai import OpenAIEmbeddings
    from langchain.vectorstores import FAISS

    embeddings = OpenAIEmbeddings()
    documents = [
        Document(page_content="气候变化的主要原因是温室气体排放增加..."),
        Document(page_content="可再生能源包括太阳能、风能、水能等..."),
        # ... 更多文档
    ]
    vectorstore = FAISS.from_documents(documents, embeddings)

    self_rag = SelfRAG(vectorstore=vectorstore, llm=ChatOpenAI(model="gpt-4o-mini"))

    # 测试查询
    query1 = "气候变化的主要原因是什么？"  # 需要检索
    response1 = self_rag.query(query1, verbose=True)
    print(f"\n答案：{response1}")

    query2 = "你好啊！"  # 不需要检索
    response2 = self_rag.query(query2, verbose=True)
    print(f"\n答案：{response2}")
```

---

## 5.2 CRAG：纠正式 RAG

### 核心思想

CRAG（Corrective RAG）在 Self-RAG 的基础上增加了**动态纠正机制**：当检索结果质量不佳时，能够主动采取纠正措施。

```
┌─────────────────────────────────────────────────────────────┐
│                    CRAG 决策流程                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  用户查询                                                    │
│     ↓                                                       │
│  ┌─────────────────┐                                        │
│  │  向量库检索     │                                        │
│  └───────┬─────────┘                                        │
│          ↓                                                  │
│  ┌─────────────────┐                                        │
│  │  质量评估       │                                        │
│  │ (0-1 相关性分数)│                                        │
│  └───────┬─────────┘                                        │
│          ↓                                                  │
│    ┌─────┼─────┐                                            │
│    ↓     ↓     ↓                                            │
│  >0.7  0.3-0.7 <0.3                                         │
│  高    中    低                                             │
│    ↓     ↓     ↓                                            │
│  直接使用  混合  Web 搜索                                     │
│  生成    生成   生成                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 完整实现代码

```python
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.tools import DuckDuckGoSearchResults
from typing import List, Tuple, Dict, Any

# ========== 1. 定义评估和输出类型 ==========

class RetrievalEvaluatorInput(BaseModel):
    relevance_score: float = Field(..., description="0-1 之间的相关性分数")

class KnowledgeRefinementInput(BaseModel):
    key_points: str = Field(description="提取的关键点，用换行分隔")

class QueryRewriterInput(BaseModel):
    query: str = Field(description="重写后的查询")


# ========== 2. CRAG 核心组件 ==========

class CRAGProcessor:
    """纠正式 RAG 处理器"""

    def __init__(
        self,
        vectorstore: FAISS,
        llm: ChatOpenAI = None,
        high_threshold: float = 0.7,
        low_threshold: float = 0.3
    ):
        """
        Args:
            vectorstore: 向量数据库
            llm: 语言模型
            high_threshold: 高相关性阈值
            low_threshold: 低相关性阈值
        """
        self.vectorstore = vectorstore
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

        # 初始化搜索工具
        self.search = DuckDuckGoSearchResults()

        # 初始化功能链
        self._init_chains()

    def _init_chains(self):
        """初始化功能链"""

        # 1. 检索评估器
        eval_prompt = PromptTemplate(
            input_variables=["query", "document"],
            template="在 0-1 的范围内，评估以下文档与查询的相关性。\n查询：{query}\n文档：{document}\n相关性分数："
        )
        self.evaluator_chain = eval_prompt | self.llm.with_structured_output(RetrievalEvaluatorInput)

        # 2. 知识精炼器
        refine_prompt = PromptTemplate(
            input_variables=["document"],
            template="从以下文档中提取关键信息点，用换行分隔：\n{document}\n关键点："
        )
        self.refinement_chain = refine_prompt | self.llm.with_structured_output(KnowledgeRefinementInput)

        # 3. 查询重写器
        rewrite_prompt = PromptTemplate(
            input_variables=["query"],
            template="重写以下查询，使其更适合网络搜索：\n{query}\n重写后的查询："
        )
        self.rewrite_chain = rewrite_prompt | self.llm.with_structured_output(QueryRewriterInput)

        # 4. 答案生成器
        answer_prompt = PromptTemplate(
            input_variables=["query", "knowledge", "sources"],
            template="基于以下知识回答问题。在答案末尾注明来源。\n查询：{query}\n知识：{knowledge}\n来源：{sources}\n答案："
        )
        self.answer_chain = answer_prompt | self.llm

    def _retrieve_documents(self, query: str, k: int = 3) -> List[str]:
        """检索文档"""
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def _evaluate_documents(self, query: str, documents: List[str]) -> List[float]:
        """评估文档相关性"""
        scores = []
        for doc in documents:
            result = self.evaluator_chain.invoke({
                "query": query,
                "document": doc
            })
            scores.append(result.relevance_score)
        return scores

    def _refine_knowledge(self, document: str) -> List[str]:
        """精炼知识"""
        result = self.refinement_chain.invoke({"document": document})
        return [point.strip() for point in result.key_points.split('\n') if point.strip()]

    def _rewrite_query(self, query: str) -> str:
        """重写查询"""
        result = self.rewrite_chain.invoke({"query": query})
        return result.query.strip()

    def _web_search(self, query: str) -> Tuple[List[str], List[Dict]]:
        """
        执行网络搜索

        Returns:
            (精炼知识列表，来源信息列表)
        """
        rewritten_query = self._rewrite_query(query)
        search_results = self.search.run(rewritten_query)

        # 精炼结果
        refined = self._refine_knowledge(search_results)

        # 解析来源（简化处理）
        sources = [{"title": "网络搜索", "link": ""}]

        return refined, sources

    def _generate_answer(self, query: str, knowledge: str, sources: List[Dict]) -> str:
        """生成答案"""
        sources_str = "\n".join([f"{s.get('title', 'Unknown')}: {s.get('link', '')}" for s in sources])
        result = self.answer_chain.invoke({
            "query": query,
            "knowledge": knowledge,
            "sources": sources_str
        })
        return result.content

    def process(self, query: str, verbose: bool = False) -> str:
        """
        处理查询的完整流程

        决策逻辑：
        - 最高分 > 0.7: 直接使用检索结果
        - 最高分 < 0.3: 使用网络搜索
        - 0.3 <= 最高分 <= 0.7: 混合检索结果和网络搜索
        """
        if verbose:
            print(f"\n处理查询：{query}")

        # 1. 检索文档
        if verbose:
            print("步骤 1: 检索文档...")
        retrieved_docs = self._retrieve_documents(query, k=3)

        # 2. 评估文档
        if verbose:
            print("步骤 2: 评估文档...")
        eval_scores = self._evaluate_documents(query, retrieved_docs)

        if verbose:
            print(f"检索到 {len(retrieved_docs)} 个文档")
            print(f"相关性分数：{eval_scores}")

        # 3. 根据分数决定行动
        max_score = max(eval_scores)
        sources = []

        if max_score > self.high_threshold:
            # 高相关性：直接使用
            if verbose:
                print(f"\n行动：高相关性 ({max_score:.2f} > {self.high_threshold})，使用检索文档")
            best_doc = retrieved_docs[eval_scores.index(max_score)]
            final_knowledge = best_doc
            sources.append({"title": "本地知识库", "link": ""})

        elif max_score < self.low_threshold:
            # 低相关性：使用网络搜索
            if verbose:
                print(f"\n行动：低相关性 ({max_score:.2f} < {self.low_threshold})，使用网络搜索")
            final_knowledge, sources = self._web_search(query)

        else:
            # 中等相关性：混合
            if verbose:
                print(f"\n行动：中等相关性 ({self.low_threshold} <= {max_score:.2f} <= {self.high_threshold})，混合使用")
            best_doc = retrieved_docs[eval_scores.index(max_score)]
            retrieved_knowledge = self._refine_knowledge(best_doc)
            web_knowledge, web_sources = self._web_search(query)
            final_knowledge = "\n".join(retrieved_knowledge + web_knowledge)
            sources = [{"title": "本地知识库", "link": ""}] + web_sources

        if verbose:
            print(f"\n最终知识:\n{final_knowledge}")
            print(f"\n来源:\n{sources}")

        # 4. 生成答案
        if verbose:
            print("\n生成答案...")
        answer = self._generate_answer(query, final_knowledge, sources)

        if verbose:
            print(f"\n答案：{answer}")

        return answer


# ========== 使用示例 ==========

if __name__ == "__main__":
    from langchain_openai import OpenAIEmbeddings
    from langchain.vectorstores import FAISS

    # 准备知识库（假设是关于气候变化的文档）
    embeddings = OpenAIEmbeddings()
    documents = [
        Document(page_content="气候变化的主要原因是温室气体排放..."),
        Document(page_content="全球变暖导致极端天气事件增加..."),
    ]
    vectorstore = FAISS.from_documents(documents, embeddings)

    # 初始化 CRAG
    crag = CRAGProcessor(
        vectorstore=vectorstore,
        llm=ChatOpenAI(model="gpt-4o-mini"),
        high_threshold=0.7,
        low_threshold=0.3
    )

    # 测试 1: 高相关性查询
    query1 = "气候变化的主要原因是什么？"
    answer1 = crag.process(query1, verbose=True)

    # 测试 2: 低相关性查询（知识库中没有）
    query2 = "如何安装 Python 的 pip？"
    answer2 = crag.process(query2, verbose=True)

    # 测试 3: 中等相关性查询
    query3 = "可再生能源有哪些类型？"
    answer3 = crag.process(query3, verbose=True)
```

---

# 实战调优指南

## 调优参数总览

| 模块 | 关键参数 | 调优策略 |
|------|----------|----------|
| 分块 | chunk_size, chunk_overlap | 根据文档结构和查询类型调整 |
| 嵌入 | 模型选择，维度 | 中文选 BGE，英文选 OpenAI |
| 检索 | k 值，相似度阈值 | 根据上下文窗口和精度需求调整 |
| 重排序 | 初始 k，重排序 k | 20→5 或 50→10是常用组合 |
| 融合 | alpha, rrf_k | 0.5 平衡，0.3 重关键词，0.7 重语义 |

## 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 检索结果不相关 | 嵌入模型不匹配 | 换用更适合的嵌入模型 |
| 答案有幻觉 | 上下文不足或质量差 | 增加 k 值，添加重排序 |
| 响应慢 | 检索文档太多 | 减小 k，优化重排序策略 |
| 遗漏关键信息 | 分块过大或过小 | 调整 chunk_size，尝试语义分块 |

---

## 前沿趋势与未来方向

### 1. 多模态 RAG
- 图像 + 文本联合检索
- 视频内容理解与检索

### 2. 端侧 RAG
- 小型化模型部署
- 隐私保护的本地处理

### 3. 自更新 RAG
- 自动知识库更新
- 增量索引构建

### 4. 推理优化 RAG
- 推理路径追踪
- 可解释性增强

---

**报告完成**。本研究报告持续更新，反映 RAG 领域最新进展。
