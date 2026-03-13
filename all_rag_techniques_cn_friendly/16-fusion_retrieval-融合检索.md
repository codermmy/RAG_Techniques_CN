# 🌟 新手入门：文档搜索中的融合检索

> **💡 给新手的说明**
> - **难度级别**：⭐⭐⭐ 中级（需要了解向量搜索和基础检索概念）
> - **预计学习时间**：50-60 分钟
> - **前置知识**：了解 Embedding、向量存储、关键词搜索的基本概念
> - **本教程你将学会**：如何结合语义搜索和关键词搜索，让检索效果更上一层楼

---

## 📖 核心概念理解

### 什么是融合检索？

想象你在找人：

**语义检索（向量搜索）**：像朋友介绍
```
你："我想找个会编程的人"
朋友："那小王不错，虽然他简历上没写'编程'两个字，但他经常做项目"
→ 理解"意图"，不依赖精确匹配
```

**关键词检索（BM25）**：像搜索引擎
```
你："找会 Python 的人"
搜索引擎：匹配简历中包含"Python"这个词的人
→ 精确匹配关键词，但不理解同义词
```

**融合检索**：两者结合
```
既理解你的意图，又能精确匹配关键词
→ 两全其美！
```

### 通俗理解

```
查询："苹果公司的手机怎么样"

纯语义检索可能找到：
✅ "iPhone 的使用体验"（理解苹果=Apple）
❌ "水果苹果的营养价值"（可能误匹配）
✅ " Cupertino 科技公司的产品"（语义相关）

纯关键词检索可能找到：
✅ "苹果公司"（精确匹配）
✅ "苹果手机"（精确匹配）
❌ "iPhone 评测"（没出现"苹果"这个词）

融合检索：
✅ "苹果公司的手机怎么样"（两者都匹配）
✅ "iPhone 使用体验"（语义匹配）
✅ "苹果公司新产品"（关键词 + 语义）
```

### 为什么需要融合？

| 检索类型 | 优点 | 缺点 |
|---------|------|------|
| **语义检索（向量）** | 理解同义词、语义相关 | 可能漏掉精确关键词匹配 |
| **关键词检索（BM25）** | 精确匹配、可解释性强 | 无法理解同义词和语义 |
| **融合检索** | 两者优势结合 | 实现稍复杂 |

### BM25 是什么？

**BM25**（Best Matching 25）是一种经典的关键词搜索算法，是 TF-IDF 的改进版。

```
BM25 的核心思想：
1. 词频（TF）：一个词在文档中出现越多，越相关
2. 逆文档频率（IDF）：一个词在越少文档中出现，越重要
3. 文档长度归一化：短文档中的匹配词权重更高

简单理解：
- "RAG" 出现在 3 个文档中 → 很能区分
- "的" 出现在所有文档中 → 没啥区分度
```

---

## 🛠️ 第一步：安装必要的包

### 💻 完整代码

```python
# 安装所需的包
# langchain: RAG 系统核心框架
# numpy: 数值计算库，用于得分计算
# rank-bm25: BM25 算法实现
!pip install langchain numpy python-dotenv rank-bm25
```

> **⚠️ 新手注意**
> - `rank-bm25` 是一个轻量级的 BM25 实现，适合中小规模数据
> - 国内用户可使用清华镜像源加速安装

### 导入必要的库

```python
import os
import sys
from dotenv import load_dotenv
from langchain.docstore.document import Document

from typing import List
from rank_bm25 import BM25Okapi  # BM25 算法实现
import numpy as np  # 数值计算

# 导入辅助函数
from helper_functions import *
from evaluation.evalute_rag import *

# 加载环境变量
load_dotenv()

# 设置 OpenAI API 密钥
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
```

> **💡 代码解释**
> - `BM25Okapi`：rank-bm25 库中最常用的 BM25 实现
> - `numpy`：用于高效的数值计算和数组操作

---

## 📂 第二步：准备和编码 PDF 文档

### 💻 完整代码

```python
# 创建 data 目录
import os
os.makedirs('data', exist_ok=True)

# 下载教程使用的 PDF 文档
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
```

### 定义文件路径

```python
path = "data/Understanding_Climate_Change.pdf"
```

---

## 🔧 第三步：创建向量存储和 BM25 索引

### 📖 这是什么？

我们需要创建**两套索引**：
1. **向量索引**（FAISS）：用于语义检索
2. **BM25 索引**：用于关键词检索

### 💻 完整代码

```python
def encode_pdf_and_get_split_documents(path, chunk_size=1000, chunk_overlap=200):
    """
    使用 OpenAI embeddings 将 PDF 书籍编码为向量存储。

    Args:
        path: PDF 文件的路径。
        chunk_size: 每个文本块的期望大小。
        chunk_overlap: 连续块之间的重叠量。

    Returns:
        包含编码后书籍内容的 FAISS 向量存储，以及分块后的文档列表。
    """
    # 导入必要的库
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain.vectorstores import FAISS

    # ========== 步骤 1：加载 PDF 文档 ==========
    loader = PyPDFLoader(path)
    documents = loader.load()

    # ========== 步骤 2：分割文档 ==========
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,      # 每块 1000 字符
        chunk_overlap=chunk_overlap, # 重叠 200 字符
        length_function=len
    )
    texts = text_splitter.split_documents(documents)

    # 清理文本（替换某些特殊字符）
    cleaned_texts = replace_t_with_space(texts)

    # ========== 步骤 3：创建 embeddings 和向量存储 ==========
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore, cleaned_texts
```

> **💡 代码解释**

```
函数返回两个东西：

1. vectorstore (FAISS 向量存储)
   └─ 用途：语义检索
   └─ 原理：把文本转成向量，计算向量相似度

2. cleaned_texts (分块后的文档列表)
   └─ 用途：创建 BM25 索引
   └─ 内容：和向量存储用的是同一份文本
```

### 创建向量存储

```python
# 创建向量存储并获取分块文档
vectorstore, cleaned_texts = encode_pdf_and_get_split_documents(path)

print(f"向量存储创建完成")
print(f"文档被分成了 {len(cleaned_texts)} 个块")
```

---

## 📚 第四步：创建 BM25 索引

### 📖 这是什么？

BM25 索引是对文本进行分词后建立的关键词索引，用于快速的关键词检索。

### 💻 完整代码

```python
def create_bm25_index(documents: List[Document]) -> BM25Okapi:
    """
    从给定文档创建 BM25 索引。

    BM25 (Best Matching 25) 是信息检索中使用的排序函数。
    它基于概率检索框架，是 TF-IDF 的改进版本。

    Args:
    documents (List[Document]): 要索引的文档列表。

    Returns:
    BM25Okapi: 可用于 BM25 评分的索引。
    """
    # 通过空格分割对每个文档进行分词
    # 这是一种简单的方法，可以使用更复杂的分词改进
    tokenized_docs = [doc.page_content.split() for doc in documents]

    # 创建并返回 BM25 索引
    return BM25Okapi(tokenized_docs)
```

> **💡 代码解释**

```
BM25 索引创建过程：

原始文档：
["Climate change is caused by human activities",
 "Renewable energy helps combat climate change"]

分词后：
[["Climate", "change", "is", "caused", "by", "human", "activities"],
 ["Renewable", "energy", "helps", "combat", "climate", "change"]]

BM25 索引内部会统计：
- 每个词出现在哪些文档
- 每个词的词频
- 每个词的逆文档频率 (IDF)

查询时：
输入："climate change" → 分词 → ["climate", "change"]
→ 查找索引 → 计算每个文档的 BM25 得分
```

> **⚠️ 新手注意**
> - 这里用的是简单的空格分词，适合英文
> - 中文需要使用专门的分词工具（如 jieba）
> - `.split()` 是 Python 字符串方法，按空格分割

### 创建 BM25 索引

```python
# 从清理后的文本块创建 BM25 索引
bm25 = create_bm25_index(cleaned_texts)

print("BM25 索引创建完成！")
```

---

## 🔍 第五步：实现融合检索函数（核心）

### 📖 这是什么？

这是整个教程的**核心函数**！它同时执行语义检索和关键词检索，然后合并结果。

### 💻 完整代码

```python
def fusion_retrieval(vectorstore, bm25, query: str, k: int = 5, alpha: float = 0.5) -> List[Document]:
    """
    执行融合检索，结合基于关键词（BM25）和基于向量的搜索。

    Args:
    vectorstore (VectorStore): 包含文档的向量存储。
    bm25 (BM25Okapi): 预计算的 BM25 索引。
    query (str): 查询字符串。
    k (int): 要检索的文档数量。
    alpha (float): 向量搜索得分的权重（1-alpha 将为 BM25 得分的权重）。
                   alpha=0.5 表示两者权重相等
                   alpha=0.8 表示更重视语义搜索
                   alpha=0.2 表示更重视关键词匹配

    Returns:
    List[Document]: 基于组合得分的前 k 个文档。
    """
    # 防止除零错误的小数值
    epsilon = 1e-8

    # ========== 步骤 1：从向量存储获取所有文档 ==========
    # 我们需要所有文档来计算得分，所以先全部取出来
    # 注意：这在文档量大时可能效率不高，生产环境需要优化
    all_docs = vectorstore.similarity_search("", k=vectorstore.index.ntotal)

    # ========== 步骤 2：执行 BM25 搜索 ==========
    # 对查询进行分词，然后计算每个文档的 BM25 得分
    bm25_scores = bm25.get_scores(query.split())

    # ========== 步骤 3：执行向量搜索 ==========
    # 获取所有文档与查询的向量相似度得分
    vector_results = vectorstore.similarity_search_with_score(query, k=len(all_docs))

    # ========== 步骤 4：归一化得分 ==========
    # 为什么需要归一化？
    # 因为 BM25 得分和向量相似度得分的量纲不同，需要转到同一尺度

    # 提取向量得分
    vector_scores = np.array([score for _, score in vector_results])

    # 向量相似度得分归一化到 [0, 1]
    # 注意：向量相似度越小表示越相似，所以要 1 - 归一化值
    vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores) + epsilon)

    # BM25 得分归一化到 [0, 1]
    # BM25 得分越大表示越相关，所以直接归一化
    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + epsilon)

    # ========== 步骤 5：组合得分 ==========
    # 加权平均：alpha * 向量得分 + (1-alpha) * BM25 得分
    combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores

    # ========== 步骤 6：对文档排序 ==========
    # argsort 返回排序后的索引，[::-1] 反转实现降序
    sorted_indices = np.argsort(combined_scores)[::-1]

    # ========== 步骤 7：返回前 k 个文档 ==========
    return [all_docs[i] for i in sorted_indices[:k]]
```

> **💡 代码解释**

### 得分归一化详解

```
问题：为什么需要归一化？

假设：
- 向量相似度得分范围：0.0 ~ 1.0
- BM25 得分范围：0 ~ 100

直接相加会有问题：
文档 A: 向量 0.9 + BM25 10 = 10.9  ← BM25 主导
文档 B: 向量 0.95 + BM25 5 = 5.95
→ 虽然 A 向量稍差，但 BM25 好很多，应该 A 排前面

归一化后：
文档 A: 向量 0.9 + BM25 1.0 = 1.9
文档 B: 向量 0.95 + BM25 0.5 = 1.45
→ 合理！
```

### Alpha 参数详解

```
alpha 控制两种检索的权重：

alpha = 0.5（默认）：
  最终得分 = 0.5 × 向量得分 + 0.5 × BM25 得分
  两者平等，平衡语义和关键词

alpha = 0.8：
  最终得分 = 0.8 × 向量得分 + 0.2 × BM25 得分
  更重视语义理解，适合模糊查询

alpha = 0.2：
  最终得分 = 0.2 × 向量得分 + 0.8 × BM25 得分
  更重视关键词匹配，适合精确查询

alpha = 1.0：
  最终得分 = 1.0 × 向量得分 + 0.0 × BM25 得分
  退化为纯语义检索

alpha = 0.0：
  最终得分 = 0.0 × 向量得分 + 1.0 × BM25 得分
  退化为纯关键词检索
```

> **⚠️ 新手注意**
> - `epsilon = 1e-8` 是为了防止分母为 0，这是数值计算的常见技巧
> - `np.argsort()` 返回的是排序后的索引，不是排序后的值
> - `[::-1]` 是 Python 切片语法，用于反转列表

---

## 🧪 第六步：测试融合检索

### 💻 完整代码

```python
# 定义查询
query = "What are the impacts of climate change on the environment?"

# 执行融合检索
# k=5 返回前 5 个最相关的文档
# alpha=0.5 表示语义和关键词权重相等
top_docs = fusion_retrieval(vectorstore, bm25, query, k=5, alpha=0.5)

# 提取文档内容
docs_content = [doc.page_content for doc in top_docs]

# 显示检索结果
show_context(docs_content)
```

### 预期输出

```
检索到的相关内容：

[1] Climate change has significant impacts on the environment,
    affecting ecosystems, biodiversity, and natural resources.
    Rising temperatures lead to melting ice caps and glaciers...

[2] The effects of global warming on ecosystems include changes
    in species distribution, altered migration patterns, and
    disruptions to food chains...

[3] Environmental impacts also include more frequent and severe
    weather events, such as hurricanes, droughts, and heatwaves...

[4] Ocean acidification, caused by increased CO2 absorption,
    threatens marine life and coral reefs...

[5] Changes in precipitation patterns affect freshwater
    availability and can lead to both flooding and water
    scarcity...
```

---

## 📊 第七步：对比不同检索方法

### 💻 完整代码

```python
# 对比纯语义检索、纯 BM25 和融合检索

query = "What are the impacts of climate change on the environment?"

print("=" * 70)
print(f"查询：{query}\n")

# ========== 1. 纯语义检索 ==========
print("【纯语义检索】")
semantic_docs = vectorstore.similarity_search(query, k=5)
for i, doc in enumerate(semantic_docs, 1):
    print(f"{i}. {doc.page_content[:100]}...")

print("\n" + "=" * 70)

# ========== 2. 纯 BM25 检索 ==========
print("【纯 BM25 检索】")
bm25_scores = bm25.get_scores(query.split())
sorted_indices = np.argsort(bm25_scores)[::-1]
for i, idx in enumerate(sorted_indices[:5], 1):
    print(f"{i}. {cleaned_texts[idx].page_content[:100]}...")

print("\n" + "=" * 70)

# ========== 3. 融合检索 ==========
print("【融合检索 (alpha=0.5)】")
fusion_docs = fusion_retrieval(vectorstore, bm25, query, k=5, alpha=0.5)
for i, doc in enumerate(fusion_docs, 1):
    print(f"{i}. {doc.page_content[:100]}...")

print("\n" + "=" * 70)
```

### 预期对比结果

```
【纯语义检索】
1. 文档谈论环境影响的语义相关内容
2. 可能包含同义词但没出现"impacts"这个词
3. ...

【纯 BM25 检索】
1. 包含"impacts"、"environment"关键词的文档
2. 关键词出现频率越高排名越靠前
3. ...

【融合检索】
1. 既包含关键词，语义又相关的文档
2. 综合排名最靠前
3. ...
```

---

## 🎯 第八步：探索不同 Alpha 值的影响

### 💻 完整代码

```python
# 测试不同 alpha 值对检索结果的影响

query = "What are the impacts of climate change on the environment?"

print(f"查询：{query}\n")
print("=" * 70)

# 测试不同的 alpha 值
alpha_values = [0.0, 0.2, 0.5, 0.8, 1.0]

for alpha in alpha_values:
    top_docs = fusion_retrieval(vectorstore, bm25, query, k=3, alpha=alpha)
    print(f"\n【Alpha = {alpha}】")
    if alpha == 0.0:
        print("(纯 BM25)")
    elif alpha == 1.0:
        print("(纯语义)")
    else:
        print(f"(语义={alpha}, BM25={1-alpha})")

    for i, doc in enumerate(top_docs, 1):
        print(f"  {i}. {doc.page_content[:80]}...")
```

### 预期结果分析

```
Alpha 对结果的影响：

Alpha = 0.0 (纯 BM25):
  → 返回包含最多查询关键词的文档
  → 适合精确查询、专业术语搜索

Alpha = 0.2 (偏 BM25):
  → 关键词匹配为主，语义为辅
  → 适合需要精确匹配但也要一定灵活性的场景

Alpha = 0.5 (平衡):
  → 语义和关键词平衡
  → 通用场景的推荐选择

Alpha = 0.8 (偏语义):
  → 语义理解为主，关键词为辅
  → 适合模糊查询、同义词匹配

Alpha = 1.0 (纯语义):
  → 只考虑语义相似度
  → 适合概念性查询、探索性搜索
```

---

## 📈 完整代码整合

```python
# ========== 1. 安装和导入 ==========
!pip install langchain numpy python-dotenv rank-bm25

import os
from dotenv import load_dotenv
from langchain.docstore.document import Document
from typing import List
from rank_bm25 import BM25Okapi
import numpy as np
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# ========== 2. 编码 PDF 并创建向量存储 ==========
def encode_pdf_and_get_split_documents(path, chunk_size=1000, chunk_overlap=200):
    loader = PyPDFLoader(path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore, cleaned_texts

path = "data/Understanding_Climate_Change.pdf"
vectorstore, cleaned_texts = encode_pdf_and_get_split_documents(path)

# ========== 3. 创建 BM25 索引 ==========
def create_bm25_index(documents: List[Document]) -> BM25Okapi:
    tokenized_docs = [doc.page_content.split() for doc in documents]
    return BM25Okapi(tokenized_docs)

bm25 = create_bm25_index(cleaned_texts)

# ========== 4. 融合检索函数 ==========
def fusion_retrieval(vectorstore, bm25, query: str, k: int = 5, alpha: float = 0.5) -> List[Document]:
    epsilon = 1e-8

    # 获取所有文档
    all_docs = vectorstore.similarity_search("", k=vectorstore.index.ntotal)

    # BM25 得分
    bm25_scores = bm25.get_scores(query.split())

    # 向量相似度得分
    vector_results = vectorstore.similarity_search_with_score(query, k=len(all_docs))
    vector_scores = np.array([score for _, score in vector_results])

    # 归一化
    vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores) + epsilon)
    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + epsilon)

    # 组合得分
    combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores

    # 排序并返回前 k 个
    sorted_indices = np.argsort(combined_scores)[::-1]
    return [all_docs[i] for i in sorted_indices[:k]]

# ========== 5. 测试 ==========
query = "What are the impacts of climate change on the environment?"
top_docs = fusion_retrieval(vectorstore, bm25, query, k=5, alpha=0.5)

print(f"查询：{query}")
print(f"\n检索到的文档：")
for i, doc in enumerate(top_docs, 1):
    print(f"{i}. {doc.page_content[:100]}...")
```

---

## ⚠️ 常见问题及解决方法

### 问题 1：文档量大时速度慢

**原因**：`fusion_retrieval` 函数中获取了所有文档 (`ntotal`)

**解决方法**：
```python
# 优化方案：先分别检索，再合并
def optimized_fusion_retrieval(vectorstore, bm25, query: str, k: int = 5, alpha: float = 0.5):
    # 1. 语义检索前 100 个候选
    semantic_results = vectorstore.similarity_search_with_score(query, k=100)
    semantic_docs = [doc for doc, _ in semantic_results]
    semantic_indices = set(range(100))

    # 2. BM25 检索前 100 个候选
    bm25_scores = bm25.get_scores(query.split())
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:100]
    bm25_docs = [cleaned_texts[i] for i in bm25_top_indices]

    # 3. 取并集
    all_candidate_indices = semantic_indices.union(set(bm25_top_indices))
    all_docs = [cleaned_texts[i] for i in all_candidate_indices]

    # 4. 在候选集上计算组合得分
    # ...（省略具体实现）

    return top_k_docs
```

### 问题 2：中文分词问题

**问题**：`.split()` 对中文无效

**解决方法**：
```python
# 使用 jieba 分词
import jieba

def create_bm25_index_chinese(documents: List[Document]) -> BM25Okapi:
    # 使用 jieba 进行中文分词
    tokenized_docs = [list(jieba.cut(doc.page_content)) for doc in documents]
    return BM25Okapi(tokenized_docs)
```

### 问题 3：得分分布不均匀

**问题**：一种方法的得分总是主导结果

**解决方法**：
```python
# 使用秩融合（Reciprocal Rank Fusion）
def reciprocal_rank_fusion(vectorstore, bm25, query: str, k: int = 5, r: float = 60.0):
    # 语义检索
    semantic_results = vectorstore.similarity_search(query, k=100)

    # BM25 检索
    bm25_scores = bm25.get_scores(query.split())
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:100]

    # RRF 融合
    doc_scores = {}

    for i, doc in enumerate(semantic_results):
        doc_id = id(doc)
        if doc_id not in doc_scores:
            doc_scores[doc_id] = {"doc": doc, "score": 0}
        doc_scores[doc_id]["score"] += 1.0 / (r + i)

    bm25_docs = [cleaned_texts[i] for i in bm25_top_indices]
    for i, doc in enumerate(bm25_docs):
        doc_id = id(doc)
        if doc_id not in doc_scores:
            doc_scores[doc_id] = {"doc": doc, "score": 0}
        doc_scores[doc_id]["score"] += 1.0 / (r + i)

    # 排序
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in sorted_docs[:k]]
```

---

## 🎓 学习总结

### 你学到了什么？

✅ **融合检索的概念**：结合语义检索和关键词检索的优势
✅ **BM25 算法**：经典的关键词检索方法
✅ **得分归一化**：为什么需要将不同检索的得分转到同一尺度
✅ **Alpha 参数**：如何调节两种检索方法的权重
✅ **完整实现**：从零开始实现融合检索系统

### 实际应用建议

| 应用场景 | 推荐 Alpha | 说明 |
|---------|-----------|------|
| 通用搜索 | 0.5 | 平衡语义和关键词 |
| 技术文档 | 0.3-0.4 | 偏重关键词精确匹配 |
| 客服问答 | 0.6-0.7 | 偏重语义理解 |
| 探索性搜索 | 0.8-0.9 | 重视语义相关 |
| 精确查询 | 0.1-0.3 | 重视关键词匹配 |

### 性能对比

```
检索方法对比（假设某测试集）：

纯语义检索:
- 准确率：75%
- 召回率：80%
- 优点：理解同义词
- 缺点：可能漏掉精确匹配

纯 BM25:
- 准确率：70%
- 召回率：65%
- 优点：精确匹配
- 缺点：无法理解语义

融合检索 (alpha=0.5):
- 准确率：82%
- 召回率：85%
- 优点：两者兼顾 ⭐
```

---

## 📚 相关资源

- [BM25 算法详解](https://en.wikipedia.org/wiki/Okapi_BM25)
- [rank-bm25 Python 库文档](https://github.com/dorianbrown/rank_bm25)
- [FAISS 向量搜索库](https://github.com/facebookresearch/faiss)
- [LangChain 向量存储文档](https://python.langchain.com/docs/modules/data_connection/vectorstores/)

---

*本教程是 RAG 技术系列教程之一。融合检索可以与重排序、上下文压缩等技术结合使用，构建更强大的检索系统。*

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--fusion-retrieval)
