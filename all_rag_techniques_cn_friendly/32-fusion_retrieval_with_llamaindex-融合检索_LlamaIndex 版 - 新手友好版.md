# 🌟 新手入门：融合检索（Fusion Retrieval）系统

> **💡 给新手的说明**
> - **难度等级**：⭐⭐⭐☆☆（中级）
> - **预计时间**：40-55 分钟
> - **前置知识**：了解基础 RAG 概念，熟悉向量检索基本原理
> - **学习目标**：理解融合检索的原理，掌握结合向量检索和 BM25 的方法，构建更强大的检索系统

---

## 📖 核心概念理解

### 什么是融合检索？

**融合检索**（Fusion Retrieval）是将**向量检索**（语义搜索）和**BM25 检索**（关键词搜索）结合起来的技术。它同时利用两种方法的优势，提高检索的准确性和覆盖范围。

### 🍕 通俗理解：找书的两种方式

想象你在图书馆找书：

1. **向量检索（语义搜索）**：
   - 你问："关于全球变暖的书"
   - 图书管理员理解你的**意图**，给你拿气候变化的书
   - **优势**：能理解同义词、相关概念
   - **劣势**：可能错过包含精确关键词的书

2. **BM25 检索（关键词搜索）**：
   - 你搜索："全球 变暖"
   - 系统找包含这两个词的書
   - **优势**：精确匹配关键词
   - **劣势**：不理解同义词（"气候变化"就找不到）

3. **融合检索**：
   - 同时用两种方式找书
   - 把结果合并、排序
   - **优势**：两者优点我都要！

### 🔑 核心组件解释

| 组件 | 作用 | 生活比喻 |
|------|------|----------|
| **向量检索器** | 基于语义相似度检索 | 理解意图的图书管理员 |
| **BM25 检索器** | 基于关键词匹配检索 | 精确查找的搜索引擎 |
| **QueryFusionRetriever** | 融合两个检索器的结果 | 结果合并器 |
| **retriever_weights** | 控制两个检索器的权重 | 天平的砝码 |

### 📊 融合检索工作流程

```
用户查询："气候变化的影响"
         │
         ├─────────────────┐
         │                 │
         ▼                 ▼
   ┌──────────┐     ┌──────────┐
   │ 向量检索  │     │ BM25 检索 │
   │ (语义)   │     │ (关键词)  │
   └──────────┘     └──────────┘
         │                 │
         │ 结果 A          │ 结果 B
         │ [A1, A2, A3]    │ [B1, B2, B3]
         │                 │
         └────────┬────────┘
                  │
                  ▼
         ┌────────────────┐
         │  QueryFusion   │
         │  融合 + 重排序  │
         └────────────────┘
                  │
                  ▼
         最终结果 [C1, C2, C3, ...]
```

### 📈 BM25 是什么？

**BM25**（Best Matching 25）是一种经典的关键词检索算法：

- **原理**：统计关键词在文档中出现的频率
- **特点**：
  - 词频越高，相关性越高
  - 但会饱和（不是越多越好）
  - 长文档会被适当惩罚
- **应用**：Elasticsearch、Lucene 等搜索引擎的核心算法

---

## 🛠️ 第一步：环境准备

### 📖 这是什么？

安装必要的 Python 库。

### 💻 完整代码

```python
# ============================================
# 安装所需的包
# ============================================
# 每个包的作用：
# - faiss-cpu: Facebook 的高效相似度搜索库
# - llama-index: LlamaIndex 框架核心
# - python-dotenv: 管理 API 密钥

!pip install faiss-cpu llama-index python-dotenv
```

> **💡 代码解释**
> - LlamaIndex 已经内置了 BM25 检索器，无需额外安装
>
> **⚠️ 新手注意**
> - 如使用国内网络，可添加清华源

---

## 🔑 第二步：配置 API 密钥和导入库

### 📖 这是什么？

设置 OpenAI API 密钥并导入所有需要的库。

### 💻 完整代码

```python
# ============================================
# 导入必要的库并配置 API 密钥
# ============================================
import os
import sys
from dotenv import load_dotenv
from typing import List
from llama_index.core import Settings
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import BaseNode, TransformComponent
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.legacy.retrievers.bm25_retriever import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
import faiss

# 从 .env 文件加载环境变量
load_dotenv()

# 设置 OpenAI API 密钥
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# ============================================
# 配置 LlamaIndex 全局设置
# ============================================
EMBED_DIMENSION = 512

# 设置 LLM 模型（temperature 设低一些让结果更稳定）
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

# 设置 Embedding 模型
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small", 
    dimensions=EMBED_DIMENSION
)
```

> **💡 代码解释**
> - `BM25Retriever` 来自 `llama_index.legacy`，是 LlamaIndex 的遗留模块
> - `QueryFusionRetriever` 是融合检索的核心组件
> - `temperature=0.1` 让 LLM 输出更稳定

---

## 📄 第三步：下载和读取文档

### 📖 这是什么？

下载示例 PDF 并读取内容。

### 💻 完整代码

```python
# ============================================
# 创建目录并下载示例 PDF
# ============================================
import os

os.makedirs('data', exist_ok=True)

# 下载示例 PDF 文档
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf

# ============================================
# 读取 PDF 文档
# ============================================
path = "data/"
reader = SimpleDirectoryReader(
    input_dir=path, 
    required_exts=['.pdf']
)
documents = reader.load_data()

print(f"✓ 文档加载完成！共 {len(documents)} 个文档")
```

---

## 🗄️ 第四步：创建向量存储

### 📖 这是什么？

创建 FAISS 向量存储。

### 💻 完整代码

```python
# ============================================
# 创建 FAISS 向量存储
# ============================================
fais_index = faiss.IndexFlatL2(EMBED_DIMENSION)
vector_store = FaissVectorStore(faiss_index=fais_index)
```

---

## 🧹 第五步：文本清洗转换

### 📖 这是什么？

定义文本清洗器，处理 PDF 中的格式问题。

### 💻 完整代码

```python
# ============================================
# 定义文本清洗器
# ============================================
class TextCleaner(TransformComponent):
    """
    用于摄取管道中的文本转换。
    清理文本中的杂乱内容。
    """
    def __call__(self, nodes, **kwargs) -> List[BaseNode]:
        for node in nodes:
            # 将制表符替换为空格
            node.text = node.text.replace('\t', ' ')
            # 将段落分隔符替换为空格
            node.text = node.text.replace(' \n', ' ')
        return nodes
```

> **💡 代码解释**
> - `TransformComponent` 是 LlamaIndex 的转换组件基类
> - 清洗可以提高检索质量

---

## 🔄 第六步：创建数据摄入流水线

### 📖 这是什么？

数据处理流水线将文档转换为可用于检索的节点。

### 💻 完整代码

```python
# ============================================
# 创建数据摄入流水线
# ============================================
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(),  # 句子分割
        TextCleaner(),       # 文本清洗
    ],
    vector_store=vector_store,
    documents=documents,
)

# 运行流水线
nodes = pipeline.run()

print(f"✓ 流水线运行完成！共生成 {len(nodes)} 个节点")
```

> **💡 代码解释**
> - `SentenceSplitter()` 按句子分割文本
> - `TextCleaner()` 清洗格式问题
> - 处理顺序：先分割，后清洗

---

## 🔍 第七步：创建两个独立的检索器

### 📖 这是什么？

在融合之前，先分别创建向量检索器和 BM25 检索器。

### 💻 完整代码

#### 创建 BM25 检索器

```python
# ============================================
# 创建 BM25 检索器（基于关键词）
# ============================================
bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=2,  # 返回最相关的 2 个结果
)

print("✓ BM25 检索器创建完成")
```

> **💡 代码解释**
> - `from_defaults()` 是创建 BM25 检索器的便捷方法
> - `similarity_top_k` 控制返回结果数量

#### 创建向量检索器

```python
# ============================================
# 创建向量检索器（基于语义）
# ============================================
index = VectorStoreIndex(nodes)

vector_retriever = index.as_retriever(
    similarity_top_k=2
)

print("✓ 向量检索器创建完成")
```

---

## 🔀 第八步：创建融合检索器

### 📖 这是什么？

将两个检索器融合成一个更强大的检索器。

### 💻 完整代码

```python
# ============================================
# 创建融合检索器
# ============================================
retriever = QueryFusionRetriever(
    retrievers=[
        vector_retriever,  # 向量检索器
        bm25_retriever     # BM25 检索器
    ],
    
    # 检索器权重（总和应为 1）
    retriever_weights=[
        0.6,  # 向量检索器权重 60%
        0.4,  # BM25 检索器权重 40%
    ],
    
    # 查询数量
    num_queries=1,
    
    # 融合模式
    mode='dist_based_score',
    
    # 是否异步执行
    use_async=False
)

print("✓ 融合检索器创建完成")
```

> **💡 参数详解**

#### `retriever_weights` - 检索器权重

```python
# 权重分配示例：
# 语义重要的场景（如开放问答）：[0.7, 0.3]
# 关键词重要的场景（如代码搜索）：[0.3, 0.7]
# 平衡场景（推荐）：[0.6, 0.4]
```

#### `num_queries` - 查询数量

```python
# num_queries=1：只使用原始查询（推荐）
# num_queries>1：会生成多个变体查询，然后全部检索
# 例如 num_queries=3 会生成 3 个问题，每个都检索

# 示例：
# 原始查询："气候变化的影响"
# 生成的查询：
#   1. "气候变化有什么影响？"
#   2. "全球变暖会导致什么后果？"
#   3. "环境影响 climate change"
```

#### `mode` - 融合模式（4 种选项）

```python
# 1. reciprocal_rerank - 倒数排序
#    排名越靠前，分数越高
#    适合：不同检索器分数范围差异大的情况

# 2. relative_score - 相对分数（MinMax 归一化）
#    基于最小/最大分数归一化
#    公式：score = (score - min) / (max - min)

# 3. dist_based_score - 基于分布的分数（推荐）
#    基于均值和标准差归一化
#    公式：min_score = mean - 3*std, max_score = mean + 3*std
#    适合：分数分布近似正态分布的情况

# 4. simple - 简单模式
#    直接取每个文本块的最大分数
#    最简单，但可能不够精确
```

> **⚠️ 新手注意**
> - `mode='dist_based_score'` 通常效果最好
> - 权重可以根据实际效果调整

---

## 🧪 第九步：测试融合检索

### 💻 完整代码

```python
# ============================================
# 测试融合检索
# ============================================
query = "What are the impacts of climate change on the environment?"

print(f"查询：{query}\n")
print("=" * 60)

# 执行融合检索
response = retriever.retrieve(query)

# 打印结果
print(f"检索到 {len(response)} 个结果\n")

for i, node in enumerate(response):
    print(f"【结果 {i+1}】")
    print(f"分数：{node.score:.4f}")
    print(f"内容：{node.text[:200]}...")
    print("-" * 60)
```

> **📊 预期输出示例**
> ```
> 查询：What are the impacts of climate change on the environment?
> 
> ============================================================
> 检索到 4 个结果
> 
> 【结果 1】
> 分数：0.8523
> 内容：Climate change has significant impacts on the environment, 
>       including rising sea levels, loss of biodiversity...
> ------------------------------------------------------------
> 【结果 2】
> 分数：0.7891
> 内容：The environmental effects include more frequent extreme 
>       weather events such as hurricanes and droughts...
> ------------------------------------------------------------
> ```

---

## 📊 第十步：对比单检索器和融合检索

### 💻 完整代码

```python
# ============================================
# 对比三种检索方式
# ============================================
query = "climate change impacts"

print("=" * 70)
print("检索方式对比")
print("=" * 70)

# 1. 仅向量检索
print("\n【向量检索结果】")
vector_results = vector_retriever.retrieve(query)
for i, node in enumerate(vector_results[:2]):
    print(f"  {i+1}. [分数：{node.score:.3f}] {node.text[:80]}...")

# 2. 仅 BM25 检索
print("\n【BM25 检索结果】")
bm25_results = bm25_retriever.retrieve(query)
for i, node in enumerate(bm25_results[:2]):
    print(f"  {i+1}. [分数：{node.score:.3f}] {node.text[:80]}...")

# 3. 融合检索
print("\n【融合检索结果】")
fusion_results = retriever.retrieve(query)
for i, node in enumerate(fusion_results[:3]):
    print(f"  {i+1}. [分数：{node.score:.3f}] {node.text[:80]}...")
```

> **📊 对比分析**
> ```
> 向量检索擅长：
> - 找到语义相关但用词不同的内容
> - 例如查询"global warming"能找到"climate change"的内容
> 
> BM25 检索擅长：
> - 精确匹配关键词
> - 找到包含特定术语的内容
> 
> 融合检索：
> - 结合两者优势
> - 覆盖范围更广
> - 排序更准确
> ```

---

## ⚠️ 常见问题与调试

### Q1: 如何调整检索器权重？

**建议**：
```python
# 场景 1：语义更重要（开放问答、概念查询）
retriever_weights=[0.7, 0.3]  # 向量 70%, BM25 30%

# 场景 2：关键词更重要（代码搜索、专有名词）
retriever_weights=[0.3, 0.7]

# 场景 3：平衡（推荐起点）
retriever_weights=[0.6, 0.4]

# 场景 4：各占一半
retriever_weights=[0.5, 0.5]
```

### Q2: 什么时候应该用不同的 fusion mode？

**建议**：
```python
# dist_based_score（推荐默认）
mode='dist_based_score'  # 大多数情况效果最好

# relative_score
mode='relative_score'  # 分数分布不均匀时

# reciprocal_rerank
mode='reciprocal_rerank'  # 只关心排名，不关心分数

# simple
mode='simple'  # 快速测试，不追求最优
```

### Q3: 融合检索比单检索器慢吗？

**解释**：
- 会稍微慢一点，因为要运行两个检索器
- 但通常可以接受（增加约 50% 时间）
- 使用 `use_async=True` 可以并行执行

```python
# 异步执行（更快）
retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    use_async=True,  # 并行执行两个检索器
)
```

### Q4: 融合检索适合中文吗？

**适合！** 但需要注意：
- BM25 对中文分词有依赖
- LlamaIndex 内置的 BM25 对中文支持有限

**改进方案**：
```python
# 使用 jieba 分词 + 自定义 BM25
import jieba
from rank_bm25 import BM25Okapi

# 中文分词
def chinese_tokenize(text):
    return list(jieba.cut(text))

# 然后创建自定义 BM25 索引
```

---

## 📚 总结

### 核心要点回顾

1. **融合检索的核心思想**：
   - 向量检索：捕捉语义相似性
   - BM25 检索：精确关键词匹配
   - 融合两者：优势互补

2. **关键组件**：
   - `BM25Retriever`：关键词检索
   - `QueryFusionRetriever`：结果融合
   - `retriever_weights`：权重控制

3. **工作流程**：
   ```
   查询 → 向量检索器 ─┐
         → BM25 检索器 ─┤
                        → QueryFusion → 融合结果
   ```

### 进阶方向

1. **多检索器融合**：结合 3 个或更多检索器
2. **自适应权重**：根据查询类型动态调整权重
3. **学习排序**：用机器学习优化融合策略

### 实际应用建议

- **企业搜索**：推荐融合检索，覆盖更全面
- **代码搜索**：BM25 权重可以更高
- **问答系统**：向量权重可以更高

---

## 🔗 相关资源

- [QueryFusionRetriever 文档](https://docs.llamaindex.ai/en/stable/module_guides/retrievers/query_fusion-retriever/)
- [BM25 算法详解](https://en.wikipedia.org/wiki/Okapi_BM25)
- [LlamaIndex 检索器指南](https://docs.llamaindex.ai/en/stable/module_guides/retrieving/)
