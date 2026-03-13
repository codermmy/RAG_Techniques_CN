# 🌟 新手入门：RAG 系统中的重排序（Reranking）技术

> **💡 给新手的说明**
> - **难度等级**：⭐⭐⭐☆☆（中级）
> - **预计时间**：40-55 分钟
> - **前置知识**：了解基础 RAG 概念，熟悉向量检索基本原理
> - **学习目标**：理解重排序的原理和作用，掌握 LLM 和 Cross-Encoder 两种重排序方法

---

## 📖 核心概念理解

### 什么是重排序（Reranking）？

**重排序**是在初始检索之后，对检索结果进行"二次筛选"和"重新排序"的过程。它使用更精确的模型重新评估每个文档的相关性，确保最相关的信息排在最前面。

### 🍕 通俗理解：选秀比赛比喻

想象一个选秀比赛：

1. **初始检索（海选）**：
   - 100 个参赛者上台表演
   - 评委快速选出 10 个进入下一轮
   - 标准：整体印象、基本条件
   - 类似：向量检索快速找出可能相关的文档

2. **重排序（决赛）**：
   - 10 个参赛者逐一展示
   - 评委仔细打分、排名
   - 标准：细节表现、综合素质
   - 类似：重排序模型精确评估相关性

3. **最终结果**：
   - 选出前 3 名
   - 这 3 名是最优秀的

**RAG 中的重排序**：
```
查询 → 检索 10 个文档 → 重排序 → 选前 5 个 → 生成答案
       (粗选)          (精选)    (最优)
```

### 🔑 核心组件解释

| 组件 | 作用 | 生活比喻 |
|------|------|----------|
| **初始检索器** | 快速获取候选文档 | 海选评委 |
| **LLMRerank** | 用 LLM 重排序 | 专家评委仔细打分 |
| **SentenceTransformerRerank** | 用 Cross-Encoder 模型重排序 | 专业评分系统 |
| **top_n** | 选择前 N 个结果 | 选出前几名 |

### 📊 重排序前后对比

```
【重排序前】（按向量相似度）
1. [分数 0.85] 气候变化影响环境。
2. [分数 0.82] 全球变暖导致海平面上升。
3. [分数 0.79] 生物多样性受到威胁。
4. [分数 0.75] 极端天气事件增多。
5. [分数 0.72] 碳排放是主要原因。

查询："气候变化对生物多样性的影响"

【重排序后】（按实际相关性）
1. [分数 0.92] 生物多样性受到威胁。（最相关！）
2. [分数 0.88] 气候变化影响环境。
3. [分数 0.75] 极端天气事件增多。
4. [分数 0.68] 全球变暖导致海平面上升。
5. [分数 0.62] 碳排放是主要原因。

看到了吗？重排序把真正相关的内容排到了前面！
```

### 🆚 两种重排序方法对比

| 特性 | LLM 重排序 | Cross-Encoder 重排序 |
|------|-----------|---------------------|
| **原理** | 让 LLM 打分 | 专用重排序模型 |
| **精度** | 高 | 非常高 |
| **速度** | 较慢 | 较快 |
| **成本** | 需要调用 LLM API | 本地运行，免费 |
| **适用场景** | 需要理解复杂语义 | 追求速度和精度平衡 |

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
# - sentence-transformers: Cross-Encoder 模型（重排序用）

!pip install faiss-cpu llama-index python-dotenv
!pip install sentence-transformers
```

> **💡 代码解释**
> - `sentence-transformers` 提供 Cross-Encoder 模型
> - 如果只用 LLM 重排序，可以不用安装这个

> **⚠️ 新手注意**
> - `sentence-transformers` 首次使用会下载模型（约 100MB）
> - 如使用国内网络，下载可能较慢

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
from llama_index.core import Document
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank, LLMRerank
from llama_index.core import QueryBundle
import faiss

# 从 .env 文件加载环境变量
load_dotenv()

# 设置 OpenAI API 密钥
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# ============================================
# 配置 LlamaIndex 全局设置
# ============================================
EMBED_DIMENSION = 512

# 设置 LLM 模型
Settings.llm = OpenAI(model="gpt-3.5-turbo")

# 设置 Embedding 模型
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small", 
    dimensions=EMBED_DIMENSION
)
```

> **💡 代码解释**
> - `SentenceTransformerRerank`：Cross-Encoder 重排序器
> - `LLMRerank`：LLM 重排序器
> - `QueryBundle`：查询的封装格式

---

## 📄 第三步：下载和读取文档

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

print(f"✓ 文档加载完成！")
```

---

## 🗄️ 第四步：创建向量存储

### 💻 完整代码

```python
# ============================================
# 创建 FAISS 向量存储
# ============================================
fais_index = faiss.IndexFlatL2(EMBED_DIMENSION)
vector_store = FaissVectorStore(faiss_index=fais_index)
```

---

## 🔄 第五步：创建数据摄入流水线

### 💻 完整代码

```python
# ============================================
# 创建数据摄入流水线
# ============================================
base_pipeline = IngestionPipeline(
    transformations=[SentenceSplitter()],  # 句子分割
    vector_store=vector_store,
    documents=documents,
)

# 运行流水线
nodes = base_pipeline.run()

print(f"✓ 流水线运行完成！共生成 {len(nodes)} 个节点")
```

---

## 🎯 方法一：基于 LLM 的重排序

### 📖 这是什么？

使用大型语言模型（LLM）来评估每个文档的相关性并重新排序。

### 💻 完整代码

```python
# ============================================
# 创建带 LLM 重排序的查询引擎
# ============================================
index = VectorStoreIndex(nodes)

query_engine_w_llm_rerank = index.as_query_engine(
    similarity_top_k=10,  # 先检索 10 个候选
    node_postprocessors=[
        LLMRerank(
            top_n=5  # 重排序后选前 5 个
        )
    ],
)

print("✓ 带 LLM 重排序的查询引擎创建完成")
```

> **💡 代码解释**
> - `similarity_top_k=10`：初始检索 10 个文档
> - `LLMRerank(top_n=5)`：重排序后保留最相关的 5 个
> - `node_postprocessors`：节点后处理器，在检索后处理结果

### 💻 测试 LLM 重排序

```python
# ============================================
# 测试查询
# ============================================
resp = query_engine_w_llm_rerank.query(
    "What are the impacts of climate change on biodiversity?"
)

print(f"问题：What are the impacts of climate change on biodiversity?")
print(f"答案：{resp}")
```

### 💡 原理解析

```
LLM 重排序的工作流程：

1. 检索 10 个候选文档
        ↓
2. 对每个文档，LLM 被问："这个文档与查询的相关性如何？请打分 1-10"
        ↓
3. LLM 给出分数：[7, 5, 9, 6, 8, 4, 10, 3, 6, 5]
        ↓
4. 按分数排序：[10, 9, 8, 7, 6, ...]
        ↓
5. 选前 5 个送给生成器
```

---

## 🎯 方法二：基于 Cross-Encoder 的重排序

### 📖 这是什么？

Cross-Encoder 是专门训练用于评估文本对相关性的模型。它比 LLM 更快、更便宜，且效果通常更好。

### 💻 完整代码

```python
# ============================================
# 创建带 Cross-Encoder 重排序的查询引擎
# ============================================
query_engine_w_cross_encoder = index.as_query_engine(
    similarity_top_k=10,  # 先检索 10 个候选
    node_postprocessors=[
        SentenceTransformerRerank(
            model='cross-encoder/ms-marco-MiniLM-L-6-v2',  # 重排序模型
            top_n=5  # 重排序后选前 5 个
        )
    ],
)

print("✓ 带 Cross-Encoder 重排序的查询引擎创建完成")
```

> **💡 代码解释**
> - `cross-encoder/ms-marco-MiniLM-L-6-v2` 是微软训练的流行重排序模型
> - 这个模型在 MS MARCO 数据集上训练，适合通用文本重排序
> - 模型会自动下载（首次使用约 100MB）

### 💻 测试 Cross-Encoder 重排序

```python
# ============================================
# 测试查询
# ============================================
resp = query_engine_w_cross_encoder.query(
    "What are the impacts of climate change on biodiversity?"
)

print(f"问题：What are the impacts of climate change on biodiversity?")
print(f"答案：{resp}")
```

### 📊 Cross-Encoder 原理

```
Cross-Encoder 的工作方式：

查询："气候变化的影响"
文档："全球变暖导致海平面上升"

        ┌─────────────────┐
        │  Cross-Encoder  │
        │    模型输入      │
        │  [Query, Doc]   │
        └────────┬────────┘
                 │
                 ▼
           相关性分数
           0.85 (0-1 之间)

特点：
- 同时处理查询和文档（不是分别编码）
- 能捕捉两者之间的细微关系
- 比向量检索更精确，但比 LLM 快
```

---

## 🔬 演示：为什么需要重排序？

### 📖 这是什么？

通过一个具体例子展示重排序如何改善检索质量。

### 💻 完整代码

```python
# ============================================
# 创建示例文档
# ============================================
chunks = [
    "The capital of France is great.",
    "The capital of France is huge.",
    "The capital of France is beautiful.",
    """Have you ever visited Paris? It is a beautiful city where you can 
    eat delicious food and see the Eiffel Tower. I really enjoyed all 
    the cities in france, but its capital with the Eiffel Tower is my 
    favorite city.""", 
    "I really enjoyed my trip to Paris, France. The city is beautiful 
    and the food is delicious. I would love to visit again. Such a great 
    capital city."
]

docs = [Document(text=sentence) for sentence in chunks]


# ============================================
# 对比函数
# ============================================
def compare_rag_techniques(query: str, docs) -> None:
    """对比基础检索和重排序的效果"""
    
    # 创建索引
    index = VectorStoreIndex.from_documents(docs)
    
    print("=" * 60)
    print("检索技术对比")
    print("=" * 60)
    print(f"查询：{query}\n")
    
    # ========== 基础检索 ==========
    print("【基础检索结果】")
    baseline_docs = index.as_retriever(similarity_top_k=5).retrieve(query)
    for i, doc in enumerate(baseline_docs[:2]):
        print(f"\n文档 {i+1}:")
        print(f"  {doc.text[:100]}...")
    
    # ========== 带重排序的检索 ==========
    print("\n\n【带重排序的检索结果】")
    reranker = LLMRerank(top_n=2)
    advanced_docs = reranker.postprocess_nodes(
        baseline_docs, 
        QueryBundle(query)
    )
    for i, doc in enumerate(advanced_docs):
        print(f"\n文档 {i+1}（重排序后）:")
        print(f"  {doc.text[:100]}...")
        if hasattr(doc, 'score'):
            print(f"  重排序分数：{doc.score}")


# ============================================
# 运行对比
# ============================================
query = "what is the capital of france?"
compare_rag_techniques(query, docs)
```

> **📊 预期输出分析**
> ```
> 查询：what is the capital of france?
> 
> 【基础检索结果】
> 文档 1: The capital of France is great.
> 文档 2: The capital of France is huge.
> 
> 分析：这两句虽然包含关键词"capital of France"，但没有实际信息！
> 
> 【带重排序的检索结果】
> 文档 1（重排序后）: Have you ever visited Paris? It is a beautiful city...
> 文档 2（重排序后）: I really enjoyed my trip to Paris, France...
> 
> 分析：重排序识别出这些文档虽然关键词匹配度低，
>       但实际包含了"Paris 是法国首都"的信息！
> ```

> **💡 关键洞察**
> - 基础检索只看表面相似度
> - 重排序能理解深层语义
> - 重排序把真正有用的信息排到前面

---

## 📊 两种重排序方法对比

### 💻 对比代码

```python
# ============================================
# 两种方法的对比
# ============================================
print("=" * 70)
print("LLM 重排序 vs Cross-Encoder 重排序")
print("=" * 70)

query = "What are the impacts of climate change on biodiversity?"

print("\n【LLM 重排序】")
print("  优点：")
print("    - 能理解复杂的语义关系")
print("    - 可以处理需要推理的问题")
print("    - 灵活，可通过提示词调整")
print("  缺点：")
print("    - 速度较慢（需要调用 API）")
print("    - 成本较高（按 token 计费）")
print("    - 结果可能有随机性")
print("  适用场景：复杂问题、需要深度理解")

print("\n【Cross-Encoder 重排序】")
print("  优点：")
print("    - 速度快（本地运行）")
print("    - 免费（无需 API）")
print("    - 结果稳定")
print("    - 在 benchmarks 上表现优异")
print("  缺点：")
print("    - 模型固定，不易调整")
print("    - 对特定领域可能需要微调")
print("  适用场景：大多数通用场景、追求性价比")
```

---

## 🏆 常用的重排序模型推荐

### 免费开源模型

```python
# 轻量级（速度快）
model = 'cross-encoder/ms-marco-MiniLM-L-6-v2'  # 推荐入门

# 中型（平衡）
model = 'cross-encoder/ms-marco-TinyBERT-L-2-v2'

# 大型（精度高）
model = 'cross-encoder/ms-marco-electra-base'
```

### 多语言模型

```python
# 支持中文
model = 'BAAI/bge-reranker-base'  # 智源，支持中文

# 多语言
model = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
```

---

## ⚠️ 常见问题与调试

### Q1: 应该用 LLM 重排序还是 Cross-Encoder？

**建议**：
```python
# 追求性价比 → Cross-Encoder
SentenceTransformerRerank(model='cross-encoder/ms-marco-MiniLM-L-6-v2')

# 需要深度理解 → LLM
LLMRerank()

# 预算充足且要求最高精度 → 都试，选好的
```

### Q2: top_n 应该设多少？

**建议**：
```python
# 一般问答：top_n=3~5
# 需要更多上下文：top_n=5~10
# 精确答案：top_n=1~3

# 初始检索数量也建议设大一些
similarity_top_k=10~20  # 给重排序足够候选
```

### Q3: 重排序会让速度变慢多少？

**解释**：
```
LLM 重排序：
- 每个文档需要调用一次 LLM
- 10 个文档 ≈ 10 次 API 调用
- 增加约 5-10 秒

Cross-Encoder 重排序：
- 本地模型，批量处理
- 10 个文档 ≈ 0.5-1 秒
- 几乎可以忽略
```

### Q4: 重排序一定比不重排序好吗？

**不一定！**
- 如果初始检索已经很精确，重排序提升有限
- 如果文档质量差，重排序也无能为力
- 需要权衡速度和精度

---

## 📚 总结

### 核心要点回顾

1. **重排序的价值**：
   - 初始检索：快速获取候选（粗选）
   - 重排序：精确评估相关性（精选）
   - 结果：最相关的文档排前面

2. **两种方法**：
   - **LLM 重排序**：灵活、强大，但慢且贵
   - **Cross-Encoder**：快速、便宜、效果好

3. **使用建议**：
   - 默认推荐 Cross-Encoder
   - 复杂问题用 LLM
   - 初始检索数量要足够

### 进阶方向

1. **级联重排序**：先用 Cross-Encoder，再用 LLM
2. **领域适配**：在特定领域数据上微调模型
3. **多阶段检索**：多轮重排序逐步筛选

---

## 🔗 相关资源

- [LlamaIndex 重排序文档](https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors/)
- [SBERT 官网](https://www.sbert.net/)
- [HuggingFace 重排序模型](https://huggingface.co/models?pipeline_tag=text-classification&search=cross-encoder)

<div style="text-align: center;">
<img src="../images/reranking-visualization.svg" alt="重排序可视化" style="width:80%; height:auto;">
</div>

<div style="text-align: center;">
<img src="../images/reranking_comparison.svg" alt="重排序对比" style="width:80%; height:auto;">
</div>
