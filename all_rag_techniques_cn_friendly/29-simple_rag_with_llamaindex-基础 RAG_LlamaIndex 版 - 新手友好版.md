# 🌟 新手入门：基础 RAG 系统（LlamaIndex 版）

> **💡 给新手的说明**
> - **难度等级**：⭐⭐☆☆☆（入门级）
> - **预计时间**：30-45 分钟
> - **前置知识**：基础 Python 编程知识
> - **学习目标**：理解 RAG 系统的基本原理，学会使用 LlamaIndex 构建 PDF 文档问答系统

---

## 📖 核心概念理解

### 什么是 RAG 系统？

**RAG**（Retrieval-Augmented Generation，检索增强生成）是一种让 AI 变得更"博学"的技术。

### 🍕 通俗理解：图书管理员比喻

想象一下你去图书馆问问题：

1. **普通 AI** 就像一个只靠记忆回答问题的图书管理员——他知道很多，但无法回答最新或很具体的问题
2. **RAG 系统** 就像一个会先查书再回答的图书管理员——他会先去书架上找到相关书籍，然后基于书的内容给你准确的答案

**RAG 的工作流程**：
```
你提问 → 系统查找相关文档 → 基于找到的内容回答 → 给你准确答案
```

### 🔑 核心组件解释

| 组件 | 作用 | 生活比喻 |
|------|------|----------|
| **SimpleDirectoryReader** | 读取文件夹中的文档 | 图书管理员收集书籍 |
| **SentenceSplitter** | 按句子分割文本 | 把书按章节分开 |
| **TextCleaner** | 清洗文本中的杂乱内容 | 整理书页上的污渍 |
| **OpenAI Embedding** | 将文字转成数字向量 | 给每本书贴上分类标签 |
| **FAISS 向量存储** | 存储和管理向量 | 图书馆的书架系统 |
| **检索器** | 查找最相关的文档块 | 图书管理员查找书籍 |

### 🆚 LlamaIndex vs LangChain

你可能听说过 LangChain，它们是类似的工具：

| 特性 | LlamaIndex | LangChain |
|------|-----------|-----------|
| **专注点** | 专注于数据索引和检索 | 更通用的 LLM 应用框架 |
| **API 设计** | 更简洁，面向对象 | 更灵活，函数式 |
| **学习曲线** | 相对平缓 | 稍陡峭 |
| **适用场景** | 文档问答、知识库 | 各种 LLM 应用 |

**新手建议**：两个框架都值得学习，本教程使用 LlamaIndex 是因为它的 API 更直观。

---

## 🛠️ 第一步：环境准备

### 📖 这是什么？

在开始之前，我们需要安装必要的 Python 库。就像做菜前要准备好厨具和食材一样。

### 💻 完整代码

```python
# ============================================
# 安装所需的包
# ============================================
# 每个包的作用：
# - faiss-cpu: Facebook 的高效相似度搜索库（用于向量检索）
# - llama-index: LlamaIndex 框架核心
# - python-dotenv: 管理 API 密钥等环境变量

!pip install faiss-cpu llama-index python-dotenv
```

> **💡 代码解释**
> - `!pip install` 是 Jupyter Notebook 中安装包的方式
> - 如果使用 Google Colab，部分包可能已经预装
>
> **⚠️ 新手注意**
> - 如果遇到安装失败，可以尝试逐个安装
> - 如使用国内网络，可添加清华源：`!pip install faiss-cpu -i https://pypi.tuna.tsinghua.edu.cn/simple`
> - LlamaIndex 安装包较大，安装可能需要几分钟

---

## 🔑 第二步：配置 API 密钥和导入库

### 📖 这是什么？

RAG 系统需要使用 OpenAI 的 API 来生成文本的向量表示（Embedding）。这一步就是设置你的 API 密钥并导入所需的库。

### 💻 完整代码

```python
# ============================================
# 导入必要的库并配置 API 密钥
# ============================================
from typing import List
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import BaseNode, TransformComponent
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
import faiss
import os
import sys
from dotenv import load_dotenv

# 配置参数
EMBED_DIMENSION = 512  # Embedding 向量维度

# 分块设置（与 LangChain 不同）
# LlamaIndex 使用 token 长度而非字符串长度
CHUNK_SIZE = 200       # 每块约 200 个 token
CHUNK_OVERLAP = 50     # 相邻块重叠 50 个 token

# 从 .env 文件加载环境变量
load_dotenv()

# 设置 OpenAI API 密钥
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# 在 LlamaIndex 全局设置中配置嵌入模型
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small", 
    dimensions=EMBED_DIMENSION
)
```

> **💡 代码解释**
> - `load_dotenv()` 从 `.env` 文件加载配置
> - `Settings.embed_model` 是 LlamaIndex 的全局配置，设置后所有地方都会使用
> - `text-embedding-3-small` 是 OpenAI 的轻量级 Embedding 模型，性价比高
>
> **⚠️ 新手注意**
> - **API 密钥安全**：永远不要把你的 API 密钥直接写在代码里提交到 Git！
> - 推荐使用 `.env` 文件：
>   ```
>   # .env 文件内容
>   OPENAI_API_KEY=sk-your-actual-key-here
>   ```
> - `.env` 文件应该添加到 `.gitignore` 中
>
> **📊 术语解释**
> - **Token**：可以理解为单词或字节的单位，英文约 4 个字符=1 token，中文约 1.5 个字=1 token
> - **Embedding**：将文字转换为数字向量的过程，相似的文本会有相似的向量表示
>
> **❓ 常见问题**
> - **Q: 我没有 OpenAI API 密钥怎么办？**
> - A: 可以注册 OpenAI 账号获取，或者使用其他 Embedding 模型（如 HuggingFace 的免费模型）

---

## 📄 第三步：下载和读取文档

### 📖 这是什么？

我们需要一个 PDF 文档来测试 RAG 系统。这里会下载一个关于气候变化的示例文档。

### 💻 完整代码

```python
# ============================================
# 创建 data 目录并下载示例 PDF
# ============================================
import os

# 创建 data 目录
os.makedirs('data', exist_ok=True)

# 下载示例 PDF 文档
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf

# 下载问答数据（用于后续评估）
!wget -O data/q_a.json https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/q_a.json
```

> **💡 代码解释**
> - `os.makedirs('data', exist_ok=True)` 创建 data 目录
> - `!wget` 从网络下载文件
> - `-O` 指定下载后的文件名

### 💻 读取文档

```python
# ============================================
# 使用 SimpleDirectoryReader 读取 PDF 文档
# ============================================
path = "data/"

# 创建文档读取器，指定只读取 PDF 文件
node_parser = SimpleDirectoryReader(
    input_dir=path, 
    required_exts=['.pdf']  # 只读取.pdf 文件
)

# 加载文档
documents = node_parser.load_data()

# 打印第一个文档的内容（预览）
print(documents[0])
```

> **💡 代码解释**
> - `SimpleDirectoryReader` 是 LlamaIndex 的文档加载器
> - `required_exts=['.pdf']` 限制只读取 PDF 文件
> - `documents` 是一个列表，包含所有加载的文档对象

> **⚠️ 新手注意**
> - 如果下载失败，可以手动下载 PDF 放到 `data` 目录下
> - 你也可以使用自己的 PDF 文件

---

## 🗄️ 第四步：创建向量存储

### 📖 这是什么？

向量存储是用来存储和快速检索文档向量的数据库。FAISS 是 Facebook 开源的高效相似度搜索库。

### 💻 完整代码

```python
# ============================================
# 创建 FAISS 向量存储
# ============================================
# 创建 FAISS 索引（使用 L2 距离）
faiss_index = faiss.IndexFlatL2(EMBED_DIMENSION)

# 创建 LlamaIndex 的 FAISS 向量存储包装器
vector_store = FaissVectorStore(faiss_index=faiss_index)
```

> **💡 代码解释**
> - `IndexFlatL2` 使用欧几里得距离（L2 距离）计算相似度
> - `EMBED_DIMENSION` 必须与 Embedding 模型的输出维度一致
> - `FaissVectorStore` 是 LlamaIndex 对 FAISS 的封装

> **📊 术语解释**
> - **FAISS**：Facebook AI Similarity Search，专门用于向量相似度搜索的库
> - **L2 距离**：欧几里得距离，计算两个向量之间的直线距离

---

 🧹 第五步：文本清洗转换

### 📖 这是什么？

PDF 文件中常有一些格式问题（如多余的制表符、换行符等），需要清洗以获得更好的处理效果。

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
        # 遍历所有节点
        for node in nodes:
            # 将制表符替换为空格
            node.text = node.text.replace('\t', ' ')
            # 将段落分隔符替换为空格
            node.text = node.text.replace(' \n', ' ')
        return nodes
```

> **💡 代码解释**
> - `TransformComponent` 是 LlamaIndex 的转换组件基类
> - `__call__` 方法定义了如何处理文本节点
> - 替换 `\t` 和 ` \n` 是为了处理 PDF 中常见的格式问题

> **⚠️ 新手注意**
> - 如果你的 PDF 格式很干净，这一步可以简化或跳过
> - 可以根据实际需要添加更多清洗规则

---

## 🔄 第六步：创建数据摄入流水线

### 📖 这是什么？

数据摄入流水线（Ingestion Pipeline）是 LlamaIndex 的核心概念，它将文档处理、转换和存储整合到一个流程中。

### 💻 完整代码

```python
# ============================================
# 创建文本分割器
# ============================================
text_splitter = SentenceSplitter(
    chunk_size=CHUNK_SIZE,       # 每块约 200 个 token
    chunk_overlap=CHUNK_OVERLAP  # 重叠 50 个 token
)

# ============================================
# 创建数据摄入流水线
# ============================================
pipeline = IngestionPipeline(
    transformations=[
        TextCleaner(),      # 第一步：清洗文本
        text_splitter,      # 第二步：分割文本
    ],
    vector_store=vector_store,  # 存储到向量数据库
)
```

> **💡 代码解释**
> - `SentenceSplitter` 按句子边界分割文本，保持语义完整性
> - `transformations` 列表定义了处理顺序
> - `vector_store` 指定存储位置

> **📊 术语解释**
> - **Chunk（分块）**：将长文档切成小块，便于处理和检索
> - **Overlap（重叠）**：相邻块之间的重叠部分，避免信息被切断

### 💻 运行流水线

```python
# ============================================
# 运行流水线，生成向量存储
# ============================================
nodes = pipeline.run(documents=documents)

print(f"✓ 处理完成！共生成 {len(nodes)} 个文本块")
```

> **💡 代码解释**
> - `pipeline.run()` 执行整个处理流程
> - `nodes` 是处理后生成的文本块列表
> - 每个 node 包含文本内容和元数据

---

## 🔍 第七步：创建检索器

### 📖 这是什么？

检索器是用来查询向量存储、获取相关文档的接口。

### 💻 完整代码

```python
# ============================================
# 从节点创建向量存储索引
# ============================================
vector_store_index = VectorStoreIndex(nodes)

# ============================================
# 创建检索器
# ============================================
# similarity_top_k=2 表示每次检索返回最相似的 2 个结果
retriever = vector_store_index.as_retriever(similarity_top_k=2)
```

> **💡 代码解释**
> - `VectorStoreIndex` 是基于向量存储的索引
> - `as_retriever()` 将索引转换为检索器接口
> - `similarity_top_k` 控制返回结果的数量

> **⚠️ 新手注意**
> - `top_k` 值越大，返回结果越多，但可能包含不相关信息
> - 一般设置为 2-5 之间

---

## 🧪 第八步：测试检索器

### 📖 这是什么？

测试检索器是否能正确找到相关文档。

### 💻 完整代码

```python
# ============================================
# 辅助函数：显示检索到的上下文
# ============================================
def show_context(context):
    """
    显示提供的上下文列表的内容。
    
    参数：
        context (list): 要显示的上下文列表
    """
    for i, c in enumerate(context):
        print(f"Context {i+1}:")
        print(c.text)
        print("\n")


# ============================================
# 测试查询
# ============================================
test_query = "气候变化的主要原因是什么？"

# 执行检索
context = retriever.retrieve(test_query)

# 显示结果
show_context(context)
```

> **💡 代码解释**
> - `retrieve()` 方法执行检索，返回相关文档列表
> - `show_context()` 是辅助函数，用于格式化显示结果

> **📊 预期输出示例**
> ```
> Context 1:
> Climate change is primarily caused by human activities, particularly 
> the burning of fossil fuels...
> 
> Context 2:
> The greenhouse effect is intensified by carbon dioxide emissions 
> from industrial processes...
> ```

---

## 📈 第九步：评估 RAG 系统（可选）

### 📖 这是什么？

评估是检验 RAG 系统性能的重要步骤。这里使用 DeepEval 框架进行自动化评估。

### 💻 完整代码

```python
# ============================================
# 导入评估相关的库
# ============================================
import json
from deepeval import evaluate
from deepeval.metrics import GEval, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCaseParams
from evaluation.evalute_rag import create_deep_eval_test_cases

# ============================================
# 配置评估参数
# ============================================
LLM_MODEL = "gpt-4o"  # 用于评估的模型

# 定义评估指标
correctness_metric = GEval(
    name="Correctness",
    model=LLM_MODEL,
    evaluation_params=[
        LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    evaluation_steps=[
        "根据期望输出确定实际输出在事实上是否正确。"
    ],
)

faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model=LLM_MODEL,
    include_reason=False
)

relevance_metric = ContextualRelevancyMetric(
    threshold=1,
    model=LLM_MODEL,
    include_reason=True
)


# ============================================
# 定义评估函数
# ============================================
def evaluate_rag(query_engine, num_questions: int = 5) -> None:
    """
    使用预定义指标评估 RAG 系统。
    
    参数：
        query_engine: 查询引擎，用于回答问题和获取检索上下文
        num_questions (int): 要评估的问题数量（默认：5）
    """
    # 从 JSON 文件加载问题和答案
    q_a_file_name = "data/q_a.json"
    with open(q_a_file_name, "r", encoding="utf-8") as json_file:
        q_a = json.load(json_file)
    
    questions = [qa["question"] for qa in q_a][:num_questions]
    ground_truth_answers = [qa["answer"] for qa in q_a][:num_questions]
    generated_answers = []
    retrieved_documents = []
    
    # 为每个问题生成答案并检索文档
    for question in questions:
        response = query_engine.query(question)
        context = [doc.text for doc in response.source_nodes]
        retrieved_documents.append(context)
        generated_answers.append(response.response)
    
    # 创建测试用例并评估
    test_cases = create_deep_eval_test_cases(
        questions, 
        ground_truth_answers, 
        generated_answers, 
        retrieved_documents
    )
    
    evaluate(
        test_cases=test_cases,
        metrics=[correctness_metric, faithfulness_metric, relevance_metric]
    )
```

> **📊 评估指标解释**
> - **Correctness（正确性）**：答案是否与标准答案一致
> - **Faithfulness（忠实度）**：答案是否基于检索到的内容
> - **Contextual Relevance（上下文相关性）**：检索到的文档是否与问题相关

### 💻 运行评估

```python
# ============================================
# 创建查询引擎并运行评估
# ============================================
query_engine = vector_store_index.as_query_engine(similarity_top_k=2)

# 评估 1 个问题（完整评估可以设置 num_questions=5）
evaluate_rag(query_engine, num_questions=1)
```

> **⚠️ 新手注意**
> - 评估需要使用 GPT-4，会消耗较多的 API 配额
> - 初次学习可以跳过评估步骤
> - 评估结果会显示各项指标的得分

---

## ⚠️ 常见问题与调试

### Q1: 安装 LlamaIndex 时遇到依赖冲突怎么办？

**解决方案**：
```bash
# 尝试升级 pip
pip install --upgrade pip

# 或者使用虚拟环境
python -m venv rag_env
source rag_env/bin/activate  # Windows: rag_env\Scripts\activate
pip install faiss-cpu llama-index python-dotenv
```

### Q2: 检索结果不相关怎么办？

**可能的原因**：
1. `top_k` 值太小
2. 分块大小不合适
3. Embedding 模型不适合你的领域

**解决方案**：
- 尝试增加 `similarity_top_k` 到 5 或 10
- 调整 `CHUNK_SIZE` 和 `CHUNK_OVERLAP`
- 尝试其他 Embedding 模型

### Q3: 如何处理多个 PDF 文件？

**解决方案**：
```python
# SimpleDirectoryReader 会自动读取目录下所有 PDF
node_parser = SimpleDirectoryReader(
    input_dir="data/", 
    required_exts=['.pdf']
)
documents = node_parser.load_data()
# 这会自动读取 data 目录下所有.pdf 文件
```

---

## 📚 总结

### 核心要点回顾

1. **RAG 工作流程**：文档加载 → 文本清洗 → 分割 → 向量化 → 存储 → 检索
2. **LlamaIndex 核心概念**：
   - `SimpleDirectoryReader`：文档加载
   - `IngestionPipeline`：数据处理流水线
   - `VectorStoreIndex`：向量索引
   - `Retriever`：检索接口
3. **关键参数**：分块大小、重叠度、检索数量

### 进阶方向

1. **自定义分块策略**：根据文档类型调整分块方式
2. **多模态 RAG**：处理包含图像的文档
3. **对话式 RAG**：支持多轮对话的问答系统

---

## 🔗 相关资源

- [LlamaIndex 官方文档](https://docs.llamaindex.ai/)
- [FAISS 文档](https://faiss.ai/)
- [DeepEval 评估框架](https://docs.deepeval.com/)
