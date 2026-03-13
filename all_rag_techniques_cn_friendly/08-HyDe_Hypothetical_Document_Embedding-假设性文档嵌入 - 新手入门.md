# 🌟 新手入门：假设性文档嵌入 (HyDE)

> **💡 给新手的说明**
> - **难度等级**：⭐⭐⭐⭐（中高级）
> - **预计学习时间**：40-50 分钟
> - **前置知识**：了解基础的 Python 编程，对 RAG 系统有基本认识
> - **本教程你将学会**：如何用 HyDE 技术提升短查询的检索效果

---

## 📖 核心概念理解

### 什么是 HyDE？

**HyDE** = **Hy**pothetical **D**ocument **E**mbedding（假设性文档嵌入）

### 通俗理解

**生活化比喻**：

想象你在图书馆找书：

🔍 **传统方法**：
- 你问："气候变化原因？"（只有 6 个字）
- 图书管理员拿着这 6 个字去书架上找匹配的书
- 结果：很难找到完全匹配的书

💡 **HyDE 方法**：
- 你先让一个"专家"写一篇文章来回答"气候变化原因？"
- 这篇假设性文章可能有 500 字，包含"化石燃料"、"温室气体"、"工业排放"等关键词
- 图书管理员拿着这篇 500 字的文章去找相似的书
- 结果：更容易找到相关的书！

### 核心思想

```
┌─────────────────────────────────────────────────────────────────┐
│                      HyDE 工作流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  用户查询 ──→ [LLM 生成假设性文档] ──→ 假设性文档              │
│     ↓                                      ↓                   │
│  传统检索                              HyDE 检索                │
│  (短查询直接匹配)                      (用假设性文档匹配)        │
│     ↓                                      ↓                   │
│  可能找不到                            找到更多相关              │
│  相关文档                              文档                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 为什么 HyDE 有效？

| 问题 | 传统 RAG | HyDE |
|------|---------|------|
| **查询太短** | 6 个字的查询很难匹配长文档 | 扩展成 500 字的假设性文档 |
| **语义差距** | 查询和文档风格不同 | 假设性文档和真实文档风格一致 |
| **词汇不匹配** | 用户用的词≠文档用的词 | LLM 会生成包含各种相关词汇的文档 |

---

## 🛠️ 第一步：环境准备

### 📖 这是什么？

安装必要的 Python 包，并配置 API 密钥。

### 💻 完整代码

```python
# 安装所需的包
# !pip install python-dotenv
```

> **💡 代码解释**
> - `python-dotenv`：用于加载环境变量（存放 API 密钥）
> - 其他包（如 `langchain`、`openai`）通常已经安装
>
> **⚠️ 新手注意**
> - 去掉 `!` 前面的 `#` 注释以在 Jupyter 中运行
> - 或在终端运行：`pip install python-dotenv langchain langchain-openai faiss-cpu`

### 克隆仓库获取辅助函数

```python
# 克隆仓库以访问辅助函数和评估模块
# !git clone https://github.com/NirDiamant/RAG_TECHNIQUES.git
import sys
sys.path.append('RAG_TECHNIQUES')
# 如果需要运行最新数据
# !cp -r RAG_TECHNIQUES/data .
```

> **💡 代码解释**
> - 从 GitHub 克隆项目仓库
> - 把仓库路径添加到 Python 搜索路径
> - 这样可以导入 `helper_functions` 和 `evaluation` 模块
>
> **⚠️ 新手注意**
> - 如果你已经有这些文件，可以跳过这一步
> - 或者手动复制需要的文件到你的工作目录

### 导入库并配置环境变量

```python
import os
import sys
from dotenv import load_dotenv

# 原始路径追加已替换为 Colab 兼容版本
from helper_functions import *
from evaluation.evalute_rag import *

# 从 .env 文件加载环境变量
load_dotenv()

# 设置 OpenAI API 密钥环境变量
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
```

> **💡 代码解释**
> - `helper_functions`：包含一些辅助函数（如 `encode_pdf`、`show_context`）
> - `evaluation.evalute_rag`：包含 RAG 评估函数
> - `load_dotenv()`：从 `.env` 文件读取环境变量
> - `os.environ["OPENAI_API_KEY"]`：设置 OpenAI API 密钥
>
> **⚠️ 新手注意**
> - 需要先创建 `.env` 文件，内容：`OPENAI_API_KEY=你的密钥`
> - API 密钥可以从 [OpenAI 官网](https://platform.openai.com/api-keys) 获取

---

## 📁 第二步：准备文档

### 📖 这是什么？

下载并准备要检索的 PDF 文档。

### 💻 完整代码

```python
# 下载所需的数据文件
import os
os.makedirs('data', exist_ok=True)

# 下载此笔记本中使用的 PDF 文档
# !wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
```

> **💡 代码解释**
> - `os.makedirs('data', exist_ok=True)`：创建 `data` 目录
> - `wget`：下载 PDF 文件
> - 这里用的是气候变化相关的 PDF 文档
>
> **⚠️ 新手注意**
> - 如果 `wget` 不可用，可以用 `curl` 或手动下载
> - 也可以用自己的 PDF 文档，修改路径即可

### 定义文档路径

```python
path = "data/Understanding_Climate_Change.pdf"
```

---

## 🔧 第三步：创建 HyDE 检索器类

### 📖 这是什么？

这是核心部分！我们将创建一个 `HyDERetriever` 类，封装 HyDE 的全部功能。

### 💻 完整代码

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from helper_functions import encode_pdf  # 假设这个函数存在

class HyDERetriever:
    def __init__(self, files_path, chunk_size=500, chunk_overlap=100):
        """
        初始化 HyDE 检索器

        参数：
        files_path: PDF 文件路径
        chunk_size: 分块大小（字符数）
        chunk_overlap: 块之间重叠大小（字符数）
        """
        # 初始化 LLM（用于生成假设性文档）
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=4000)

        # 初始化 embedding 模型（用于向量化）
        self.embeddings = OpenAIEmbeddings()

        # 保存分块参数
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 编码 PDF 并创建向量存储
        # encode_pdf 函数会：
        # 1. 读取 PDF
        # 2. 分割成块
        # 3. 将每个块转成向量
        # 4. 存入 FAISS 向量数据库
        self.vectorstore = encode_pdf(files_path, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

        # 创建 HyDE 提示模板
        self.hyde_prompt = PromptTemplate(
            input_variables=["query", "chunk_size"],
            template="""给定问题'{query}'，生成直接回答此问题的假设性文档。文档应该详细且深入。
            文档大小必须恰好是{chunk_size}个字符。""",
        )

        # 创建 HyDE 链：提示词 → LLM
        self.hyde_chain = self.hyde_prompt | self.llm

    def generate_hypothetical_document(self, query):
        """
        根据查询生成假设性文档

        参数：
        query (str): 用户查询

        返回：
        str: 生成的假设性文档内容
        """
        # 准备输入变量
        input_variables = {"query": query, "chunk_size": self.chunk_size}
        # 调用 HyDE 链生成文档
        return self.hyde_chain.invoke(input_variables).content

    def retrieve(self, query, k=3):
        """
        使用 HyDE 进行检索

        参数：
        query (str): 用户查询
        k (int): 返回最相关的 k 个文档

        返回：
        tuple: (相似文档列表，假设性文档内容)
        """
        # 步骤 1：生成假设性文档
        hypothetical_doc = self.generate_hypothetical_document(query)

        # 步骤 2：用假设性文档在向量存储中搜索相似文档
        similar_docs = self.vectorstore.similarity_search(hypothetical_doc, k=k)

        # 步骤 3：返回结果
        return similar_docs, hypothetical_doc
```

> **💡 代码解释**
>
> **`__init__` 初始化方法**：
> - `ChatOpenAI`：用于生成假设性文档的 LLM
> - `OpenAIEmbeddings`：用于将文本转为向量的模型
> - `encode_pdf`：辅助函数，处理 PDF 并创建向量存储
> - `hyde_prompt`：告诉 LLM 如何生成假设性文档
> - `hyde_chain`：把提示词和 LLM 连接成链
>
> **`generate_hypothetical_document` 方法**：
> - 输入用户查询
> - 调用 LLM 生成假设性文档
> - 返回生成的文档内容
>
> **`retrieve` 方法**：
> - 步骤 1：生成假设性文档
> - 步骤 2：用假设性文档搜索相似文档
> - 步骤 3：返回相似文档和假设性文档
>
> **⚠️ 新手注意**
> - `chunk_size` 很重要！假设性文档会按照这个大小生成
> - `k` 控制返回多少结果，太小可能漏掉信息，太多会有噪音
> - 如果 `encode_pdf` 函数不存在，需要自己实现（见下文）

### 创建 HyDE 检索器实例

```python
retriever = HyDERetriever(path)
```

> **💡 代码解释**
> - 用 PDF 文件路径初始化检索器
> - 这会自动完成 PDF 编码和向量存储创建
> - 可能需要几十秒时间

---

## 🔍 第四步：测试检索效果

### 📖 这是什么？

用一个示例查询来测试 HyDE 检索器。

### 💻 完整代码

```python
test_query = "气候变化的主要原因是什么？"
results, hypothetical_doc = retriever.retrieve(test_query)
```

> **💡 代码解释**
> - `test_query`：示例查询
> - `retrieve`：调用检索方法
> - `results`：检索到的相似文档列表
> - `hypothetical_doc`：生成的假设性文档

### 查看结果

```python
# 提取检索到的文档内容
docs_content = [doc.page_content for doc in results]

print("假设性文档:\n")
print(text_wrap(hypothetical_doc)+"\n")

# 显示检索到的上下文
show_context(docs_content)
```

> **💡 预期输出示例**
> ```
> 假设性文档:
>
> 气候变化的主要原因包括温室气体排放、森林砍伐和工业活动。
> 二氧化碳是最主要的温室气体，来源于化石燃料的燃烧...
> （约 500 字的详细回答）
>
> ---
>
> 检索到的相关文档:
>
> 1. 化石燃料燃烧产生的二氧化碳排放是气候变化的主要驱动因素...
> 2. 森林砍伐导致碳汇减少，加剧了温室效应...
> 3. 工业活动释放了大量温室气体和空气污染物...
> ```
>
> **⚠️ 新手注意**
> - `text_wrap` 和 `show_context` 是辅助函数，用于美观地显示文本
> - 如果没有这些函数，可以直接 `print`

---

## 📊 第五步：理解 HyDE 的工作流程

### 📖 完整流程图

```
┌──────────────────────────────────────────────────────────────────────┐
│                         HyDE 完整流程                                │
└──────────────────────────────────────────────────────────────────────┘

    用户查询："气候变化的主要原因是什么？"
         │
         ▼
    ┌─────────────────┐
    │  HyDE Prompt    │ "根据这个问题生成一篇 500 字的假设性文档"
    │  + LLM (GPT-4)  │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────────────────────────────────────────────┐
    │              假设性文档（约 500 字）                       │
    │  "气候变化的主要原因包括温室气体排放、森林砍伐和...       │
    │   二氧化碳排放主要来自化石燃料燃烧，包括煤炭、石油...     │
    │   甲烷是另一种强效温室气体，来源包括畜牧业和稻田...       │
    │   ..."                                                   │
    └────────────────────┬────────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────────┐
    │              FAISS 向量存储检索                           │
    │  用假设性文档的向量去搜索相似的真实文档                  │
    └────────────────────┬────────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────────┐
    │              检索到的相关文档（k=3）                      │
    │  1. "化石燃料燃烧产生的二氧化碳排放是气候变化的..."       │
    │  2. "森林砍伐导致碳汇减少，加剧了温室效应..."             │
    │  3. "工业活动释放了大量温室气体和空气污染物..."           │
    └─────────────────────────────────────────────────────────┘
```

### 为什么这样更好？

| 比较项 | 传统检索 | HyDE 检索 |
|--------|---------|----------|
| 输入长度 | 6-20 字的查询 | 300-500 字的假设性文档 |
| 词汇丰富度 | 有限的词汇 | 包含各种同义词和相关概念 |
| 语义匹配 | 可能不准确 | 假设性文档和真实文档风格一致 |
| 检索效果 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 🛠️ 附录：encode_pdf 函数实现

### 📖 这是什么？

如果 `helper_functions` 中没有 `encode_pdf` 函数，可以自己实现。

### 💻 完整代码

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def encode_pdf(path, chunk_size=500, chunk_overlap=100):
    """
    将 PDF 文件编码到 FAISS 向量存储中

    参数：
    path: PDF 文件路径
    chunk_size: 分块大小
    chunk_overlap: 块之间重叠

    返回：
    FAISS 向量存储对象
    """
    # 步骤 1：加载 PDF
    loader = PyPDFLoader(path)
    documents = loader.load()

    # 步骤 2：分割成块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)

    # 步骤 3：创建向量存储
    # OpenAIEmbeddings 会自动调用 OpenAI API 将每个块转为向量
    # FAISS 将这些向量组织成高效的索引结构
    vectorstore = FAISS.from_documents(texts, OpenAIEmbeddings())

    return vectorstore
```

> **💡 代码解释**
> - `PyPDFLoader`：读取 PDF 文件
> - `RecursiveCharacterTextSplitter`：递归字符文本分割器
> - `FAISS.from_documents`：直接从文档创建向量存储
>
> **⚠️ 新手注意**
> - 这个函数会调用 OpenAI API，确保 API 密钥已设置
> - 大 PDF 文件可能需要较长时间处理

---

## ❓ 常见问题 FAQ

### Q1：HyDE 和查询重写有什么区别？

**A**：
- **查询重写**：把查询改写成另一个查询（仍然是短文本）
- **HyDE**：把查询扩展成一篇完整的"假设性文档"（长文本）

```
查询重写："气候变化的原因" → "导致气候变化的主要因素有哪些"
HyDE："气候变化的原因" → 生成一篇 500 字的详细回答
```

### Q2：chunk_size 应该设多少？

**A**：
- 太小（<200）：假设性文档太短，信息不足
- 太大（>1000）：可能包含太多噪音，影响检索精度
- **推荐范围**：400-600 字符

### Q3：可以用本地模型吗（不用 OpenAI）？

**A**：可以！修改代码使用 Ollama 或其他本地模型：

```python
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

# LLM 用 Ollama
self.llm = Ollama(model="llama3.1:70b")

# Embedding 也用 Ollama
self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
```

### Q4：HyDE 会增加多少延迟？

**A**：
- 生成假设性文档：1-3 秒（取决于 LLM）
- 向量检索：<0.1 秒（FAISS 很快）
- **总延迟**：约 1-5 秒

### Q5：HyDE 适合所有场景吗？

**A**：不是。HyDE 特别适合：
- ✅ 短查询、模糊查询
- ✅ 需要语义理解的复杂查询
- ❌ 非常简单的事实性查询（如"谁是美国总统"）
- ❌ 对延迟要求极高的场景

### Q6：如果生成的假设性文档不准确怎么办？

**A**：
- 调整提示词，让 LLM 更保守
- 降低 `temperature` 参数（如设为 0）
- 结合传统检索方法，做结果融合

---

## 🎯 进阶技巧

### 技巧 1：自定义 HyDE 提示词

```python
# 针对特定领域的提示词
custom_hyde_prompt = PromptTemplate(
    input_variables=["query", "chunk_size"],
    template="""你是一位气候科学专家。给定问题'{query}'，
    撰写一篇科学准确的假设性回答文档。
    使用专业术语，包含具体数据和引用。
    文档大小：{chunk_size}个字符。""",
)
```

### 技巧 2：混合检索（HyDE + 传统）

```python
def hybrid_retrieve(self, query, k=3, hyde_weight=0.7):
    """
    混合检索：结合 HyDE 和传统检索

    参数：
    query: 用户查询
    k: 返回文档数
    hyde_weight: HyDE 结果权重（0-1）
    """
    # HyDE 检索
    hyde_docs, _ = self.retrieve(query, k=int(k * hyde_weight))

    # 传统检索（直接用查询）
    traditional_docs = self.vectorstore.similarity_search(query, k=int(k * (1-hyde_weight)))

    # 合并结果
    return hyde_docs + traditional_docs
```

### 技巧 3：多假设性文档

```python
def multi_hyde_retrieve(self, query, n_documents=3, k=3):
    """
    生成多个假设性文档，然后合并检索结果

    参数：
    query: 用户查询
    n_documents: 生成的假设性文档数量
    k: 每个假设性文档检索的文档数
    """
    all_docs = []

    for i in range(n_documents):
        # 每次用略有不同的提示词
        varied_prompt = f"从不同角度回答：{query}"
        hypo_doc = self.generate_hypothetical_document(varied_prompt)
        docs = self.vectorstore.similarity_search(hypo_doc, k=k)
        all_docs.extend(docs)

    # 去重
    unique_docs = []
    seen = set()
    for doc in all_docs:
        if doc.page_content not in seen:
            unique_docs.append(doc)
            seen.add(doc.page_content)

    return unique_docs[:k]  # 返回前 k 个
```

---

## 📈 可视化：HyDE vs 传统检索

```
传统检索流程：
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ 用户查询    │ ──→ │ 向量检索     │ ──→ │ 相关文档    │
│ (6-20 字)    │     │ (直接匹配)   │     │ (可能不准)  │
└─────────────┘     └──────────────┘     └─────────────┘

HyDE 检索流程：
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────────┐
│ 用户查询    │ ──→ │ 生成假设性   │ ──→ │ 向量检索     │ ──→ │ 相关文档    │
│ (6-20 字)    │     │ 文档 (500 字)  │     │ (语义匹配)   │     │ (更精准)    │
└─────────────┘     └──────────────┘     └─────────────┘     └─────────────┘
                           ↑
                      LLM (GPT-4)
```

---

## 🎉 恭喜你学完了！

现在你已经掌握了：
1. ✅ HyDE 的核心概念和工作原理
2. ✅ 完整的代码实现
3. ✅ 如何创建和使用 HyDE 检索器
4. ✅ 进阶技巧和常见问题解决方法

**下一步建议**：
- 用自己的文档测试 HyDE
- 调整 `chunk_size` 和 `k` 参数看效果
- 尝试结合其他 RAG 技术（如查询重写）

---

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--hyde-hypothetical-document-embedding)
