# 🌟 新手入门：基础 RAG（检索增强生成）系统

> **💡 给新手的说明**
> - **难度等级**：⭐⭐☆☆☆（入门级）
> - **预计时间**：30-45 分钟
> - **前置知识**：基础 Python 编程知识
> - **学习目标**：理解 RAG 系统的基本原理，能够构建一个简单的 PDF 文档问答系统

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
| **文档加载** | 读取 PDF 文件 | 把书放进图书馆 |
| **文本分块** | 把长文档切成小块 | 把书按章节分开存放 |
| **Embedding** | 将文字转成数字向量 | 给每本书贴上分类标签 |
| **向量存储** | 存储和管理这些向量 | 图书馆的书架系统 |
| **检索器** | 查找最相关的文档块 | 图书管理员查找书籍 |
| **FAISS** | 高效搜索相似内容 | 快速找书的索引系统 |

---

## 🛠️ 第一步：环境准备

### 📖 这是什么？

在开始之前，我们需要安装必要的 Python 库。就像做菜前要准备好厨具和食材一样。

### 💻 完整代码

```python
# ============================================
# 安装所需的包
# ============================================
# 下面的命令会安装运行此笔记本所需的所有包
# 每个包的作用：
# - pypdf: 读取 PDF 文件
# - PyMuPDF: 另一种 PDF 处理工具
# - python-dotenv: 管理 API 密钥
# - langchain-community: RAG 框架的核心组件
# - langchain_openai: OpenAI 的集成
# - rank_bm25: 文本检索算法
# - faiss-cpu: Facebook 的高效相似度搜索库
# - deepeval: 评估 RAG 系统性能

!pip install pypdf==5.6.0
!pip install PyMuPDF==1.26.1
!pip install python-dotenv==1.1.0
!pip install langchain-community==0.3.25
!pip install langchain_openai==0.3.23
!pip install rank_bm25==0.2.2
!pip install faiss-cpu==1.11.0
!pip install deepeval==3.1.0
```

> **💡 代码解释**
> - `!pip install` 是 Jupyter Notebook 中安装包的方式
> - `==` 后面的数字是版本号，确保使用特定版本可以避免兼容性问题
>
> **⚠️ 新手注意**
> - 如果遇到安装失败，可以尝试去掉版本号（如 `!pip install pypdf`）
> - 某些包可能需要较长时间安装，请耐心等待
> - 如使用国内网络，可添加清华源：`!pip install pypdf -i https://pypi.tuna.tsinghua.edu.cn/simple`

### 📖 克隆代码仓库（可选）

```python
# 克隆仓库以访问辅助函数和评估模块
!git clone https://github.com/NirDiamant/RAG_TECHNIQUES.git
import sys
sys.path.append('RAG_TECHNIQUES')

# 如果需要运行最新数据
# !cp -r RAG_TECHNIQUES/data .
```

> **💡 代码解释**
> - `git clone` 下载整个项目代码
> - `sys.path.append()` 让 Python 能找到项目中的辅助函数
>
> **⚠️ 新手注意**
> - 如果已经下载了项目，这一步可以跳过
> - 注释掉的 `cp` 命令是用来复制数据文件的

---

## 🔑 第二步：配置 API 密钥

### 📖 这是什么？

RAG 系统需要使用 OpenAI 的 API 来生成文本的向量表示（Embedding）。这一步就是设置你的 API 密钥。

### 💻 完整代码

```python
# ============================================
# 导入必要的库
# ============================================
import os
import sys
from dotenv import load_dotenv
from google.colab import userdata

# 从 .env 文件加载环境变量
load_dotenv()

# 设置 OpenAI API 密钥环境变量
# 如果不在 Google Colab 环境运行，可以简化为下面的方式
if not userdata.get('OPENAI_API_KEY'):
    os.environ["OPENAI_API_KEY"] = input("请输入您的 OpenAI API 密钥：")
else:
    os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')

# 导入 LangChain 的相关组件
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from helper_functions import (
    EmbeddingProvider,
    retrieve_context_per_question,
    replace_t_with_space,
    get_langchain_embedding_provider,
    show_context
)
from evaluation.evalute_rag import evaluate_rag
from langchain.vectorstores import FAISS
```

> **💡 代码解释**
> - `load_dotenv()` 从 `.env` 文件加载配置
> - `os.environ` 设置环境变量，让其他库可以访问
> - `userdata.get()` 是 Google Colab 的秘密管理方式
>
> **⚠️ 新手注意**
> - **API 密钥安全**：永远不要把你的 API 密钥直接写在代码里！
> - 更安全的做法是使用 `.env` 文件：
>   ```
>   # .env 文件内容
>   OPENAI_API_KEY=sk-your-actual-key-here
>   ```
> - 如果你不使用 OpenAI，可以改用其他 Embedding 服务（代码中有注释说明）
>
> **❓ 常见问题**
> - **Q: 我没有 OpenAI API 密钥怎么办？**
> - A: 你可以注册 OpenAI 账号获取，或者使用其他免费的 Embedding 模型
> - **Q: 出现 "ModuleNotFoundError" 怎么办？**
> - A: 确保上面的包都已经安装成功

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
os.makedirs('data', exist_ok=True)

# 下载此笔记本中使用的 PDF 文档
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf

# 指定 PDF 文件路径
path = "data/Understanding_Climate_Change.pdf"
```

> **💡 代码解释**
> - `os.makedirs('data', exist_ok=True)` 创建 data 目录，如果已存在也不会报错
> - `!wget` 从网络下载文件
> - `-O` 指定下载后的文件名
>
> **⚠️ 新手注意**
> - 如果下载失败，可以手动下载 PDF 放到 `data` 目录下
> - 你也可以使用自己的 PDF 文件，只需修改 `path` 变量
>
> **📊 术语解释**
> - **PDF**：Portable Document Format，便携式文档格式，一种常用的文档格式

---

## 🔧 第四步：编码文档（核心步骤）

### 📖 这是什么？

这是 RAG 系统最核心的部分！我们需要：
1. 读取 PDF 文件
2. 把长文本切成小块
3. 把文字转换成向量（数字表示）
4. 存储到向量数据库中

### 💻 完整代码

```python
def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """
    使用 OpenAI embeddings 将 PDF 书籍编码到向量存储中。

    参数：
        path: PDF 文件的路径。
        chunk_size: 每个文本块的期望大小（字符数）。
        chunk_overlap: 连续块之间的重叠量（字符数）。

    返回：
        包含编码后书籍内容的 FAISS 向量存储。
    """

    # ========== 步骤 1：加载 PDF 文档 ==========
    loader = PyPDFLoader(path)
    documents = loader.load()
    # 现在 documents 包含了 PDF 的所有内容

    # ========== 步骤 2：将文本分割成块 ==========
    # 为什么要分块？因为：
    # 1. 太长的文本超过 AI 的处理限制
    # 2. 小块更容易找到精确的相关信息
    # 3. 检索效率更高
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,      # 每块最多 1000 个字符
        chunk_overlap=chunk_overlap, # 相邻块重叠 200 个字符
        length_function=len
    )
    texts = text_splitter.split_documents(documents)

    # 清理文本（处理 PDF 中的特殊格式问题）
    cleaned_texts = replace_t_with_space(texts)

    # ========== 步骤 3：创建 Embeddings ==========
    # Embedding 是什么？
    # 想象一下图书馆的分类系统，它把相似主题的书放在一起
    # Embedding 就是把文字转成数字向量，相似的内容向量距离更近
    embeddings = get_langchain_embedding_provider(EmbeddingProvider.OPENAI)
    # 也可以使用 Amazon Bedrock:
    # embeddings = get_langchain_embedding_provider(EmbeddingProvider.AMAZON_BEDROCK)

    # ========== 步骤 4：创建向量存储 ==========
    # FAISS 是什么？
    # Facebook AI 开发的快速相似度搜索库
    # 就像图书馆的快速检索系统，可以秒级找到相似内容
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore
```

> **💡 代码解释**
>
> **关于 chunk_size（块大小）**：
> - `chunk_size=1000` 表示每块最多 1000 个字符
> - 太小：可能丢失上下文
> - 太大：检索不够精确
>
> **关于 chunk_overlap（重叠）**：
> - `chunk_overlap=200` 表示相邻块有 200 个字符重叠
> - 为什么需要重叠？避免关键信息被切分到两块中间
>
> **⚠️ 新手注意**
> - 这个函数会需要一些时间运行（取决于 PDF 大小）
> - 运行时会调用 OpenAI API，会产生少量费用
>
> **📊 术语解释**
> - **Embedding**：将文本转换为高维空间中的向量，语义相似的文本向量距离更近
> - **向量存储**：专门存储和检索向量数据的数据库

### 执行编码

```python
# 调用函数，将 PDF 编码到向量存储
chunks_vector_store = encode_pdf(path, chunk_size=1000, chunk_overlap=200)
```

> **💡 预期输出**
> - 如果没有报错，说明编码成功
> - 现在你的内容已经存储在 `chunks_vector_store` 中了

---

## 🔍 第五步：创建检索器

### 📖 这是什么？

检索器就像是一个"图书管理员"，当你提问时，它会帮你找到最相关的文档块。

### 💻 完整代码

```python
# 将向量存储转换为检索器
# search_kwargs={"k": 2} 表示每次检索返回 2 个最相关的结果
chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 2})
```

> **💡 代码解释**
> - `as_retriever()` 把向量存储转换成检索器接口
> - `k=2` 可以调整，比如 `k=5` 会返回 5 个相关结果
>
> **⚠️ 新手注意**
> - `k` 值太小可能遗漏信息，太大可能引入噪音
> - 一般建议从 2-5 开始尝试

---

## 🧪 第六步：测试检索器

### 📖 这是什么？

现在来测试一下我们的 RAG 系统是否正常工作！

### 💻 完整代码

```python
# 设置一个测试问题
test_query = "气候变化的主要原因是什么？"

# 检索相关上下文
context = retrieve_context_per_question(test_query, chunks_query_retriever)

# 显示检索到的内容
show_context(context)
```

> **💡 代码解释**
> - `retrieve_context_per_question()` 从辅助函数导入，用于检索
> - `show_context()` 格式化显示检索结果
>
> **⚠️ 新手注意**
> - 如果遇到问题，确保 helper_functions 模块已正确导入
> - 检索结果可能因 PDF 内容而异
>
> **📊 预期输出示例**
> ```
> 检索到的上下文 1:
> [相关的气候变化原因描述...]
>
> 检索到的上下文 2:
> [更多相关信息...]
> ```

---

## 📊 第七步：评估结果（可选）

### 📖 这是什么？

评估功能可以帮你了解 RAG 系统的检索质量如何。

### 💻 完整代码

```python
# 注意：这目前仅适用于 OPENAI
evaluate_rag(chunks_query_retriever)
```

> **💡 代码解释**
> - `evaluate_rag()` 会运行一系列测试来评估检索质量
> - 评估指标通常包括准确率、召回率等
>
> **⚠️ 新手注意**
> - 这个功能可能需要额外的设置
> - 初学者可以先跳过这一步，等理解基础后再尝试

---

## 🎯 完整代码总结

下面是一个可以独立运行的简化版本：

```python
# 1. 导入必要的库
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os

# 2. 设置 API 密钥
os.environ["OPENAI_API_KEY"] = "你的 API 密钥"

# 3. 定义编码函数
def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    loader = PyPDFLoader(path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)

    return vectorstore

# 4. 执行编码
vectorstore = encode_pdf("data/your_document.pdf")

# 5. 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 6. 检索测试
results = retriever.invoke("你的问题")
for doc in results:
    print(doc.page_content)
```

---

## ❓ 常见问题 FAQ

### Q1: 我可以用中文 PDF 吗？
**A**: 可以！这个系统支持多语言，包括中文。不过 OpenAI 的 Embedding 对英文优化更好，中文效果也不错。

### Q2: 为什么检索结果不准确？
**A**: 可能的原因：
- PDF 内容本身不包含答案
- `chunk_size` 设置不合适
- `k` 值太小，遗漏了相关信息
- 问题表述不够清晰

### Q3: 如何处理大文件？
**A**:
- 增加 `chunk_size` 减少块数量
- 使用更强大的向量数据库（如 FAISS GPU 版本）
- 考虑分布式处理

### Q4: 费用大概多少？
**A**:
- OpenAI Embedding 费用很低，处理一本普通书籍通常只需几美分
- 可以在 OpenAI 官网查看最新定价

### Q5: 可以不用 OpenAI 吗？
**A**:
- 可以！代码中已经包含了使用 Amazon Bedrock 的选项
- 还有其他免费模型如 HuggingFace 的模型

---

## 🚀 下一步学习建议

恭喜你完成了基础 RAG 系统！接下来你可以：

1. **尝试不同文档**：用你自己的 PDF 文件测试
2. **调整参数**：尝试不同的 `chunk_size` 和 `k` 值
3. **学习进阶技巧**：继续学习本系列的其他教程
4. **构建完整应用**：结合前端做一个问答网站

---

## 📚 关键知识点回顾

| 概念 | 说明 |
|------|------|
| **RAG** | 检索增强生成，先检索再生成的 AI 架构 |
| **Embedding** | 将文本转换为数字向量的技术 |
| **FAISS** | Facebook 开发的高效相似度搜索库 |
| **分块** | 将长文档切分成小块以便处理 |
| **重叠** | 相邻块之间的重复内容，避免信息丢失 |
| **检索器** | 负责查找最相关文档的组件 |

---

*本教程是 RAG 技术系列教程的基础篇，建议按顺序学习后续教程以掌握更高级的技巧。*

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--simple-rag)
