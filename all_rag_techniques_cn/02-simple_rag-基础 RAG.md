# 简单 RAG（检索增强生成）系统

## 概述

此代码实现了一个基本的检索增强生成 (RAG) 系统，用于处理和查询 PDF 文档。系统将文档内容编码到向量存储中，然后可以查询以检索相关信息。

## 关键组件

1. PDF 处理和文本提取
2. 文本分块以便于处理
3. 向量存储创建使用 [FAISS](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) 和 OpenAI embeddings
4. 检索器设置用于查询处理后的文档
5. RAG 系统评估

## 方法详情

### 文档预处理

1. 使用 PyPDFLoader 加载 PDF。
2. 使用 RecursiveCharacterTextSplitter 将文本分割成块，具有指定的块大小和重叠。

### 文本清理

应用自定义函数 `replace_t_with_space` 来清理文本块。这可能解决了 PDF 中的特定格式问题。

### 向量存储创建

1. 使用 OpenAI embeddings 创建文本块的向量表示。
2. 从这些 embeddings 创建 FAISS 向量存储以进行高效的相似性搜索。

### 检索器设置

1. 配置检索器以获取给定查询的前 2 个最相关块。

### 编码函数

`encode_pdf` 函数封装了加载、分块、清理和将 PDF 编码到向量存储的整个过程。

## 关键特性

1. 模块化设计：编码过程封装在单个函数中，便于重用。
2. 可配置分块：允许调整块大小和重叠。
3. 高效检索：使用 FAISS 进行快速相似性搜索。
4. 评估：包含评估 RAG 系统性能的函数。

## 使用示例

代码包含一个测试查询："气候变化的主要原因是什么？"。这演示了如何使用检索器从处理后的文档中获取相关上下文。

## 评估

系统包含一个 `evaluate_rag` 函数来评估检索器的性能，尽管提供的代码中没有详细说明使用的具体指标。

## 此方法的优势

1. 可扩展性：可以通过分块处理大型文档。
2. 灵活性：易于调整块大小和检索结果数量等参数。
3. 效率：利用 FAISS 在高维空间中进行快速相似性搜索。
4. 与高级 NLP 集成：使用 OpenAI embeddings 进行最先进的文本表示。

## 结论

这个简单的 RAG 系统为构建更复杂的信息检索和问答系统提供了坚实的基础。通过将文档内容编码到可搜索的向量存储中，它能够高效地检索相关信息以响应查询。这种方法对于需要在大型文档或文档集合中快速访问特定信息的应用程序特别有用。

# 包安装和导入

下面的单元格安装运行此笔记本所需的所有必要包。


```python
# 安装所需的包
!pip install pypdf==5.6.0
!pip install PyMuPDF==1.26.1
!pip install python-dotenv==1.1.0
!pip install langchain-community==0.3.25
!pip install langchain_openai==0.3.23
!pip install rank_bm25==0.2.2
!pip install faiss-cpu==1.11.0
!pip install deepeval==3.1.0
```

```python
# 克隆仓库以访问辅助函数和评估模块
!git clone https://github.com/NirDiamant/RAG_TECHNIQUES.git
import sys
sys.path.append('RAG_TECHNIQUES')

# 如果需要运行最新数据
# !cp -r RAG_TECHNIQUES/data .
```

```python
import os
import sys
from dotenv import load_dotenv
from google.colab import userdata



# 从 .env 文件加载环境变量
load_dotenv()

# 设置 OpenAI API 密钥环境变量（如果不使用 OpenAI 请注释掉）
if not userdata.get('OPENAI_API_KEY'):
    os.environ["OPENAI_API_KEY"] = input("请输入您的 OpenAI API 密钥：")
else:
    os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')

# 原始路径追加已替换为 Colab 兼容版本

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from helper_functions import (EmbeddingProvider,
                              retrieve_context_per_question,
                              replace_t_with_space,
                              get_langchain_embedding_provider,
                              show_context)

from evaluation.evalute_rag import evaluate_rag

from langchain.vectorstores import FAISS

```

### 读取文档

```python
# 下载所需的数据文件
import os
os.makedirs('data', exist_ok=True)

# 下载此笔记本中使用的 PDF 文档
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf

```

```python
path = "data/Understanding_Climate_Change.pdf"
```

### 编码文档

```python
def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """
    使用 OpenAI embeddings 将 PDF 书籍编码到向量存储中。

    参数：
        path: PDF 文件的路径。
        chunk_size: 每个文本块的期望大小。
        chunk_overlap: 连续块之间的重叠量。

    返回：
        包含编码后书籍内容的 FAISS 向量存储。
    """

    # 加载 PDF 文档
    loader = PyPDFLoader(path)
    documents = loader.load()

    # 将文档分割成块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    # 创建 embeddings（已测试 OpenAI 和 Amazon Bedrock）
    embeddings = get_langchain_embedding_provider(EmbeddingProvider.OPENAI)
    #embeddings = get_langchain_embedding_provider(EmbeddingProvider.AMAZON_BEDROCK)

    # 创建向量存储
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore
```

```python
chunks_vector_store = encode_pdf(path, chunk_size=1000, chunk_overlap=200)
```

### 创建检索器

```python
chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 2})
```

### 测试检索器

```python
test_query = "气候变化的主要原因是什么？"
context = retrieve_context_per_question(test_query, chunks_query_retriever)
show_context(context)
```

### 评估结果

```python
# 注意 - 这目前仅适用于 OPENAI
evaluate_rag(chunks_query_retriever)
```

```python

```

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--simple-rag)
