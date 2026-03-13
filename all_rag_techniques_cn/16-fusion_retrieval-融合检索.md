# 文档搜索中的融合检索

## 概述

本代码实现了一个融合检索（Fusion Retrieval）系统，该系统将基于向量的相似性搜索与基于关键词的 BM25 检索相结合。该方法旨在利用两种方法的优势，提高文档检索的整体质量和相关性。

## 动机

传统的检索方法通常依赖于语义理解（基于向量）或关键词匹配（BM25）。每种方法都有其优缺点。融合检索旨在结合这些方法，创建一个更健壮、更准确的检索系统，能够有效处理更广泛的查询类型。

## 核心组件

1. PDF 文档处理和文本分块
2. 使用 FAISS 和 OpenAI Embedding 创建向量存储
3. 创建 BM25 索引用于基于关键词的检索
4. 自定义融合检索函数，结合两种方法

## 方法细节

### 文档预处理

1. 使用 RecursiveCharacterTextSplitter 加载 PDF 并将其分割成块。
2. 通过将 't' 替换为空格来清理文本块（可能是为了解决特定的格式问题）。

### 向量存储创建

1. 使用 OpenAI Embedding 为文本块创建向量表示。
2. 从这些 Embedding 创建 FAISS 向量存储，用于高效的相似性搜索。

### BM25 索引创建

1. 从用于向量存储的相同文本块创建 BM25 索引。
2. 这允许在基于向量的方法之外进行基于关键词的检索。

### 融合检索函数

`fusion_retrieval` 函数是此实现的核心：

1. 它接受一个查询，同时执行基于向量和基于 BM25 的检索。
2. 将两种方法的得分归一化到同一尺度。
3. 计算这些得分的加权组合（由 `alpha` 参数控制）。
4. 根据组合得分对文档进行排序，返回前 k 个结果。

## 该方法的优势

1. **提高检索质量**：通过结合语义搜索和关键词搜索，系统可以同时捕获概念相似性和精确的关键词匹配。
2. **灵活性**：`alpha` 参数允许根据具体用例或查询类型调整向量搜索和关键词搜索之间的平衡。
3. **健壮性**：组合方法可以有效处理更广泛的查询，弥补单一方法的不足。
4. **可定制性**：系统可以轻松适配使用不同的向量存储或基于关键词的检索方法。

## 结论

融合检索代表了一种强大的文档搜索方法，它结合了语义理解和关键词匹配的优势。通过同时利用基于向量和 BM25 的检索方法，它为信息检索任务提供了一个更全面、更灵活的解决方案。这种方法在概念相似性和关键词相关性都很重要的各个领域都有潜在应用，例如学术研究、法律文档搜索或通用搜索引擎。

<div style="text-align: center;">

<img src="../images/fusion_retrieval.svg" alt="Fusion Retrieval" style="width:100%; height:auto;">
</div>

# 包安装和导入

下面的单元格安装了运行此 notebook 所需的所有包。

```python
# 安装所需的包
!pip install langchain numpy python-dotenv rank-bm25
```

```python
# 克隆仓库以访问辅助函数和评估模块
!git clone https://github.com/NirDiamant/RAG_TECHNIQUES.git
import sys
sys.path.append('RAG_TECHNIQUES')
# 如果需要使用最新数据运行
# !cp -r RAG_TECHNIQUES/data .
```

```python
import os
import sys
from dotenv import load_dotenv
from langchain.docstore.document import Document

from typing import List
from rank_bm25 import BM25Okapi
import numpy as np


# 原始路径追加已替换为 Colab 兼容性
from helper_functions import *
from evaluation.evalute_rag import *

# 从 .env 文件加载环境变量
load_dotenv()

# 设置 OpenAI API 密钥环境变量
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
```

### 定义文档路径

```python
# 下载所需的数据文件
import os
os.makedirs('data', exist_ok=True)

# 下载本笔记本使用的 PDF 文档
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
```

```python
path = "data/Understanding_Climate_Change.pdf"
```

### 将 PDF 编码到向量存储并返回上一步的分块文档以创建 BM25 实例

```python
def encode_pdf_and_get_split_documents(path, chunk_size=1000, chunk_overlap=200):
    """
    使用 OpenAI embeddings 将 PDF 书籍编码为向量存储。

    Args:
        path: PDF 文件的路径。
        chunk_size: 每个文本块的期望大小。
        chunk_overlap: 连续块之间的重叠量。

    Returns:
        包含编码后书籍内容的 FAISS 向量存储。
    """

    # 加载 PDF 文档
    loader = PyPDFLoader(path)
    documents = loader.load()

    # 将文档分割为块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    # 创建 embeddings 和向量存储
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore, cleaned_texts
```

### 创建向量存储并获取分块文档

```python
vectorstore, cleaned_texts = encode_pdf_and_get_split_documents(path)
```

### 创建 BM25 索引用于通过关键词检索文档

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
    return BM25Okapi(tokenized_docs)
```

```python
bm25 = create_bm25_index(cleaned_texts) # 从清理后的文本（块）创建 BM25 索引
```

### 定义一个函数，同时进行语义检索和关键词检索，归一化得分并获取前 k 个文档

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

    Returns:
    List[Document]: 基于组合得分的前 k 个文档。
    """

    epsilon = 1e-8

    # 步骤 1：从向量存储获取所有文档
    all_docs = vectorstore.similarity_search("", k=vectorstore.index.ntotal)

    # 步骤 2：执行 BM25 搜索
    bm25_scores = bm25.get_scores(query.split())

    # 步骤 3：执行向量搜索
    vector_results = vectorstore.similarity_search_with_score(query, k=len(all_docs))

    # 步骤 4：归一化得分
    vector_scores = np.array([score for _, score in vector_results])
    vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores) + epsilon)

    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) -  np.min(bm25_scores) + epsilon)

    # 步骤 5：组合得分
    combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores

    # 步骤 6：对文档排序
    sorted_indices = np.argsort(combined_scores)[::-1]

    # 步骤 7：返回前 k 个文档
    return [all_docs[i] for i in sorted_indices[:k]]
```

### 使用示例

```python
# 查询
query = "What are the impacts of climate change on the environment?"

# 执行融合检索
top_docs = fusion_retrieval(vectorstore, bm25, query, k=5, alpha=0.5)
docs_content = [doc.page_content for doc in top_docs]
show_context(docs_content)
```

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--fusion-retrieval)
