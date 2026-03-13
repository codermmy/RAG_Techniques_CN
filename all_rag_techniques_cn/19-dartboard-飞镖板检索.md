# Dartboard RAG：平衡相关性和多样性的检索增强生成

## 概述

**Dartboard RAG** 流程解决了大型知识库中的一个常见挑战：确保检索到的信息既相关又不冗余。通过显式优化组合的相关性 - 多样性评分函数，它防止多个 top-k 文档提供相同的信息。这种方法源自论文中的优雅方法：

> [*使用相关信息增益的更好 RAG*](https://arxiv.org/abs/2407.12101)

论文概述了核心思想的三种变体——混合 RAG（密集 + 稀疏）、交叉编码器版本和基础方法。**基础方法**最直接地传达了基本概念，本实现扩展了它，添加了可选权重来控制相关性和多样性之间的平衡。

## 动机

1. **密集、重叠的知识库**
   在大型数据库中，文档可能会重复相似的内容，导致 top-k 检索中的冗余。

2. **改进信息覆盖**
   结合相关性和多样性可以产生更丰富的文档集，减轻过度相似内容的"回音室"效应。

## 关键组件

1. **相关性和多样性组合**
   - 计算一个分数，既考虑文档与查询的相关性，也考虑它与已选文档的区别程度。

2. **加权平衡**
   - 引入 RELEVANCE_WEIGHT 和 DIVERSITY_WEIGHT 以允许动态控制评分。
   - 有助于避免过度多样化但相关性较低的结果。

3. **生产就绪代码**
   - 源自官方实现，但为清晰起见进行了重组。
   - 允许更轻松地集成到现有的 RAG 管道中。

## 方法详情

1. **文档检索**
   - 基于相似度（例如余弦或 BM25）获取初始候选文档集。
   - 通常检索 top-N 候选作为起点。

2. **评分和选择**
   - 每个文档的总体得分结合了**相关性**和**多样性**：
   - 选择得分最高的文档，然后惩罚与它过度相似的文档。
   - 重复直到识别出 top-k 文档。

3. **混合/融合和交叉编码器支持**
   本质上，您只需要文档与查询之间的距离以及文档之间的距离。您可以轻松地从混合/融合检索或交叉编码器检索中提取这些。我唯一的建议是少依赖基于 raking 的分数。
   - 对于**混合/融合检索**：将相似度（密集和稀疏/BM25）合并为单个距离。这可以通过组合密集和稀疏向量上的余弦相似度（例如平均它们）来实现。转换为距离很简单（1 - 平均余弦相似度）。
   - 对于**交叉编码器**：您可以直接使用交叉编码器相似度分数（1-相似度），可能会用缩放因子进行调整。

4. **平衡和调整**
   - 根据您的需求和数据集的密度调整 DIVERSITY_WEIGHT 和 RELEVANCE_WEIGHT。

通过将**相关性**和**多样性**都集成到检索中，Dartboard RAG 方法确保 top-k 文档共同提供更丰富、更全面的信息——从而在检索增强生成系统中产生更高质量的响应。

该论文也有官方代码实现，本代码基于它，但我认为这里的代码更具可读性、可管理性和生产就绪性。

# 包安装和导入

下面的单元格安装运行此笔记本所需的所有包。

```python
# 安装所需的包
!pip install numpy python-dotenv
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
from scipy.special import logsumexp
from typing import Tuple, List, Any
import numpy as np

# 从.env 文件加载环境变量
load_dotenv()
# 设置 OpenAI API 密钥环境变量（如果不使用 OpenAI 请注释）
if not os.getenv('OPENAI_API_KEY'):
    print("Please enter your OpenAI API key: ")
    os.environ["OPENAI_API_KEY"] = input("Please enter your OpenAI API key: ")
else:
    os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# 原始路径追加已替换为 Colab 兼容性
from helper_functions import *
from evaluation.evalute_rag import *

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
# 这部分与 simple_rag.ipynb 相同，只是模拟密集数据集
def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
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
    documents=documents*5 # 加载每个文档 5 次以模拟密集数据集

    # 将文档分成块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    # 创建嵌入（已在 OpenAI 和 Amazon Bedrock 上测试）
    embeddings = get_langchain_embedding_provider(EmbeddingProvider.OPENAI)
    #embeddings = get_langchain_embedding_provider(EmbeddingProvider.AMAZON_BEDROCK)

    # 创建向量存储
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore
```

### 创建向量存储

```python
chunks_vector_store = encode_pdf(path, chunk_size=1000, chunk_overlap=200)
```

### 一些用于使用向量存储进行检索的辅助函数。
这部分与 simple_rag.ipynb 相同，只是它使用实际的 FAISS 索引（而不是包装器）

```python

def idx_to_text(idx:int):
    """
    将向量存储索引转换为相应的文本。
    """
    docstore_id = chunks_vector_store.index_to_docstore_id[idx]
    document = chunks_vector_store.docstore.search(docstore_id)
    return document.page_content


def get_context(query:str,k:int=5) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    使用 top k 检索为查询检索前 k 个上下文项。
    """
    # 常规 top k 检索
    q_vec=chunks_vector_store.embedding_function.embed_documents([query])
    _,indices=chunks_vector_store.index.search(np.array(q_vec),k=k)

    texts = [idx_to_text(i) for i in indices[0]]
    return texts

```

```python

test_query = "What is the main cause of climate change?"

```

### 常规 top k 检索
- 此演示表明，当数据库是密集的（这里我们通过将每个文档加载 5 次来模拟密度），结果并不好，我们没有得到最相关的结果。注意 top 3 结果都是同一文档的重复。

```python
texts=get_context(test_query,k=3)
show_context(texts)
```

## 现在是真正的部分 :)

### 更多用于距离归一化的工具

```python
def lognorm(dist:np.ndarray, sigma:float):
    """
    计算给定距离和 sigma 的对数正态概率。
    """
    if sigma < 1e-9:
        return -np.inf * dist
    return -np.log(sigma) - 0.5 * np.log(2 * np.pi) - dist**2 / (2 * sigma**2)

```

## 贪心 Dartboard 搜索

这是核心算法：一种搜索算法，通过平衡两个因素从集合中选择多样化的相关文档集：与查询的相关性和所选文档之间的多样性。

给定查询与文档之间的距离以及所有文档之间的距离，该算法：

1. 首先选择最相关的文档
2. 通过组合以下方式迭代选择其他文档：
   - 与原始查询的相关性
   - 与之前所选文档的多样性

相关性和多样性之间的平衡由权重控制：
- `DIVERSITY_WEIGHT`：与现有选择差异的重要性
- `RELEVANCE_WEIGHT`：与查询相关性的重要性
- `SIGMA`：概率转换的平滑参数

该算法返回所选文档及其选择分数，使其适用于搜索结果等应用，在这些应用中您需要相关但多样的结果。

例如，在搜索新闻文章时，它首先返回最相关的文章，然后找到既切题又提供新信息的文章，避免冗余选择。

```python
# 配置参数
DIVERSITY_WEIGHT = 1.0  # 文档选择中多样性的权重
RELEVANCE_WEIGHT = 1.0  # 与查询相关性的权重
SIGMA = 0.1  # 概率分布的平滑参数

def greedy_dartsearch(
    query_distances: np.ndarray,
    document_distances: np.ndarray,
    documents: List[str],
    num_results: int
) -> Tuple[List[str], List[float]]:
    """
    执行贪心飞镖板搜索以选择平衡相关性和多样性的前 k 个文档。

    Args:
        query_distances: 查询与每个文档之间的距离
        document_distances: 文档之间的成对距离
        documents: 文档文本列表
        num_results: 要返回的文档数量

    Returns:
        包含以下内容的元组：
        - 所选文档文本列表
        - 每个文档的选择分数列表
    """
    # 避免概率计算中的除零错误
    sigma = max(SIGMA, 1e-5)

    # 将距离转换为概率分布
    query_probabilities = lognorm(query_distances, sigma)
    document_probabilities = lognorm(document_distances, sigma)

    # 用最相关的文档初始化
    most_relevant_idx = np.argmax(query_probabilities)
    selected_indices = np.array([most_relevant_idx])
    selection_scores = [1.0] # 第一个文档的虚拟分数
    # 从第一个所选文档获取初始距离
    max_distances = document_probabilities[most_relevant_idx]

    # 选择剩余文档
    while len(selected_indices) < num_results:
        # 考虑新文档更新最大距离
        updated_distances = np.maximum(max_distances, document_probabilities)

        # 计算组合的多样性和相关性分数
        combined_scores = (
            updated_distances * DIVERSITY_WEIGHT +
            query_probabilities * RELEVANCE_WEIGHT
        )

        # 归一化分数并屏蔽已选文档
        normalized_scores = logsumexp(combined_scores, axis=1)
        normalized_scores[selected_indices] = -np.inf

        # 选择最佳剩余文档
        best_idx = np.argmax(normalized_scores)
        best_score = np.max(normalized_scores)

        # 更新跟踪变量
        max_distances = updated_distances[best_idx]
        selected_indices = np.append(selected_indices, best_idx)
        selection_scores.append(best_score)

    # 返回所选文档及其分数
    selected_documents = [documents[i] for i in selected_indices]
    return selected_documents, selection_scores
```

## Dartboard 上下文检索

### 用于使用 dartboard 检索的主函数。这替代了 get_context（这是简单 RAG）。它：

1. 获取文本查询，向量化，通过简单 RAG 获取 top k 文档（及其向量）
2. 使用这些向量计算与查询的相似度以及候选匹配之间的相似度
3. 运行 dartboard 算法将候选匹配优化为最终的 k 文档列表
4. 返回最终的文档列表及其分数

```python

def get_context_with_dartboard(
    query: str,
    num_results: int = 5,
    oversampling_factor: int = 3
) -> Tuple[List[str], List[float]]:
    """
    使用 dartboard 算法为查询检索最相关和多样的上下文项。

    Args:
        query: 搜索查询字符串
        num_results: 要返回的上下文项数量（默认值：5）
        oversampling_factor: 用于过采样初始结果以获得更好多样性的因子（默认值：3）

    Returns:
        包含以下内容的元组：
        - 所选上下文文本列表
        - 选择分数列表

    Note:
        该函数使用余弦相似度转换为距离。初始检索获取
        oversampling_factor * num_results 项以确保最终选择中有足够的多样性。
    """
    # 嵌入查询并检索初始候选
    query_embedding = chunks_vector_store.embedding_function.embed_documents([query])
    _, candidate_indices = chunks_vector_store.index.search(
        np.array(query_embedding),
        k=num_results * oversampling_factor
    )

    # 获取候选文档的向量和文本
    candidate_vectors = np.array(
        chunks_vector_store.index.reconstruct_batch(candidate_indices[0])
    )
    candidate_texts = [idx_to_text(idx) for idx in candidate_indices[0]]

    # 计算距离矩阵
    # 使用 1 - 余弦相似度作为距离度量
    document_distances = 1 - np.dot(candidate_vectors, candidate_vectors.T)
    query_distances = 1 - np.dot(query_embedding, candidate_vectors.T)

    # 应用 dartboard 选择算法
    selected_texts, selection_scores = greedy_dartsearch(
        query_distances,
        document_distances,
        candidate_texts,
        num_results
    )

    return selected_texts, selection_scores
```

### dartboard 检索 - 相同查询、k 和数据集的结果
- 如您现在所见，top 3 结果不仅仅是重复。

```python
texts,scores=get_context_with_dartboard(test_query,k=3)
show_context(texts)

```

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--dartboard)
