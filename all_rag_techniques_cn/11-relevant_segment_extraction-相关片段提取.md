# 相关片段提取 (RSE)

## 概述

相关片段提取（RSE）是一种从检索到的块中重建连续文本的多块片段的方法。此步骤发生在向量搜索之后（以及可选的重排序之后），但在将检索到的上下文呈现给 LLM 之前。此方法确保相邻的块按其在原始文档中出现的顺序呈现给 LLM。它还添加了未被标记为相关但夹在高度相关块之间的块，进一步改善了提供给 LLM 的上下文。如本笔记本末尾呈现的评估结果所示，此方法显著改善了 RAG 性能。

## 动机

在为 RAG 分块文档时，选择正确的块大小是一个权衡过程。大块比小块为 LLM 提供更好的上下文，但它们也使得更难精确检索特定的信息片段。某些查询（如简单的知识点问题）最好由小块处理，而其他查询（如更高层次的问题）需要非常大的块。有些查询可以用文档中的单个句子回答，而其他查询需要整个部分或章节才能正确回答。大多数现实世界的 RAG 用例面临着这些类型查询的组合。

我们真正需要的是一个更动态的系统，可以在只需要时检索短块，但也可以在需要时检索非常大的块。我们该怎么做？

我们的解决方案源于一个简单的洞察：**相关块倾向于在其原始文档中聚集**。

## 关键组件

#### 块文本键值存储
RSE 需要能够使用 doc_id 和 chunk_index 作为键从数据库快速检索块文本。这是因为并非所有需要包含在给定片段中的块都会在初始搜索结果中返回。因此，除了向量数据库之外，可能还需要使用某种键值存储。

## 方法细节

#### 文档分块
可以使用标准的文档分块方法。这里唯一的特殊要求是文档分块时没有重叠。这允许我们通过连接块来重建文档的部分（即片段）。

#### RSE 优化
在标准块检索过程完成后（理想情况下包括重排序步骤），RSE 过程就可以开始了。第一步是组合绝对相关性值（即相似度分数）和相关性排名。这比单独使用相似度分数或单独使用排名提供了更稳健的起点。然后我们从每个块的值中减去一个恒定的阈值（比如 0.2），使得不相关的块具有负值（低至 -0.2），而相关的块具有正值（高达 0.8）。通过这种方式计算块值，我们可以将片段值定义为块值的总和。

例如，假设文档中块 0-4 具有以下块值：[-0.2, -0.2, 0.4, 0.8, -0.1]。仅包含块 2-3 的片段的值为 0.4+0.8=1.2。

然后，找到最佳片段就变成了最大和子数组问题的约束版本。我们使用带有一些启发式方法的暴力搜索使其高效。这通常需要约 5-10 毫秒。

![相关片段提取](../images/relevant-segment-extraction.svg)

# 设置
首先，一些设置。您需要一个 Cohere API 密钥来运行其中一些单元格，因为我们使用他们优秀的重排序器来计算相关性分数。

# 包安装和导入

下面的单元格安装运行此笔记本所需的所有必要包。

```python
# 安装所需的包
!pip install matplotlib numpy python-dotenv
```

```python
import os
import numpy as np
from typing import List
from scipy.stats import beta
import matplotlib.pyplot as plt
import cohere
from dotenv import load_dotenv

# 从 .env 文件加载环境变量
load_dotenv()
os.environ["CO_API_KEY"] = os.getenv('CO_API_KEY') # Cohere API 密钥
```

我们定义了一些辅助函数。我们将使用 Cohere Rerank API 来计算块的相关性值。通常，我们会从向量和/或关键词搜索开始以缩小候选列表，但由于这里我们只处理单个文档，我们可以直接将所有块发送到重排序器，使事情更简单一些。

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_into_chunks(text: str, chunk_size: int):
    """
    使用 RecursiveCharacterTextSplitter 将给定文本分割成指定大小的块。

    参数：
        text (str): 要分割成块的输入文本。
        chunk_size (int, optional): 每个块的最大大小。默认值为 800。

    返回：
        list[str]: 文本块列表。

    示例：
        >>> text = "This is a sample text to be split into chunks."
        >>> chunks = split_into_chunks(text, chunk_size=10)
        >>> print(chunks)
        ['This is a', 'sample', 'text to', 'be split', 'into', 'chunks.']
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0, length_function=len)
    texts = text_splitter.create_documents([text])
    chunks = [text.page_content for text in texts]
    return chunks

def transform(x: float):
    """
    转换函数，将绝对相关性值映射到 0 和 1 之间更均匀分布的值。Cohere 重排序器给出的相关性值往往非常接近 0 或 1。这里使用的 beta 函数有助于更均匀地分布这些值。

    参数：
        x (float): Cohere 重排序器返回的绝对相关性值

    返回：
        float: 转换后的相关性值
    """
    a, b = 0.4, 0.4  # 这些可以调整以改变分布形状
    return beta.cdf(x, a, b)

def rerank_chunks(query: str, chunks: List[str]):
    """
    使用 Cohere Rerank API 重新排序搜索结果

    参数：
        query (str): 搜索查询
        chunks (list): 要重新排序的块列表

    返回：
        similarity_scores (list): 每个块的相似度分数列表
        chunk_values (list): 每个块的相关性值列表（排名和相似度的融合）
    """
    model = "rerank-english-v3.0"
    client = cohere.Client(api_key=os.environ["CO_API_KEY"])
    decay_rate = 30

    reranked_results = client.rerank(model=model, query=query, documents=chunks)
    results = reranked_results.results
    reranked_indices = [result.index for result in results]
    reranked_similarity_scores = [result.relevance_score for result in results] # 按 reranked_indices 顺序

    # 转换回原始文档顺序并计算块值
    similarity_scores = [0] * len(chunks)
    chunk_values = [0] * len(chunks)
    for i, index in enumerate(reranked_indices):
        absolute_relevance_value = transform(reranked_similarity_scores[i])
        similarity_scores[index] = absolute_relevance_value
        chunk_values[index] = np.exp(-i/decay_rate)*absolute_relevance_value # 根据排名衰减相关性值

    return similarity_scores, chunk_values

def plot_relevance_scores(chunk_values: List[float], start_index: int = None, end_index: int = None) -> None:
    """
    可视化文档中每个块与搜索查询的相关性分数

    参数：
        chunk_values (list): 每个块的相关性值列表
        start_index (int): 要绘制的块的起始索引
        end_index (int): 要绘制的块的结束索引

    返回：
        None

    绘图：
        文档中每个块与搜索查询相关性分数的散点图
    """
    plt.figure(figsize=(12, 5))
    plt.title(f"文档中每个块与搜索查询的相似度")
    plt.ylim(0, 1)
    plt.xlabel("块索引")
    plt.ylabel("查询 - 块相似度")
    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = len(chunk_values)
    plt.scatter(range(start_index, end_index), chunk_values[start_index:end_index])
```

```python
# 下载所需的数据文件
import os
os.makedirs('data', exist_ok=True)

# 下载本笔记本使用的 PDF 文档
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
!wget -O data/nike_2023_annual_report.txt https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/nike_2023_annual_report.txt
```

```python
# 输入文档的文件路径
FILE_PATH = "data/nike_2023_annual_report.txt"

with open(FILE_PATH, 'r') as file:
    text = file.read()

chunks = split_into_chunks(text, chunk_size=800)

print (f"将文档分割成 {len(chunks)} 个块")
```

# 可视化单个文档中的块相关性分布

```python
# 示例查询，需要比单个块更长的结果
query = "Nike 合并财务报表"

similarity_scores, chunk_values = rerank_chunks(query, chunks)
```

```python
plot_relevance_scores(chunk_values)
```

### 如何解读上面的块相关性图
在上面的图中，x 轴表示块索引。文档中的第一个块索引为 0，下一个块索引为 1，以此类推。y 轴表示每个块与查询的相关性。以这种方式查看，我们可以看到相关块倾向于如何聚集在文档的一个或多个部分中。

注意：此图中的相关性值实际上是原始相关性值和相关性排名的组合。对排名应用指数衰减函数，然后将其乘以原始相关性值。使用这种组合提供了比单独使用其中一种更稳健的相关性度量。

### 放大查看
现在让我们放大查看相关块集群以进行更仔细的观察。

```python
plot_relevance_scores(chunk_values, 320, 340)
```

这里有趣的是，这 20 个块中只有 7 个被我们的重排序器标记为相关。许多不相关的块夹在相关块之间。查看 323-336 的范围，恰好一半的块被标记为相关，另一半被标记为不相关。

### 让我们看看这部分文档包含什么

```python
def print_document_segment(chunks: List[str], start_index: int, end_index: int):
    """
    打印文档片段的文本内容

    参数：
        chunks (list): 文本块列表
        start_index (int): 片段的起始索引
        end_index (int): 片段的结束索引（不包含）

    返回：
        None

    打印：
        文档指定片段的文本内容
    """
    for i in range(start_index, end_index):
        print(f"\n块 {i}")
        print(chunks[i])

print_document_segment(chunks, 320, 340)
```

我们可以看到，合并利润表从块 323 开始，直到块 333 的所有内容都包含合并财务报表，这正是我们正在寻找的。因此，该范围内的每个块确实都与我们的查询相关且必要，但只有大约一半的块被重排序器标记为相关。因此，除了向 LLM 提供更完整的上下文外，通过组合这些相关块集群，我们实际上发现了否则会被忽略的重要块。

### 我们可以对这些相关块集群做什么？
核心思想是，相关块集群以其原始连续形式，为 LLM 提供了比单个块更好的上下文。现在是困难的部分：我们如何实际识别这些集群？

如果我们能以一种方式计算块值，使得片段的值只是其组成块的值之和，那么找到最佳片段就是最大子数组问题的一个版本，可以相对容易地找到解决方案。我们如何以这种方式定义块值？我们将从高度相关的块是好的，不相关的块是坏的这一想法开始。我们已经有了一个很好的块相关性度量，在 0-1 的范围内，所以我们需要做的就是从中减去一个恒定的阈值。这将使不相关块的块值变为负数，同时保持相关块的值为正。我们称之为`irrelevant_chunk_penalty`。经验上，大约 0.2 的值效果很好。

```python
def get_best_segments(relevance_values: list, max_length: int, overall_max_length: int, minimum_value: float):
    """
    此函数获取块相关性值，然后运行优化算法以找到最佳片段。用更技术的术语来说，它解决了约束版本的最大和子数组问题。

    注意：这是一个简化的实现，旨在用于演示目的。生产用途需要更复杂的实现，可在 dsRAG 库中获得。

    参数：
        relevance_values (list): 文档每个块的相关性值列表
        max_length (int): 单个片段的最大长度（以块数衡量）
        overall_max_length (int): 所有片段的最大长度（以块数衡量）
        minimum_value (float): 片段必须具有的最小值才能被视为最佳片段

    返回：
        best_segments (list): 元组 (start, end) 列表，表示文档中最佳片段的索引（结束索引不包含）
        scores (list): 每个最佳片段的分数列表
    """
    best_segments = []
    scores = []
    total_length = 0
    while total_length < overall_max_length:
        # 找到剩余的最佳片段
        best_segment = None
        best_value = -1000
        for start in range(len(relevance_values)):
            # 跳过负值起点
            if relevance_values[start] < 0:
                continue
            for end in range(start+1, min(start+max_length+1, len(relevance_values)+1)):
                # 跳过负值终点
                if relevance_values[end-1] < 0:
                    continue
                # 检查此片段是否与任何最佳片段重叠，如果重叠则跳过
                if any(start < seg_end and end > seg_start for seg_start, seg_end in best_segments):
                    continue
                # 检查此片段是否会使我们超过总体最大长度，如果会则跳过
                if total_length + end - start > overall_max_length:
                    continue

                # 定义片段值为其块的相关性值之和
                segment_value = sum(relevance_values[start:end])
                if segment_value > best_value:
                    best_value = segment_value
                    best_segment = (start, end)

        # 如果没有找到有效片段，则完成
        if best_segment is None or best_value < minimum_value:
            break

        # 否则，将片段添加到最佳片段列表
        best_segments.append(best_segment)
        scores.append(best_value)
        total_length += best_segment[1] - best_segment[0]

    return best_segments, scores
```

```python
# 定义优化的一些参数和约束
irrelevant_chunk_penalty = 0.2 # 经验上，大约 0.2 效果很好；较低的值偏向更长的片段
max_length = 20
overall_max_length = 30
minimum_value = 0.7

# 从块相关性值中减去恒定阈值
relevance_values = [v - irrelevant_chunk_penalty for v in chunk_values]

# 运行优化
best_segments, scores = get_best_segments(relevance_values, max_length, overall_max_length, minimum_value)

# 打印结果
print ("最佳片段索引")
print (best_segments) # 最佳片段的索引，结束索引不包含
print ()
print ("最佳片段分数")
print (scores)
print ()
```

优化算法给出的第一个片段是块 323-336。手动查看块后，我们认为 323-333 是理想的片段，所以我们得到了一些我们并不真正需要的额外块，但总体而言，这将是 LLM 处理的一个很好的上下文片段。我们还从文档的其他部分识别了一些较短的片段，这些片段也可以提供给 LLM。

### 如果答案包含在单个块中怎么办？
在只有单个块或少数几个孤立的块与查询相关的情况下，我们不想从它们创建大的片段。我们只想返回那些特定的块。RSE 也可以很好地处理这种情况。由于没有相关块集群，在这种情况下它基本上退化为标准的 top-k 检索。我们将留给读者作为练习，看看对于此类查询，块相关性图和结果最佳片段会发生什么。

# 评估结果

### KITE
我们在我们创建的端到端 RAG 基准测试 [KITE](https://github.com/D-Star-AI/KITE)（知识密集型任务评估）上评估了 RSE。

KITE 目前由 4 个数据集和总共 50 个问题组成。
- **AI Papers** - 约 100 篇关于 AI 和 RAG 的学术论文，以 PDF 形式从 arXiv 下载。
- **BVP Cloud 10-Ks** - Bessemer Cloud Index 中所有公司（约 70 家）的 10-K，以 PDF 形式提供。
- **Sourcegraph Company Handbook** - 约 800 个 markdown 文件，保留其原始目录结构，从 Sourcegraph 公开访问的公司手册 GitHub[页面](https://github.com/sourcegraph/handbook/tree/main/content) 下载。
- **Supreme Court Opinions** - 2022 年任期年度的所有最高法院意见（从 23 年 1 月到 23 年 6 月发布），以 PDF 形式从官方最高法院 [网站](https://www.supremecourt.gov/opinions/slipopinion/22) 下载。

每个样本都包含标准答案。大多数样本还包含评分标准。评分按 0-10 分进行，由强大的 LLM 进行评分。

我们将 RSE 与标准 Top-k 检索（k=20）进行比较。两种配置之间的所有其他参数保持相同。我们使用 Cohere 3 重排序器，并使用 GPT-4o 进行响应生成。相关知识字符串的平均长度在两种配置之间大致相同，因此成本和延迟相似。

|                         | Top-k    | RSE    |
|-------------------------|----------|--------|
| AI Papers               | 4.5      | 7.9    |
| BVP Cloud               | 2.6      | 4.4    |
| Sourcegraph             | 5.7      | 6.6    |
| Supreme Court Opinions  | 6.1      | 8.0    |
| **Average**             | 4.72     | 6.73   |

我们可以看到，RSE 在所有四个数据集上都带来了性能提升。总体平均分从 4.72 增加到 6.73，增长了 42.6%。

### FinanceBench
我们还在 FinanceBench 上评估了 RSE，它贡献了 83% 的分数，而基准分数为 19%。对于该基准测试，我们联合测试了上下文块头（CCH）和 RSE，因此我们无法准确说明 RSE 对结果贡献了多少。但 CCH 和 RSE 的组合显然在 FinanceBench 上带来了显著的准确性改进。

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--relevant-segment-extraction)
