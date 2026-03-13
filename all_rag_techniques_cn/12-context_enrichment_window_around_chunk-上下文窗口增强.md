# 文档检索的上下文窗口增强

## 概述

此代码实现了向量数据库中文档检索的上下文窗口增强技术。它通过为每个检索到的块添加周围上下文来增强标准检索过程，从而改善返回信息的连贯性和完整性。

## 动机

传统的向量搜索通常返回孤立的文本块，可能缺乏完整理解所需的上下文。此方法旨在通过包含相邻的文本块来提供检索信息的更全面视图。

## 关键组件

1. PDF 处理和文本分块
2. 使用 FAISS 和 OpenAI embeddings 创建向量存储
3. 带上下文窗口的自定义检索函数
4. 标准检索与上下文增强检索的比较

## 方法细节

### 文档预处理

1. PDF 被读取并转换为字符串。
2. 文本被分割成带有重叠的块，每个块标记有其索引。

### 向量存储创建

1. 使用 OpenAI embeddings 为块创建向量表示。
2. 从这些 embeddings 创建 FAISS 向量存储。

### 上下文增强检索

1. `retrieve_with_context_overlap` 函数执行以下步骤：
   - 基于查询检索相关块
   - 对于每个相关块，获取相邻块
   - 连接块，考虑重叠
   - 返回每个相关块的扩展上下文

### 检索比较

笔记本包含一个比较标准检索和上下文增强方法的部分。

## 此方法的优势

1. 提供更连贯和上下文丰富的结果
2. 保持向量搜索的优势，同时减轻其返回孤立文本片段的倾向
3. 允许灵活调整上下文窗口大小

## 结论

此上下文窗口增强技术为改善向量文档搜索系统中检索信息的质量提供了一种有前景的方法。通过提供周围的上下文，它有助于保持检索信息的连贯性和完整性，潜在地在问答等下游任务中带来更好的理解和更准确的响应。

<div style="text-align: center;">

<img src="../images/vector-search-comparison_context_enrichment.svg" alt="上下文窗口增强" style="width:70%; height:auto;">
</div>

<div style="text-align: center;">

<img src="../images/context_enrichment_window.svg" alt="上下文窗口增强" style="width:70%; height:auto;">
</div>

# 包安装和导入

下面的单元格安装运行此笔记本所需的所有必要包。

```python
# 安装所需的包
!pip install langchain python-dotenv
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


# 原始路径追加已替换为 Colab 兼容性
from helper_functions import *
from evaluation.evalute_rag import *

# 从 .env 文件加载环境变量
load_dotenv()

# 设置 OpenAI API 密钥环境变量
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
```

### 定义 PDF 路径

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

### 将 PDF 读取为字符串

```python
content = read_pdf_to_string(path)
```

### 将文本分割为带有块时间顺序索引元数据的块的函数

```python
def split_text_to_chunks_with_indices(text: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    使用 RecursiveCharacterTextSplitter 将给定文本分割成指定大小的块。

    参数：
        text (str): 要分割成块的输入文本。
        chunk_size (int, optional): 每个块的最大大小。默认值为 800。

    返回：
        list[str]: 文本块列表。

    示例：
        >>> text = "This is a sample text to be split into chunks."
        >>> chunks = split_text_to_chunks_with_indices(text, chunk_size=10)
        >>> print(chunks)
        ['This is a', 'sample', 'text to', 'be split', 'into', 'chunks.']
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(Document(page_content=chunk, metadata={"index": len(chunks), "text": text}))
        start += chunk_size - chunk_overlap
    return chunks
```

### 相应地分割我们的文档

```python
chunks_size = 400
chunk_overlap = 200
docs = split_text_to_chunks_with_indices(content, chunks_size, chunk_overlap)
```

### 创建向量存储和检索器

```python
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
chunks_query_retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
```

### 从向量存储中获取第 k 个块（按原始顺序）的函数

```python
def get_chunk_by_index(vectorstore, target_index: int) -> Document:
    """
    根据块在其元数据中的索引从 vectorstore 检索块。

    参数：
        vectorstore (VectorStore): 包含块的向量存储。
        target_index (int): 要检索的块的索引。

    返回：
        Optional[Document]: 检索到的块作为 Document 对象，如果未找到则返回 None。
    """
    # 这是一个简化版本。在实践中，根据向量存储的实现，
    # 可能需要更有效的方法来按索引检索块。
    all_docs = vectorstore.similarity_search("", k=vectorstore.index.ntotal)
    for doc in all_docs:
        if doc.metadata.get('index') == target_index:
            return doc
    return None
```

### 检查函数

```python
chunk = get_chunk_by_index(vectorstore, 0)
print(chunk.page_content)
```

### 基于语义相似性从向量存储检索，然后用其前后的 num_neighbors 个邻居填充每个检索到的块，考虑块重叠以在其周围构建有意义的大窗口

```python
def retrieve_with_context_overlap(vectorstore, retriever, query: str, num_neighbors: int = 1, chunk_size: int = 200, chunk_overlap: int = 20) -> List[str]:
    """
    基于查询检索块，然后获取相邻块并连接它们，
    考虑重叠和正确的索引。

    参数：
        vectorstore (VectorStore): 包含块的向量存储。
        retriever: 用于获取相关文档的检索器对象。
        query (str): 搜索相关块的查询。
        num_neighbors (int): 在每个相关块之前和之后检索的块数。
        chunk_size (int): 最初分割时每个块的大小。
        chunk_overlap (int): 最初分割时块之间的重叠。

    返回：
        List[str]: 连接块序列的列表，每个序列以相关块为中心。
    """
    relevant_chunks = retriever.get_relevant_documents(query)
    result_sequences = []

    for chunk in relevant_chunks:
        current_index = chunk.metadata.get('index')
        if current_index is None:
            continue

        # 确定要检索的块范围
        start_index = max(0, current_index - num_neighbors)
        end_index = current_index + num_neighbors + 1  # +1 因为范围在结束时是-exclusive 的

        # 检索范围内的所有块
        neighbor_chunks = []
        for i in range(start_index, end_index):
            neighbor_chunk = get_chunk_by_index(vectorstore, i)
            if neighbor_chunk:
                neighbor_chunks.append(neighbor_chunk)

        # 按其索引排序块以确保正确顺序
        neighbor_chunks.sort(key=lambda x: x.metadata.get('index', 0))

        # 连接块，考虑重叠
        concatenated_text = neighbor_chunks[0].page_content
        for i in range(1, len(neighbor_chunks)):
            current_chunk = neighbor_chunks[i].page_content
            overlap_start = max(0, len(concatenated_text) - chunk_overlap)
            concatenated_text = concatenated_text[:overlap_start] + current_chunk

        result_sequences.append(concatenated_text)

    return result_sequences
```

### 比较常规检索和带上下文窗口的检索

```python
# 基线方法
query = "解释森林砍伐和化石燃料在气候变化中的作用。"
baseline_chunk = chunks_query_retriever.get_relevant_documents(query, k=1)
# 聚焦上下文增强方法
enriched_chunks = retrieve_with_context_overlap(
    vectorstore,
    chunks_query_retriever,
    query,
    num_neighbors=1,
    chunk_size=400,
    chunk_overlap=200
)

print("基线块：")
print(baseline_chunk[0].page_content)
print("\n增强块：")
print(enriched_chunks[0])
```

### 一个展示额外上下文窗口优势的示例

```python
document_content = """
人工智能（AI）有着丰富的历史，可以追溯到 20 世纪中叶。"人工智能"一词于 1956 年在达特茅斯会议上被创造出来，标志着该领域的正式开始。

在 1950 年代和 1960 年代，AI 研究专注于符号方法和问题解决。Logic Theorist 由 Allen Newell 和 Herbert A. Simon 于 1955 年创建，通常被认为是第一个 AI 程序。

1960 年代见证了专家系统的发展，该系统使用预定义规则来解决复杂问题。DENDRAL 创建于 1965 年，是最早的专家系统之一，设计用于分析化学化合物。

然而，1970 年代带来了第一个"AI 冬季"，这是一个资金减少和 AI 研究兴趣降低的时期，主要是由于承诺过多而成果不足。

1980 年代见证了专家系统在公司中的普及而复兴。日本政府的第五代计算机项目也刺激了全球 AI 研究投资的增加。

神经网络在 1980 年代和 1990 年代崭露头角。反向传播算法虽然更早被发现，但在此期间被广泛用于训练多层网络。

1990 年代后期和 2000 年代标志着机器学习方法的兴起。支持向量机（SVM）和随机森林在各种分类和回归任务中变得流行。

深度学习是机器学习的一个子集，使用具有多层的神经网络，在 2010 年代初期开始显示出有希望的结果。突破出现在 2012 年，当时深度神经网络在 ImageNet 竞赛中显著优于其他机器学习方法。

从那时起，深度学习彻底改变了许多 AI 应用，包括图像和语音识别、自然语言处理和游戏。2016 年，Google 的 AlphaGo 击败了世界冠军围棋选手，这是 AI 的一个里程碑式成就。

当前的 AI 时代特征是将深度学习与其他 AI 技术集成，开发更高效和更强大的硬件，以及围绕 AI 部署的伦理考虑。

Transformers 于 2017 年推出，已成为自然语言处理中的主导架构，使 GPT（生成式预训练 Transformer）等模型能够生成类似人类的文本。

随着 AI 的不断发展，新的挑战 и机遇也随之出现。可解释 AI、稳健和公平的机器学习以及人工通用智能（AGI）是该领域当前和未来研究的关键领域。
"""

chunks_size = 250
chunk_overlap = 20
document_chunks = split_text_to_chunks_with_indices(document_content, chunks_size, chunk_overlap)
document_vectorstore = FAISS.from_documents(document_chunks, embeddings)
document_retriever = document_vectorstore.as_retriever(search_kwargs={"k": 1})

query = "深度学习何时在 AI 中变得突出？"
context = document_retriever.get_relevant_documents(query)
context_pages_content = [doc.page_content for doc in context]

print "常规检索:\n"
show_context(context_pages_content)

sequences = retrieve_with_context_overlap(document_vectorstore, document_retriever, query, num_neighbors=1)
print("\n带上下文增强的检索:\n")
show_context(sequences)
```

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--context-enrichment-window-around-chunk)
