# 文档检索中的层次索引

## 概述

本代码实现了一个用于文档检索的层次索引系统，采用两个编码层次：文档级别的摘要和详细的文本块。该方法旨在通过首先通过摘要识别相关文档部分，然后深入到这些部分的具体细节，来提高信息检索的效率和相关性。

## 动机

传统的扁平索引方法在处理大型文档或语料库时可能会遇到困难，可能会遗漏上下文或返回不相关的信息。层次索引通过创建一个两层搜索系统来解决这个问题，允许进行更高效和更具上下文感知的检索。

## 核心组件

1. PDF 文档处理和文本分块
2. 使用 OpenAI 的 GPT-4 进行异步文档摘要
3. 使用 FAISS 和 OpenAI Embedding 为摘要和详细文本块创建向量存储
4. 自定义层次检索函数

## 方法细节

### 文档预处理和编码

1. 加载 PDF 并将其分割成文档（可能按页分割）。
2. 使用 GPT-4 异步摘要每个文档。
3. 原始文档也被分割成更小的、详细的文本块。
4. 创建两个独立的向量存储：
   - 一个用于文档级别的摘要
   - 一个用于详细的文本块

### 异步处理和速率限制

1. 代码使用异步编程（asyncio）来提高效率。
2. 实现批处理和指数退避来处理 API 速率限制。

### 层次检索

`retrieve_hierarchical` 函数实现两层搜索：

1. 首先搜索摘要向量存储以识别相关的文档部分。
2. 对于每个相关摘要，然后搜索详细文本块向量存储，按相应的页码进行过滤。
3. 这种方法确保只从最相关的文档部分检索详细信息。

## 该方法的优势

1. **提高检索效率**：通过首先搜索摘要，系统可以快速识别相关文档部分，而无需处理所有详细文本块。
2. **更好的上下文保持**：层次方法有助于保持检索信息的更广泛上下文。
3. **可扩展性**：这种方法对于大型文档或语料库特别有益，因为扁平搜索可能效率低下或遗漏重要上下文。
4. **灵活性**：系统允许调整检索的摘要和文本块数量，可以为不同用例进行微调。

## 实现细节

1. **异步编程**：利用 Python 的 asyncio 进行高效的 I/O 操作和 API 调用。
2. **速率限制处理**：实现批处理和指数退避来有效管理 API 速率限制。
3. **持久存储**：将生成的向量存储在本地保存，以避免不必要的重复计算。

## 结论

层次索引代表了一种复杂的文档检索方法，特别适合大型或复杂的文档集。通过同时利用高级摘要和详细文本块，它在广泛上下文理解和具体信息检索之间提供了平衡。这种方法在需要高效和上下文感知信息检索的各个领域都有潜在应用，例如法律文档分析、学术研究或大规模内容管理系统。

<div style="text-align: center;">

<img src="../images/hierarchical_indices.svg" alt="hierarchical_indices" style="width:50%; height:auto;">
</div>

<div style="text-align: center;">

<img src="../images/hierarchical_indices_example.svg" alt="hierarchical_indices" style="width:100%; height:auto;">
</div>

# 包安装和导入

下面的单元格安装了运行此 notebook 所需的所有包。

```python
# 安装所需的包
!pip install langchain langchain-openai python-dotenv
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
import asyncio
import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains.summarize.chain import load_summarize_chain
from langchain.docstore.document import Document

# 原始路径追加已替换为 Colab 兼容性
from helper_functions import *
from evaluation.evalute_rag import *
from helper_functions import encode_pdf, encode_from_string

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

### 编码到摘要和文本块两个层次并共享页面元数据的函数

```python
async def encode_pdf_hierarchical(path, chunk_size=1000, chunk_overlap=200, is_string=False):
    """
    使用 OpenAI embeddings 异步将 PDF 书籍编码为层次向量存储。
    包含使用指数退避的速率限制处理。

    Args:
        path: PDF 文件的路径。
        chunk_size: 每个文本块的期望大小。
        chunk_overlap: 连续块之间的重叠量。

    Returns:
        包含两个 FAISS 向量存储的元组：
        1. 文档级别摘要
        2. 详细文本块
    """

    # 加载 PDF 文档
    if not is_string:
        loader = PyPDFLoader(path)
        documents = await asyncio.to_thread(loader.load)
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            # 设置一个非常小的块大小，仅用于演示。
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        documents = text_splitter.create_documents([path])


    # 创建文档级别摘要
    summary_llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=4000)
    summary_chain = load_summarize_chain(summary_llm, chain_type="map_reduce")

    async def summarize_doc(doc):
        """
        使用速率限制处理摘要单个文档。

        Args:
            doc: 要被摘要的文档。

        Returns:
            摘要后的 Document 对象。
        """
        # 使用指数退避重试摘要
        summary_output = await retry_with_exponential_backoff(summary_chain.ainvoke([doc]))
        summary = summary_output['output_text']
        return Document(
            page_content=summary,
            metadata={"source": path, "page": doc.metadata["page"], "summary": True}
        )

    # 分批处理文档以避免速率限制
    batch_size = 5  # 根据你的速率限制调整此值
    summaries = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        batch_summaries = await asyncio.gather(*[summarize_doc(doc) for doc in batch])
        summaries.extend(batch_summaries)
        await asyncio.sleep(1)  # 批次之间的短暂暂停

    # 将文档分割为详细文本块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    detailed_chunks = await asyncio.to_thread(text_splitter.split_documents, documents)

    # 更新详细文本块的元数据
    for i, chunk in enumerate(detailed_chunks):
        chunk.metadata.update({
            "chunk_id": i,
            "summary": False,
            "page": int(chunk.metadata.get("page", 0))
        })

    # 创建 embeddings
    embeddings = OpenAIEmbeddings()

    # 使用速率限制处理异步创建向量存储
    async def create_vectorstore(docs):
        """
        使用速率限制处理从文档列表创建向量存储。

        Args:
            docs: 要被嵌入的文档列表。

        Returns:
            包含嵌入文档的 FAISS 向量存储。
        """
        return await retry_with_exponential_backoff(
            asyncio.to_thread(FAISS.from_documents, docs, embeddings)
        )

    # 并发地为摘要和详细文本块生成向量存储
    summary_vectorstore, detailed_vectorstore = await asyncio.gather(
        create_vectorstore(summaries),
        create_vectorstore(detailed_chunks)
    )

    return summary_vectorstore, detailed_vectorstore
```

### 如果向量存储不存在，将 PDF 编码到文档级别摘要和详细文本块

```python
if os.path.exists("../vector_stores/summary_store") and os.path.exists("../vector_stores/detailed_store"):
   embeddings = OpenAIEmbeddings()
   summary_store = FAISS.load_local("../vector_stores/summary_store", embeddings, allow_dangerous_deserialization=True)
   detailed_store = FAISS.load_local("../vector_stores/detailed_store", embeddings, allow_dangerous_deserialization=True)

else:
    summary_store, detailed_store = await encode_pdf_hierarchical(path)
    summary_store.save_local("../vector_stores/summary_store")
    detailed_store.save_local("../vector_stores/detailed_store")
```

### 根据摘要层次检索信息，然后从文本块层次向量存储检索信息并根据摘要层次的页面进行过滤

```python
def retrieve_hierarchical(query, summary_vectorstore, detailed_vectorstore, k_summaries=3, k_chunks=5):
    """
    使用查询执行层次检索。

    Args:
        query: 搜索查询。
        summary_vectorstore: 包含文档摘要的向量存储。
        detailed_vectorstore: 包含详细文本块的向量存储。
        k_summaries: 要检索的顶部摘要数量。
        k_chunks: 每个摘要要检索的详细文本块数量。

    Returns:
        相关详细文本块的列表。
    """

    # 检索顶部摘要
    top_summaries = summary_vectorstore.similarity_search(query, k=k_summaries)

    relevant_chunks = []
    for summary in top_summaries:
        # 对于每个摘要，检索相关的详细文本块
        page_number = summary.metadata["page"]
        page_filter = lambda metadata: metadata["page"] == page_number
        page_chunks = detailed_vectorstore.similarity_search(
            query,
            k=k_chunks,
            filter=page_filter
        )
        relevant_chunks.extend(page_chunks)

    return relevant_chunks
```

### 用例演示

```python
query = "What is the greenhouse effect?"
results = retrieve_hierarchical(query, summary_store, detailed_store)

# 打印结果
for chunk in results:
    print(f"Page: {chunk.metadata['page']}")
    print(f"Content: {chunk.page_content}...")  # 打印前 100 个字符
    print("---")
```

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--hierarchical-indices)
