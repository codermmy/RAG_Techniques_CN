# RAG系统中的重排序方法

## 概述
重排序是检索增强生成（RAG）系统中的关键步骤，旨在提高检索文档的相关性和质量。它涉及重新评估和重新排序最初检索的文档，以确保最相关的信息在后续处理或展示中被优先考虑。

## 动机
RAG系统中重排序的主要动机是克服初始检索方法的局限性，这些方法通常依赖于更简单的相似性度量。重排序允许进行更复杂的相关性评估，考虑到查询和文档之间可能被传统检索技术忽略的细微关系。这个过程旨在通过确保在生成阶段使用最相关的信息来提高RAG系统的整体性能。

## 关键组件
重排序系统通常包括以下组件：

1. **初始检索器**：通常是使用基于Embedding的相似性搜索的向量存储。
2. **重排序模型**：可以是：
   - 用于评分相关性的大型语言模型（LLM）
   - 专门为相关性评估训练的Cross-Encoder模型
3. **评分机制**：为文档分配相关性分数的方法
4. **排序和选择逻辑**：根据新分数重新排序文档

## 方法详情
重排序过程通常遵循以下步骤：

1. **初始检索**：获取一组初始的潜在相关文档。
2. **配对创建**：为每个检索到的文档形成查询-文档对。
3. **评分**：
   - LLM方法：使用提示让LLM评估文档相关性。
   - Cross-Encoder方法：直接将查询-文档对输入模型。
4. **分数解释**：解析并归一化相关性分数。
5. **重排序**：根据新的相关性分数对文档进行排序。
6. **选择**：从重排序后的列表中选择前K个文档。

## 此方法的优势
重排序提供了几个优势：

1. **提高相关性**：通过使用更复杂的模型，重排序可以捕捉细微的相关性因素。
2. **灵活性**：可以根据特定需求和资源应用不同的重排序方法。
3. **增强上下文质量**：向RAG系统提供更相关的文档可以提高生成回答的质量。
4. **减少噪音**：重排序有助于过滤掉不太相关的信息，专注于最相关的内容。

## 结论
重排序是RAG系统中一种强大的技术，可以显著提高检索信息的质量。无论是使用基于LLM的评分还是专门的Cross-Encoder模型，重排序都允许对文档相关性进行更细致、更准确的评估。这种改进的相关性直接转化为下游任务的更好性能，使重排序成为高级RAG实现中的重要组成部分。

在基于LLM和Cross-Encoder的重排序方法之间的选择取决于所需的准确性、可用的计算资源和特定应用需求等因素。两种方法都比基本检索方法提供了实质性的改进，并有助于RAG系统的整体有效性。

<div style="text-align: center;">

<img src="../images/reranking-visualization.svg" alt="重排序LLM" style="width:100%; height:auto;">
</div>

<div style="text-align: center;">

<img src="../images/reranking_comparison.svg" alt="重排序对比" style="width:100%; height:auto;">
</div>

# 包安装和导入

下面的单元格安装运行此notebook所需的所有必要包。

```python
# 安装所需的包!pip install faiss-cpu llama-index python-dotenv
```

```python
import osimport sysfrom dotenv import load_dotenvfrom typing import Listfrom llama_index.core import Documentfrom llama_index.core import Settingsfrom llama_index.embeddings.openai import OpenAIEmbeddingfrom llama_index.llms.openai import OpenAIfrom llama_index.core.readers import SimpleDirectoryReaderfrom llama_index.vector_stores.faiss import FaissVectorStorefrom llama_index.core.ingestion import IngestionPipelinefrom llama_index.core.node_parser import SentenceSplitterfrom llama_index.core import VectorStoreIndexfrom llama_index.core.postprocessor import SentenceTransformerRerank, LLMRerankfrom llama_index.core import QueryBundleimport faiss# 原始路径追加已替换为 Colab 兼容# 从 .env 文件加载环境变量load_dotenv()# 设置 OpenAI API 密钥环境变量os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')# Llamaindex 全局设置用于 LLM 和嵌入EMBED_DIMENSION=512Settings.llm = OpenAI(model="gpt-3.5-turbo")Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=EMBED_DIMENSION)
```

### 读取文档

```python
# 下载所需的数据文件import osos.makedirs('data', exist_ok=True)# 下载本 notebook 中使用的 PDF 文档!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
```

```python
path = "data/"
reader = SimpleDirectoryReader(input_dir=path, required_exts=['.pdf'])
documents = reader.load_data()
```

### 创建向量存储

```python
# 创建 FaisVectorStore 以存储嵌入fais_index = faiss.IndexFlatL2(EMBED_DIMENSION)vector_store = FaissVectorStore(faiss_index=fais_index)
```

## 数据摄取管道

```python
base_pipeline = IngestionPipeline(
    transformations=[SentenceSplitter()],
    vector_store=vector_store,
    documents=documents
)

nodes = base_pipeline.run()
```

## 查询

### 方法1：基于LLM的检索文档重排序

<div style="text-align: center;">

<img src="../images/rerank_llm.svg" alt="LLM重排序" style="width:40%; height:auto;">
</div>

```python
# 从基础节点创建向量索引index = VectorStoreIndex(nodes)query_engine_w_llm_rerank = index.as_query_engine(    similarity_top_k=10,    node_postprocessors=[        LLMRerank(            top_n=5        )    ],)
```

```python
resp = query_engine_w_llm_rerank.query("What are the impacts of climate change on biodiversity?")
print(resp)
```

#### 演示为什么我们应该使用重排序的示例

```python
chunks = [
    "The capital of France is great.",
    "The capital of France is huge.",
    "The capital of France is beautiful.",
    """Have you ever visited Paris? It is a beautiful city where you can eat delicious food and see the Eiffel Tower. I really enjoyed all the cities in france, but its capital with the Eiffel Tower is my favorite city.""", 
    "I really enjoyed my trip to Paris, France. The city is beautiful and the food is delicious. I would love to visit again. Such a great capital city."
]
docs = [Document(page_content=sentence) for sentence in chunks]


def compare_rag_techniques(query: str, docs: List[Document] = docs) -> None:
    docs = [Document(text=sentence) for sentence in chunks]
    index = VectorStoreIndex.from_documents(docs)
    
    
    print("Comparison of Retrieval Techniques")
    print("==================================")
    print(f"Query: {query}\n")
    
    print("Baseline Retrieval Result:")
    baseline_docs = index.as_retriever(similarity_top_k=5).retrieve(query)
    for i, doc in enumerate(baseline_docs[:2]): # Get only the first two retrieved docs
        print(f"\nDocument {i+1}:")
        print(doc.text)

    print("\nAdvanced Retrieval Result:")
    reranker = LLMRerank(
        top_n=2,
    )
    advanced_docs = reranker.postprocess_nodes(
            baseline_docs, 
            QueryBundle(query)
        )
    for i, doc in enumerate(advanced_docs):
        print(f"\nDocument {i+1}:")
        print(doc.text)


query = "what is the capital of france?"
compare_rag_techniques(query, docs)
```

### 方法2：Cross-Encoder模型

<div style="text-align: center;">

<img src="../images/rerank_cross_encoder.svg" alt="Cross-Encoder重排序" style="width:40%; height:auto;">
</div>

LlamaIndex内置支持[SBERT](https://www.sbert.net/index.html)模型，可以直接用作节点后处理器。

```python
query_engine_w_cross_encoder = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[
        SentenceTransformerRerank(
            model='cross-encoder/ms-marco-MiniLM-L-6-v2',
            top_n=5
        )
    ],
)

resp = query_engine_w_cross_encoder.query("What are the impacts of climate change on biodiversity?")
print(resp)
```

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--reranking-with-llamaindex)
