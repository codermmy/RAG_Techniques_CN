# RAPTOR: 递归抽象处理与主题组织检索

## 概述
RAPTOR 是一个先进的信息检索和问答系统，它结合了层次化文档摘要、基于 Embedding 的检索和上下文答案生成。它通过创建多级摘要树来高效处理大型文档集合，既能进行广泛的信息检索，也能进行详细的细节检索。

## 动机
传统检索系统在处理大型文档集时常常面临困难，要么遗漏重要细节，要么被无关信息淹没。RAPTOR 通过创建文档集合的层次结构来解决这个问题，使其能够在高层概念和具体细节之间灵活导航。

## 核心组件
1. **树构建 (Tree Building)**: 创建文档摘要的层次结构。
2. **Embedding 和聚类**: 基于语义相似度组织文档和摘要。
3. **向量存储 (Vectorstore)**: 高效存储和检索文档及摘要的 Embedding。
4. **上下文检索器**: 为给定查询选择最相关的信息。
5. **答案生成**: 基于检索到的信息生成连贯的回复。

## 方法细节

### 树构建
1. 从第 0 层的原始文档开始。
2. 对于每一层:
   - 使用语言模型对文本进行 Embedding。
   - 对 Embedding 进行聚类（例如使用高斯混合模型）。
   - 为每个聚类生成摘要。
   - 将这些摘要作为下一层的文本。
3. 持续进行直到达到单个摘要或最大层级。

### Embedding 和检索
1. 对树中所有层级的文档和摘要进行 Embedding。
2. 将这些 Embedding 存储在向量存储（如 FAISS）中以进行高效的相似度搜索。
3. 对于给定查询:
   - 对查询进行 Embedding。
   - 从向量存储中检索最相似的文档/摘要。

### 上下文压缩
1. 获取检索到的文档/摘要。
2. 使用语言模型仅提取与给定查询最相关的部分。

### 答案生成
1. 将相关部分组合成上下文。
2. 使用语言模型基于此上下文和原始查询生成答案。

## 方法优势
1. **可扩展性**: 通过处理不同层级的摘要，能够处理大型文档集合。
2. **灵活性**: 能够提供高层概览和具体细节。
3. **上下文感知**: 从最合适的抽象层级检索信息。
4. **高效性**: 使用 Embedding 和向量存储进行快速检索。
5. **可追溯性**: 维护摘要与原始文档之间的链接，便于来源验证。

## 结论
RAPTOR 代表了信息检索和问答系统的重大进步。通过将层次化摘要与基于 Embedding 的检索和上下文答案生成相结合，它为处理大型文档集合提供了一种强大而灵活的方法。系统在不同抽象层级之间导航的能力使其能够为各种查询提供相关且符合上下文的答案。

虽然 RAPTOR 展现出巨大潜力，但未来的工作可以集中在优化树构建过程、提高摘要质量以及增强检索机制以更好地处理复杂的多面查询。此外，将此方法与其他 AI 技术集成可能会产生更复杂的信息处理系统。

<div style="text-align: center;">

<img src="../images/raptor.svg" alt="RAPTOR" style="width:100%; height:auto;">
</div>

# 包安装与导入

下面的单元格安装运行此 notebook 所需的所有必要包。


```python
# 安装所需的包!pip install faiss-cpu langchain langchain-openai matplotlib numpy pandas python-dotenv scikit-learn
```

```python
# 克隆仓库以访问辅助函数和评估模块!git clone https://github.com/NirDiamant/RAG_TECHNIQUES.gitimport syssys.path.append('RAG_TECHNIQUES')# 如果需要运行最新数据# !cp -r RAG_TECHNIQUES/data .
```

```python
import numpy as npimport pandas as pdfrom typing import List, Dict, Anyfrom sklearn.mixture import GaussianMixturefrom langchain.chains.llm import LLMChainfrom langchain.embeddings import OpenAIEmbeddingsfrom langchain.vectorstores import FAISSfrom langchain_openai import ChatOpenAIfrom langchain.prompts import ChatPromptTemplatefrom langchain.retrievers import ContextualCompressionRetrieverfrom langchain.retrievers.document_compressors import LLMChainExtractorfrom langchain.schema import AIMessagefrom langchain.docstore.document import Documentimport matplotlib.pyplot as pltimport loggingimport osimport sysfrom dotenv import load_dotenv# 原始路径追加已替换为 Colab 兼容from helper_functions import *from evaluation.evalute_rag import *# 从 .env 文件加载环境变量load_dotenv()# 设置 OpenAI API 密钥环境变量os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
```

### 定义日志、LLM 和 Embeddings

```python
# 设置日志记录logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')embeddings = OpenAIEmbeddings()llm = ChatOpenAI(model_name="gpt-4o-mini")
```

### 辅助函数


```python
def extract_text(item):
    """从字符串或 AIMessage 对象中提取文本内容。"""
    if isinstance(item, AIMessage):
        return item.content
    return item

def embed_texts(texts: List[str]) -> List[List[float]]:
    """使用 OpenAIEmbeddings 对文本进行 Embedding。"""
    logging.info(f"Embedding {len(texts)} texts")
    return embeddings.embed_documents([extract_text(text) for text in texts])

def perform_clustering(embeddings: np.ndarray, n_clusters: int = 10) -> np.ndarray:
    """使用高斯混合模型对 Embedding 进行聚类。"""
    logging.info(f"Performing clustering with {n_clusters} clusters")
    gm = GaussianMixture(n_components=n_clusters, random_state=42)
    return gm.fit_predict(embeddings)

def summarize_texts(texts: List[str]) -> str:
    """使用 OpenAI 总结文本列表。"""
    logging.info(f"Summarizing {len(texts)} texts")
    prompt = ChatPromptTemplate.from_template(
        "简洁地总结以下文本：\n\n{text}"
    )
    chain = prompt | llm
    input_data = {"text": texts}
    return chain.invoke(input_data)

def visualize_clusters(embeddings: np.ndarray, labels: np.ndarray, level: int):
    """使用 PCA 可视化聚类。"""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title(f'Cluster Visualization - Level {level}')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()
```

### RAPTOR 核心函数


```python


def build_raptor_tree(texts: List[str], max_levels: int = 3) -> Dict[int, pd.DataFrame]:
    """构建带有层级元数据和父子关系的 RAPTOR 树结构。"""
    results = {}
    current_texts = [extract_text(text) for text in texts]
    current_metadata = [{"level": 0, "origin": "original", "parent_id": None} for _ in texts]
    
    for level in range(1, max_levels + 1):
        logging.info(f"Processing level {level}")
        
        embeddings = embed_texts(current_texts)
        n_clusters = min(10, len(current_texts) // 2)
        cluster_labels = perform_clustering(np.array(embeddings), n_clusters)
        
        df = pd.DataFrame({
            'text': current_texts,
            'embedding': embeddings,
            'cluster': cluster_labels,
            'metadata': current_metadata
        })
        
        results[level-1] = df
        
        summaries = []
        new_metadata = []
        for cluster in df['cluster'].unique():
            cluster_docs = df[df['cluster'] == cluster]
            cluster_texts = cluster_docs['text'].tolist()
            cluster_metadata = cluster_docs['metadata'].tolist()
            summary = summarize_texts(cluster_texts)
            summaries.append(summary)
            new_metadata.append({
                "level": level,
                "origin": f"summary_of_cluster_{cluster}_level_{level-1}",
                "child_ids": [meta.get('id') for meta in cluster_metadata],
                "id": f"summary_{level}_{cluster}"
            })
        
        current_texts = summaries
        current_metadata = new_metadata
        
        if len(current_texts) <= 1:
            results[level] = pd.DataFrame({
                'text': current_texts,
                'embedding': embed_texts(current_texts),
                'cluster': [0],
                'metadata': current_metadata
            })
            logging.info(f"Stopping at level {level} as we have only one summary")
            break
    
    return results
```

### 向量存储函数


```python
def build_vectorstore(tree_results: Dict[int, pd.DataFrame]) -> FAISS:    """从 RAPTOR 树中的所有文本构建 FAISS 向量存储。"""    all_texts = []    all_embeddings = []    all_metadatas = []        for level, df in tree_results.items():        all_texts.extend([str(text) for text in df['text'].tolist()])        all_embeddings.extend([embedding.tolist() if isinstance(embedding, np.ndarray) else embedding for embedding in df['embedding'].tolist()])        all_metadatas.extend(df['metadata'].tolist())        logging.info(f"Building vectorstore with {len(all_texts)} texts")        # 手动创建 Document 对象以确保正确的类型    documents = [Document(page_content=str(text), metadata=metadata)                  for text, metadata in zip(all_texts, all_metadatas)]        return FAISS.from_documents(documents, embeddings)
```

### 定义树遍历检索

```python
def tree_traversal_retrieval(query: str, vectorstore: FAISS, k: int = 3) -> List[Document]:
    """执行树遍历检索。"""
    query_embedding = embeddings.embed_query(query)
    
    def retrieve_level(level: int, parent_ids: List[str] = None) -> List[Document]:
        if parent_ids:
            docs = vectorstore.similarity_search_by_vector_with_relevance_scores(
                query_embedding,
                k=k,
                filter=lambda meta: meta['level'] == level and meta['id'] in parent_ids
            )
        else:
            docs = vectorstore.similarity_search_by_vector_with_relevance_scores(
                query_embedding,
                k=k,
                filter=lambda meta: meta['level'] == level
            )
        
        if not docs or level == 0:
            return docs
        
        child_ids = [doc.metadata.get('child_ids', []) for doc, _ in docs]
        child_ids = [item for sublist in child_ids for item in sublist]  # 展平列表
        
        child_docs = retrieve_level(level - 1, child_ids)
        return docs + child_docs
    
    max_level = max(doc.metadata['level'] for doc in vectorstore.docstore.values())
    return retrieve_level(max_level)
```

### 创建检索器


```python
def create_retriever(vectorstore: FAISS) -> ContextualCompressionRetriever:
    """创建带有上下文压缩的检索器。"""
    logging.info("Creating contextual compression retriever")
    base_retriever = vectorstore.as_retriever()
    
    prompt = ChatPromptTemplate.from_template(
        "给定以下上下文和问题，仅提取与回答问题相关的信息：\n\n"
        "上下文：{context}\n"
        "问题：{question}\n\n"
        "相关信息："
    )
    
    extractor = LLMChainExtractor.from_llm(llm, prompt=prompt)
    
    return ContextualCompressionRetriever(
        base_compressor=extractor,
        base_retriever=base_retriever
    )

```

### 定义层次化检索

```python
def hierarchical_retrieval(query: str, retriever: ContextualCompressionRetriever, max_level: int) -> List[Document]:    """从最高层级开始执行层次化检索，处理可能的 None 值。"""    all_retrieved_docs = []        for level in range(max_level, -1, -1):        # 从当前层级检索文档        level_docs = retriever.get_relevant_documents(            query,            filter=lambda meta: meta['level'] == level        )        all_retrieved_docs.extend(level_docs)                # 如果已找到文档，从下一层级检索其子文档        if level_docs and level > 0:            child_ids = [doc.metadata.get('child_ids', []) for doc in level_docs]            child_ids = [item for sublist in child_ids for item in sublist if item is not None]  # 展平并过滤 None                        if child_ids:  # Only modify query if there are valid child IDs                child_query = f" AND id:({' OR '.join(str(id) for id in child_ids)})"                query += child_query        return all_retrieved_docs
```

### RAPTOR 查询流程（在线流程）

```python
def raptor_query(query: str, retriever: ContextualCompressionRetriever, max_level: int) -> Dict[str, Any]:
    """使用 RAPTOR 系统和层次化检索处理查询。"""
    logging.info(f"Processing query: {query}")
    
    relevant_docs = hierarchical_retrieval(query, retriever, max_level)
    
    doc_details = []
    for i, doc in enumerate(relevant_docs, 1):
        doc_details.append({
            "index": i,
            "content": doc.page_content,
            "metadata": doc.metadata,
            "level": doc.metadata.get('level', 'Unknown'),
            "similarity_score": doc.metadata.get('score', 'N/A')
        })
    
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    prompt = ChatPromptTemplate.from_template(
        "给定以下上下文，请回答问题：\n\n"
        "上下文：{context}\n\n"
        "问题：{question}\n\n"
        "答案："
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    answer = chain.run(context=context, question=query)
    
    logging.info("Query processing completed")
    
    result = {
        "query": query,
        "retrieved_documents": doc_details,
        "num_docs_retrieved": len(relevant_docs),
        "context_used": context,
        "answer": answer,
        "model_used": llm.model_name,
    }
    
    return result


def print_query_details(result: Dict[str, Any]):
    """打印查询过程的详细信息，包括树层级元数据。"""
    print(f"Query: {result['query']}")
    print(f"\nNumber of documents retrieved: {result['num_docs_retrieved']}")
    print(f"\nRetrieved Documents:")
    for doc in result['retrieved_documents']:
        print(f"  Document {doc['index']}:")
        print(f"    Content: {doc['content'][:100]}...")  # 显示前 100 个字符
        print(f"    Similarity Score: {doc['similarity_score']}")
        print(f"    Tree Level: {doc['metadata'].get('level', 'Unknown')}")
        print(f"    Origin: {doc['metadata'].get('origin', 'Unknown')}")
        if 'child_docs' in doc['metadata']:
            print(f"    Number of Child Documents: {len(doc['metadata']['child_docs'])}")
        print()
    
    print(f"\nContext used for answer generation:")
    print(result['context_used'])
    
    print(f"\nGenerated Answer:")
    print(result['answer'])
    
    print(f"\nModel Used: {result['model_used']}")

```

## 示例使用与可视化


## 定义数据文件夹

```python
# 下载所需的数据文件import osos.makedirs('data', exist_ok=True)# 下载本 notebook 中使用的 PDF 文档!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
```

```python
path = "data/Understanding_Climate_Change.pdf"
```

### 处理文本

```python
loader = PyPDFLoader(path)
documents = loader.load()
texts = [doc.page_content for doc in documents]
```

### 创建 RAPTOR 组件实例

```python
# 构建 RAPTOR 树tree_results = build_raptor_tree(texts)
```

```python
# 构建向量存储vectorstore = build_vectorstore(tree_results)
```

```python
# 创建检索器retriever = create_retriever(vectorstore)
```

### 运行查询并观察数据来源与结果

```python
# 运行流水线max_level = 3  # 根据您的树深度调整query = "温室效应是什么？"result = raptor_query(query, retriever, max_level)print_query_details(result)
```

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--raptor)
