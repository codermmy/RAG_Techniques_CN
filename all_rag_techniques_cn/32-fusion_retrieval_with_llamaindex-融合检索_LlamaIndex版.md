# 文档搜索中的融合检索

## 概述

本代码实现了一个融合检索系统，它将基于向量的相似性搜索与基于关键词的BM25检索相结合。该方法旨在利用两种方法的优势，提高文档检索的整体质量和相关性。

## 动机

传统的检索方法通常依赖于语义理解（基于向量）或关键词匹配（BM25）。每种方法都有其优缺点。融合检索旨在结合这些方法，创建一个更健壮、更准确的检索系统，能够有效处理更广泛的查询类型。

## 关键组件

1. PDF处理和文本分块
2. 使用FAISS和OpenAI Embedding创建向量存储
3. 创建BM25索引用于基于关键词的检索
4. 融合BM25和向量搜索结果以获得更好的检索效果

## 方法详情

### 文档预处理

1. 使用SentenceSplitter加载PDF并分割成文本块。
2. 通过将't'替换为空格和清理换行符来清理文本块（可能是为了解决特定的格式问题）。

### 向量存储创建

1. 使用OpenAI Embedding创建文本块的向量表示。
2. 从这些Embedding创建FAISS向量存储以进行高效的相似性搜索。

### BM25索引创建

1. 使用与向量存储相同的文本块创建BM25索引。
2. 这允许在基于向量的方法之外进行基于关键词的检索。

### 查询融合检索

创建两个索引后，查询融合检索将它们结合起来，实现混合检索。

## 此方法的优势

1. **提高检索质量**：通过结合语义搜索和关键词搜索，系统可以同时捕捉概念相似性和精确的关键词匹配。
2. **灵活性**：`retriever_weights`参数允许根据具体用例或查询类型调整向量和关键词搜索之间的平衡。
3. **健壮性**：组合方法可以有效处理更广泛的查询类型，减轻单个方法的弱点。
4. **可定制性**：系统可以轻松适配不同的向量存储或基于关键词的检索方法。

## 结论

融合检索代表了一种强大的文档搜索方法，结合了语义理解和关键词匹配的优势。通过利用基于向量和BM25检索方法，它为信息检索任务提供了更全面、更灵活的解决方案。该方法在概念相似性和关键词相关性都很重要的各个领域都有潜在应用，如学术研究、法律文档搜索或通用搜索引擎。

# 包安装和导入

下面的单元格安装运行此notebook所需的所有必要包。

```python
# 安装所需的包!pip install faiss-cpu llama-index python-dotenv
```

```python
import osimport sysfrom dotenv import load_dotenvfrom typing import Listfrom llama_index.core import Settingsfrom llama_index.core.readers import SimpleDirectoryReaderfrom llama_index.core.node_parser import SentenceSplitterfrom llama_index.core.ingestion import IngestionPipelinefrom llama_index.core.schema import BaseNode, TransformComponentfrom llama_index.vector_stores.faiss import FaissVectorStorefrom llama_index.core import VectorStoreIndexfrom llama_index.llms.openai import OpenAIfrom llama_index.embeddings.openai import OpenAIEmbeddingfrom llama_index.legacy.retrievers.bm25_retriever import BM25Retrieverfrom llama_index.core.retrievers import QueryFusionRetrieverimport faiss# 原始路径追加已替换为 Colab 兼容# 从 .env 文件加载环境变量load_dotenv()# 设置 OpenAI API 密钥环境变量os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')# Llamaindex 全局设置用于 LLM 和嵌入EMBED_DIMENSION=512Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=EMBED_DIMENSION)
```

### 读取文档

```python
# 下载所需的数据文件import osos.makedirs('data', exist_ok=True)# 下载本 notebook 中使用的 PDF 文档!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
```

```python
path = "data/"
reader = SimpleDirectoryReader(input_dir=path, required_exts=['.pdf'])
documents = reader.load_data()
print(documents[0])
```

### 创建向量存储

```python
# 创建 FaisVectorStore 以存储嵌入fais_index = faiss.IndexFlatL2(EMBED_DIMENSION)vector_store = FaissVectorStore(faiss_index=fais_index)
```

### 文本清理转换器

```python
class TextCleaner(TransformComponent):
    """
    用于摄取管道中的转换。
    清理文本中的杂乱内容。
    """
    def __call__(self, nodes, **kwargs) -> List[BaseNode]:
        
        for node in nodes:
            node.text = node.text.replace('\t', ' ') # 将制表符替换为空格
            node.text = node.text.replace(' \n', ' ') # 将段落分隔符替换为空格
            
        return nodes
```

### 数据摄取管道

```python
# 流水线实例化包含： # 节点解析器、自定义转换器、向量存储和文档pipeline = IngestionPipeline(    transformations=[        SentenceSplitter(),        TextCleaner()    ],    vector_store=vector_store,    documents=documents)# 运行流水线以获取节点nodes = pipeline.run()
```

## 检索器

### BM25检索器

```python
bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=2,
)
```

### 向量检索器

```python
index = VectorStoreIndex(nodes)
vector_retriever = index.as_retriever(similarity_top_k=2)
```

### 融合两个检索器

```python
retriever = QueryFusionRetriever(
    retrievers=[
        vector_retriever,
        bm25_retriever
    ],
    retriever_weights=[
        0.6, # vector retriever weight
        0.4 # BM25 retriever weight
    ],
    num_queries=1, 
    mode='dist_based_score',
    use_async=False
)
```

关于参数说明

1. `num_queries`：查询融合检索器不仅可以组合检索器，还可以从给定查询生成多个问题。此参数控制将传递给检索器的总查询数。因此，将其设置为1会禁用查询生成，最终检索器只使用初始查询。
2. `mode`：此参数有4个选项。
   - **reciprocal_rerank**：应用倒数排序。（由于没有归一化，此方法不适合这种应用。因为不同的检索器会返回不同的分数范围）
   - **relative_score**：基于所有节点中的最小和最大分数应用MinMax归一化。然后缩放到0到1之间。最后根据`retriever_weights`按相对检索器权重加权。
      ```math
      min\_score = min(scores)
      \\ max\_score = max(scores)
      ```
   - **dist_based_score**：与`relative_score`唯一的区别是MinMax缩放基于分数的均值和标准差。缩放和加权方式相同。
      ```math
       min\_score = mean\_score - 3 * std\_dev
      \\ max\_score = mean\_score + 3 * std\_dev
      ```
   - **simple**：此方法简单地取每个文本块的最大分数。

### 使用示例

```python
# 查询query = "What are the impacts of climate change on the environment?"# 执行融合检索response = retriever.retrieve(query)
```

### 打印最终检索的节点及分数

```python
for node in response:
    print(f"节点分数：{node.score:.2}")
    print(f"节点内容：{node.text}")
    print("-"*100)
```

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--fusion-retrieval-with-llamaindex)
