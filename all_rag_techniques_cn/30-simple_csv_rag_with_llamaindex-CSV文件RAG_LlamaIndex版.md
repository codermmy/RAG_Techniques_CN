# CSV 文件基础 RAG (检索增强生成) 系统

## 概述

本代码实现了一个基础的检索增强生成 (RAG) 系统，用于处理和查询 CSV 文档。系统将文档内容编码到向量存储中，然后可以通过查询检索相关信息。

# CSV 文件结构与用例
该 CSV 文件包含虚拟客户数据，包括名字、姓氏、公司等各种属性。该数据集将用于 RAG 用例，便于创建客户信息问答系统。

## 核心组件

1. 加载和分割 CSV 文件。
2. 使用 [FAISS](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) 和 OpenAI Embeddings 创建向量存储
3. 查询引擎设置用于查询处理后的文档
4. 基于 CSV 数据创建问答功能。

## 方法细节

### 文档预处理

1. 使用 LlamaIndex 的 [PagedCSVReader](https://docs.llamaindex.ai/en/stable/api_reference/readers/file/#llama_index.readers.file.PagedCSVReader) 加载 CSV 文件
2. 该读取器将每一行连同表的相应列名转换为 LlamaIndex Document。不需要进一步分割。


### 向量存储创建

1. 使用 OpenAI Embeddings 创建文本块的向量表示。
2. 从这些 Embeddings 创建 FAISS 向量存储以进行高效的相似度搜索。

### 查询引擎设置

1. 配置查询引擎为给定查询获取最相关的块，然后回答问题。

## 方法优势

1. 可扩展性: 通过分块处理可以处理大型文档。
2. 灵活性: 易于调整块大小和检索结果数量等参数。
3. 高效性: 利用 FAISS 在高维空间中进行快速相似度搜索。
4. 与先进 NLP 集成: 使用 OpenAI Embeddings 进行最先进的文本表示。

## 结论

这个基础 RAG 系统为构建更复杂的信息检索和问答系统提供了坚实的基础。通过将文档内容编码到可搜索的向量存储中，它能够高效地检索响应查询的相关信息。这种方法特别适用于需要在 CSV 文件中快速访问特定信息的应用场景。

# 包安装与导入

下面的单元格安装运行此 notebook 所需的所有必要包。


```python
# 安装所需的包!pip install faiss-cpu llama-index pandas python-dotenv
```

```python
from llama_index.core.readers import SimpleDirectoryReaderfrom llama_index.core import Settingsfrom llama_index.llms.openai import OpenAIfrom llama_index.embeddings.openai import OpenAIEmbeddingfrom llama_index.readers.file import PagedCSVReaderfrom llama_index.vector_stores.faiss import FaissVectorStorefrom llama_index.core.ingestion import IngestionPipelinefrom llama_index.core import VectorStoreIndeximport faissimport osimport pandas as pdfrom dotenv import load_dotenv# 从 .env 文件加载环境变量load_dotenv()# 设置 OpenAI API 密钥环境变量os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')# Llamaindex 全局设置用于 LLM 和嵌入EMBED_DIMENSION=512Settings.llm = OpenAI(model="gpt-3.5-turbo")Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=EMBED_DIMENSION)
```

### CSV 文件结构与用例
该 CSV 文件包含虚拟客户数据，包括名字、姓氏、公司等各种属性。该数据集将用于 RAG 用例，便于创建客户信息问答系统。

```python
# 下载所需的数据文件import osos.makedirs('data', exist_ok=True)# 下载本 notebook 中使用的 PDF 文档!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf!wget -O data/customers-100.csv https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/customers-100.csv
```

```python
file_path = ('data/customers-100.csv') # insert the path of the csv filedata = pd.read_csv(file_path)# 预览 CSV 文件data.head()
```

### 向量存储

```python
# 创建 FaisVectorStore 以存储嵌入fais_index = faiss.IndexFlatL2(EMBED_DIMENSION)vector_store = FaissVectorStore(faiss_index=fais_index)
```

### 加载和处理 CSV 数据作为文档

```python
csv_reader = PagedCSVReader()

reader = SimpleDirectoryReader( 
    input_files=[file_path],
    file_extractor= {".csv": csv_reader}
    )

docs = reader.load_data()
```

```python
# 检查示例分块print(docs[0].text)
```

### 数据摄入流水线

```python
pipeline = IngestionPipeline(
    vector_store=vector_store,
    documents=docs
)

nodes = pipeline.run()
```

### 创建查询引擎

```python
vector_store_index = VectorStoreIndex(nodes)
query_engine = vector_store_index.as_query_engine(similarity_top_k=2)
```

### 基于 CSV 数据向 RAG 机器人提问

```python
response = query_engine.query("which company does sheryl Baxter work for?")
response.response
```

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--simple-csv-rag-with-llamaindex)
