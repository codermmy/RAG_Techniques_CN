# 文档检索的上下文增强窗口

## 概述

本代码实现了向量数据库中文档检索的上下文增强窗口技术。它通过为每个检索到的文本块添加周围上下文来增强标准检索过程，从而提高返回信息的连贯性和完整性。

## 动机

传统的向量搜索通常返回孤立的文本块，这些文本块可能缺乏完整理解所需的上下文。本方法旨在通过包含相邻的文本块，为检索到的信息提供更全面的视角。

## 关键组件

1. PDF处理和文本分块
2. 使用FAISS和OpenAI Embedding创建向量存储
3. 带上下文窗口的自定义检索函数
4. 标准检索与上下文增强检索的对比

## 方法详情

### 文档预处理

1. 读取PDF并将其转换为字符串。
2. 将文本分割为带有周围句子的文本块。

### 向量存储创建

1. 使用OpenAI Embedding创建文本块的向量表示。
2. 从这些Embedding创建FAISS向量存储。

### 上下文增强检索

LlamaIndex有一个专门用于此任务的特殊解析器。[SentenceWindowNodeParser](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/#sentencewindownodeparser) 这个解析器将文档分割成句子。但生成的节点包含带有关系结构的周围句子。然后，在查询时，[MetadataReplacementPostProcessor](https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors/#metadatareplacementpostprocessor) 帮助重新连接这些相关句子。

### 检索对比

本notebook包含一个对比标准检索与上下文增强方法的章节。

## 此方法的优势

1. 提供更连贯、上下文更丰富的结果
2. 在保持向量搜索优势的同时，减轻其返回孤立文本片段的倾向
3. 允许灵活调整上下文窗口大小

## 结论

这种上下文增强窗口技术为提高基于向量的文档搜索系统中检索信息的质量提供了一种有前景的方法。通过提供周围上下文，它有助于保持检索信息的连贯性和完整性，从而可能在问答等下游任务中带来更好的理解和更准确的回答。

<div style="text-align: center;">

<img src="../images/vector-search-comparison_context_enrichment.svg" alt="上下文增强窗口" style="width:70%; height:auto;">
</div>

# 包安装和导入

下面的单元格安装运行此notebook所需的所有必要包。

```python
# 安装所需的包!pip install faiss-cpu llama-index python-dotenv
```

```python
from llama_index.core import Settingsfrom llama_index.llms.openai import OpenAIfrom llama_index.embeddings.openai import OpenAIEmbeddingfrom llama_index.core.readers import SimpleDirectoryReaderfrom llama_index.vector_stores.faiss import FaissVectorStorefrom llama_index.core.ingestion import IngestionPipelinefrom llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitterfrom llama_index.core import VectorStoreIndexfrom llama_index.core.postprocessor import MetadataReplacementPostProcessorimport faissimport osimport sysfrom dotenv import load_dotenvfrom pprint import pprint# 原始路径追加已替换为 Colab 兼容# 从 .env 文件加载环境变量load_dotenv()# 设置 OpenAI API 密钥环境变量os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')# Llamaindex 全局设置用于 LLM 和嵌入EMBED_DIMENSION=512Settings.llm = OpenAI(model="gpt-3.5-turbo")Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=EMBED_DIMENSION)
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

### 创建向量存储和检索器

```python
# 创建 FaisVectorStore 以存储嵌入fais_index = faiss.IndexFlatL2(EMBED_DIMENSION)vector_store = FaissVectorStore(faiss_index=fais_index)
```

## 数据摄取管道

### 使用句子分割器的数据摄取管道

```python
base_pipeline = IngestionPipeline(
    transformations=[SentenceSplitter()],
    vector_store=vector_store
)

base_nodes = base_pipeline.run(documents=documents)
```

### 使用句子窗口的数据摄取管道

```python
node_parser = SentenceWindowNodeParser(    # 要捕获的两侧句子数量。     # 设置为 3 会得到 7 个句子。    window_size=3,    # 用于 MetadataReplacementPostProcessor 的元数据键    window_metadata_key="window",    # 存储原始句子的元数据键    original_text_metadata_key="original_sentence")# 使用定义的文档转换和向量存储创建流水线pipeline = IngestionPipeline(    transformations=[node_parser],    vector_store=vector_store,)windowed_nodes = pipeline.run(documents=documents)
```

## 查询

```python
query = "Explain the role of deforestation and fossil fuels in climate change"
```

### 不使用元数据替换的查询

```python
# 从基础节点创建向量索引base_index = VectorStoreIndex(base_nodes)# 从向量索引实例化查询引擎base_query_engine = base_index.as_query_engine(    similarity_top_k=1,)# 向引擎发送查询以获取相关节点base_response = base_query_engine.query(query)print(base_response)
```

#### 打印检索节点的元数据

```python
pprint(base_response.source_nodes[0].node.metadata)
```

### 使用元数据替换的查询
"元数据替换"直观上可能听起来有点离题，因为我们是在基础句子上工作的。但LlamaIndex将这些"前/后句子"存储在节点的元数据中。因此，要重建这些句子窗口，我们需要元数据替换后处理器。

```python
# 从 SentenceWindowNodeParser 创建的节点创建窗口索引windowed_index = VectorStoreIndex(windowed_nodes)# 使用 MetadataReplacementPostProcessor 实例化查询引擎windowed_query_engine = windowed_index.as_query_engine(    similarity_top_k=1,    node_postprocessors=[        MetadataReplacementPostProcessor(            target_metadata_key="window" # `window_metadata_key` key defined in SentenceWindowNodeParser            )        ],)# 向引擎发送查询以获取相关节点windowed_response = windowed_query_engine.query(query)print(windowed_response)
```

#### 打印检索节点的元数据

```python
# 窗口和原始句子被添加到元数据中pprint(windowed_response.source_nodes[0].node.metadata)
```

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--context-enrichment-window-around-chunk-with-llamaindex)
