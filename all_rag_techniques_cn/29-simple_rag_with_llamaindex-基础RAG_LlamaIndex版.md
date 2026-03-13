# 基础 RAG (检索增强生成) 系统

## 概述

本代码实现了一个基础的检索增强生成 (RAG) 系统，用于处理和查询 PDF 文档。系统使用流水线对文档进行编码并创建节点。这些节点随后可用于构建向量索引以检索相关信息。

## 核心组件

1. PDF 处理和文本提取
2. 文本分块以便于管理处理
3. 使用 FAISS 作为向量存储和 OpenAI Embeddings 创建数据摄入流水线
4. 检索器设置用于查询处理后的文档
5. RAG 系统评估

## 方法细节

### 文档预处理

1. 使用 [SimpleDirectoryReader](https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/) 加载 PDF。
2. 使用 [SentenceSplitter](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/#sentencesplitter) 将文本分割成[节点/块](https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/)，可指定块大小和重叠大小。

### 文本清洗

应用自定义转换 `TextCleaner` 来清洗文本。这可能会处理 PDF 中的特定格式问题。

### 数据摄入流水线创建

1. 使用 OpenAI Embeddings 创建文本节点的向量表示。
2. 从这些 Embeddings 创建 FAISS 向量存储以进行高效的相似度搜索。

### 检索器设置

1. 配置检索器为给定查询获取前 2 个最相关的块。


## 主要特性

1. 模块化设计: 数据摄入过程封装在单个函数中，便于重用。
2. 可配置分块: 允许调整块大小和重叠大小。
3. 高效检索: 使用 FAISS 进行快速相似度搜索。
4. 评估: 包含评估 RAG 系统性能的函数。

## 使用示例

代码包含一个测试查询: "气候变化的主要原因是什么？"。这演示了如何使用检索器从处理后的文档中获取相关上下文。

## 评估

系统包含 `evaluate_rag` 函数来评估检索器的性能，虽然提供的代码中未详细说明使用的具体指标。

## 方法优势

1. 可扩展性: 通过分块处理可以处理大型文档。
2. 灵活性: 易于调整块大小和检索结果数量等参数。
3. 高效性: 利用 FAISS 在高维空间中进行快速相似度搜索。
4. 与先进 NLP 集成: 使用 OpenAI Embeddings 进行最先进的文本表示。

## 结论

这个基础 RAG 系统为构建更复杂的信息检索和问答系统提供了坚实的基础。通过将文档内容编码到可搜索的向量存储中，它能够高效地检索响应查询的相关信息。这种方法特别适用于需要在大型文档或文档集合中快速访问特定信息的应用场景。

# 包安装与导入

下面的单元格安装运行此 notebook 所需的所有必要包。


```python
# 安装所需的包!pip install faiss-cpu llama-index python-dotenv
```

```python
# 克隆仓库以访问辅助函数和评估模块!git clone https://github.com/NirDiamant/RAG_TECHNIQUES.gitimport syssys.path.append('RAG_TECHNIQUES')# 如果需要运行最新数据# !cp -r RAG_TECHNIQUES/data .
```

```python
from typing import Listfrom llama_index.core import SimpleDirectoryReader, VectorStoreIndexfrom llama_index.core.ingestion import IngestionPipelinefrom llama_index.core.schema import BaseNode, TransformComponentfrom llama_index.vector_stores.faiss import FaissVectorStorefrom llama_index.core.text_splitter import SentenceSplitterfrom llama_index.embeddings.openai import OpenAIEmbeddingfrom llama_index.core import Settingsimport faissimport osimport sysfrom dotenv import load_dotenv# 原始路径追加已替换为 Colab 兼容EMBED_DIMENSION = 512# 分块设置与 langchain 示例非常不同# 因为对于分块长度，langchain 使用字符串长度，# 而 llamaindex 使用 token 长度CHUNK_SIZE = 200CHUNK_OVERLAP = 50# 从 .env 文件加载环境变量load_dotenv()# 设置 OpenAI API 密钥环境变量os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')# 在 LlamaIndex 全局设置中设置嵌入模型Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=EMBED_DIMENSION)
```

### 读取文档

```python
# 下载所需的数据文件import osos.makedirs('data', exist_ok=True)# 下载本 notebook 中使用的 PDF 文档!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf!wget -O data/q_a.json https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/q_a.json
```

```python
path = "data/"
node_parser = SimpleDirectoryReader(input_dir=path, required_exts=['.pdf'])
documents = node_parser.load_data()
print(documents[0])
```

### 向量存储

```python
# 创建 FaisVectorStore 以存储嵌入faiss_index = faiss.IndexFlatL2(EMBED_DIMENSION)vector_store = FaissVectorStore(faiss_index=faiss_index)
```

### 文本清洗转换

```python
class TextCleaner(TransformComponent):
    """
    用于摄取管道中的转换。
    清理文本中的杂乱内容。
    """
    def __call__(self, nodes, **kwargs) -> List[BaseNode]:
        
        for node in nodes:
            node.text = node.text.replace('\t', ' ') # 将制表符替换为空格
            node.text = node.text.replace(' \n', ' ') # Replace paragraph seperator with spacaes
            
        return nodes
```

### 数据摄入流水线

```python
text_splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)# 使用定义的文档转换和向量存储创建流水线pipeline = IngestionPipeline(    transformations=[        TextCleaner(),        text_splitter,    ],    vector_store=vector_store, )
```

```python
# 运行流水线并从过程中获取生成的节点nodes = pipeline.run(documents=documents)
```

### 创建检索器

```python
vector_store_index = VectorStoreIndex(nodes)
retriever = vector_store_index.as_retriever(similarity_top_k=2)
```

### 测试检索器

```python
def show_context(context):
    """
    显示提供的上下文列表的内容。

    Args:
        context (list): A list of context items to be displayed.

    打印列表中的每个上下文项，并带有指示其位置的标题。
    """
    for i, c in enumerate(context):
        print(f"Context {i+1}:")
        print(c.text)
        print("\n")
```

```python
test_query = "气候变化的主要原因是什么？"
context = retriever.retrieve(test_query)
show_context(context)
```

### 让我们看看它的表现如何:

```python
import jsonfrom deepeval import evaluatefrom deepeval.metrics import GEval, FaithfulnessMetric, ContextualRelevancyMetricfrom deepeval.test_case import LLMTestCaseParamsfrom evaluation.evalute_rag import create_deep_eval_test_cases# 设置用于评估问题和答案的 LLM 模型 LLM_MODEL = "gpt-4o"# 定义评估指标correctness_metric = GEval(    name="Correctness",    model=LLM_MODEL,    evaluation_params=[        LLMTestCaseParams.EXPECTED_OUTPUT,        LLMTestCaseParams.ACTUAL_OUTPUT    ],    evaluation_steps=[        "根据期望输出确定实际输出在事实上是否正确。"    ],)faithfulness_metric = FaithfulnessMetric(    threshold=0.7,    model=LLM_MODEL,    include_reason=False)relevance_metric = ContextualRelevancyMetric(    threshold=1,    model=LLM_MODEL,    include_reason=True)def evaluate_rag(query_engine, num_questions: int = 5) -> None:    """    Evaluate the RAG system using predefined metrics.    Args:        query_engine: Query engine to ask questions and get answers along with retrieved context.        num_questions (int): Number of questions to evaluate (default: 5).    """            # 从 JSON 文件加载问题和答案    q_a_file_name = "data/q_a.json"    with open(q_a_file_name, "r", encoding="utf-8") as json_file:        q_a = json.load(json_file)    questions = [qa["question"] for qa in q_a][:num_questions]    ground_truth_answers = [qa["answer"] for qa in q_a][:num_questions]    generated_answers = []    retrieved_documents = []    # 为每个问题生成答案并检索文档    for question in questions:        response = query_engine.query(question)        context = [doc.text for doc in response.source_nodes]        retrieved_documents.append(context)        generated_answers.append(response.response)    # Create test cases and evaluate    test_cases = create_deep_eval_test_cases(questions, ground_truth_answers, generated_answers, retrieved_documents)    evaluate(        test_cases=test_cases,        metrics=[correctness_metric, faithfulness_metric, relevance_metric]    )
```

### 评估结果

```python
query_engine  = vector_store_index.as_query_engine(similarity_top_k=2)
evaluate_rag(query_engine, num_questions=1)
```

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--simple-rag-with-llamaindex)
