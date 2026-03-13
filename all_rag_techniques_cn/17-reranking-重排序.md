# RAG 系统中的重排序方法

## 概述
重排序是检索增强生成（RAG）系统中的关键步骤，旨在提高检索文档的相关性和质量。它涉及重新评估和重新排序最初检索的文档，以确保最相关的信息在后续处理或展示中被优先考虑。

## 动机
RAG 系统中进行重排序的主要动机是克服初始检索方法的局限性，这些方法通常依赖于较简单的相似性度量。重排序允许进行更复杂的相关性评估，考虑到查询与文档之间可能被传统检索技术忽略的细微关系。此过程旨在确保在生成阶段使用最相关的信息，从而提高 RAG 系统的整体性能。

## 核心组件
重排序系统通常包括以下组件：

1. **初始检索器**：通常使用基于 Embedding 相似性搜索的向量存储。
2. **重排序模型**：可以是以下之一：
   - 用于评分相关性大型语言模型（LLM）
   - 专门为相关性评估训练的 Cross-Encoder 模型
3. **评分机制**：为文档分配相关性得分的方法
4. **排序和选择逻辑**：根据新得分重新排序文档

## 方法细节
重排序过程通常遵循以下步骤：

1. **初始检索**：获取一组初始的潜在相关文档。
2. **配对创建**：为每个检索到的文档形成查询 - 文档配对。
3. **评分**：
   - LLM 方法：使用提示词让 LLM 评估文档相关性。
   - Cross-Encoder 方法：将查询 - 文档配对直接输入模型。
4. **分数解释**：解析并归一化相关性得分。
5. **重新排序**：根据新的相关性得分对文档进行排序。
6. **选择**：从重新排序的列表中选择前 K 个文档。

## 该方法的优势
重排序提供了以下优势：

1. **提高相关性**：通过使用更复杂的模型，重排序可以捕获细微的相关性因素。
2. **灵活性**：可以根据具体需求和资源应用不同的重排序方法。
3. **增强上下文质量**：向 RAG 系统提供更相关的文档可提高生成响应的质量。
4. **减少噪声**：重排序有助于过滤掉不太相关的信息，聚焦于最相关的内容。

## 结论
重排序是 RAG 系统中一项强大的技术，可显著提高检索信息的质量。无论是使用基于 LLM 的评分还是专门的 Cross-Encoder 模型，重排序都能对文档相关性进行更细致、更准确的评估。这种改进的相关性直接转化为下游任务的更好性能，使重排序成为高级 RAG 实现中的重要组件。

在基于 LLM 和 Cross-Encoder 重排序方法之间的选择取决于所需精度、可用计算资源和具体应用需求等因素。这两种方法都比基本检索方法提供了显著改进，并有助于 RAG 系统的整体效果。

<div style="text-align: center;">

<img src="../images/reranking-visualization.svg" alt="rerank llm" style="width:100%; height:auto;">
</div>

<div style="text-align: center;">

<img src="../images/reranking_comparison.svg" alt="rerank llm" style="width:100%; height:auto;">
</div>

# 包安装和导入

下面的单元格安装了运行此 notebook 所需的所有包。

```python
# 安装所需的包
!pip install langchain langchain-openai python-dotenv sentence-transformers
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
from typing import List, Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import CrossEncoder


# 原始路径追加已替换为 Colab 兼容性
from helper_functions import *
from evaluation.evalute_rag import *

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

### 创建向量存储

```python
vectorstore = encode_pdf(path)
```

## 方法 1：基于 LLM 的函数对检索到的文档进行重排序

<div style="text-align: center;">

<img src="../images/rerank_llm.svg" alt="rerank llm" style="width:40%; height:auto;">
</div>

### 创建自定义重排序函数

```python
class RatingScore(BaseModel):
    relevance_score: float = Field(..., description="文档对查询的相关性得分。")

def rerank_documents(query: str, docs: List[Document], top_n: int = 3) -> List[Document]:
    prompt_template = PromptTemplate(
        input_variables=["query", "doc"],
        template="""在 1-10 的评分标准上，评估以下文档与查询的相关性。考虑查询的具体上下文和意图，不仅仅是关键词匹配。
        查询：{query}
        文档：{doc}
        相关性得分："""
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
    llm_chain = prompt_template | llm.with_structured_output(RatingScore)

    scored_docs = []
    for doc in docs:
        input_data = {"query": query, "doc": doc.page_content}
        score = llm_chain.invoke(input_data).relevance_score
        try:
            score = float(score)
        except ValueError:
            score = 0  # 解析失败时的默认得分
        scored_docs.append((doc, score))

    reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked_docs[:top_n]]
```

### 重排序函数的使用示例，使用与文档相关的示例查询

```python
query = "What are the impacts of climate change on biodiversity?"
initial_docs = vectorstore.similarity_search(query, k=15)
reranked_docs = rerank_documents(query, initial_docs)

# 打印前 3 个初始文档
print("初始前 3 个文档:")
for i, doc in enumerate(initial_docs[:3]):
    print(f"\n文档 {i+1}:")
    print(doc.page_content[:200] + "...")  # 打印每个文档的前 200 个字符


# 打印结果
print(f"查询：{query}\n")
print("重排序后的前 3 个文档:")
for i, doc in enumerate(reranked_docs):
    print(f"\n文档 {i+1}:")
    print(doc.page_content[:200] + "...")  # 打印每个文档的前 200 个字符
```

### 基于我们的重排序器创建自定义检索器

```python
# 创建自定义检索器类
class CustomRetriever(BaseRetriever, BaseModel):

    vectorstore: Any = Field(description="用于初始检索的向量存储")

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str, num_docs=2) -> List[Document]:
        initial_docs = self.vectorstore.similarity_search(query, k=30)
        return rerank_documents(query, initial_docs, top_n=num_docs)


# 创建自定义检索器
custom_retriever = CustomRetriever(vectorstore=vectorstore)

# 创建用于回答问题的 LLM
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

# 创建带有自定义检索器的 RetrievalQA 链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=custom_retriever,
    return_source_documents=True
)
```

### 示例查询

```python
result = qa_chain({"query": query})

print(f"\n问题：{query}")
print(f"答案：{result['result']}")
print("\n相关源文档:")
for i, doc in enumerate(result["source_documents"]):
    print(f"\n文档 {i+1}:")
    print(doc.page_content[:200] + "...")  # 打印每个文档的前 200 个字符
```

### 展示为什么应该使用重排序的示例

```python
chunks = [
    "The capital of France is great.",
    "The capital of France is huge.",
    "The capital of France is beautiful.",
    """Have you ever visited Paris? It is a beautiful city where you can eat delicious food and see the Eiffel Tower.
    I really enjoyed all the cities in france, but its capital with the Eiffel Tower is my favorite city.""",
    "I really enjoyed my trip to Paris, France. The city is beautiful and the food is delicious. I would love to visit again. Such a great capital city."
]
docs = [Document(page_content=sentence) for sentence in chunks]


def compare_rag_techniques(query: str, docs: List[Document] = docs) -> None:
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    print("检索技术比较")
    print("==================================")
    print(f"查询：{query}\n")

    print("基线检索结果:")
    baseline_docs = vectorstore.similarity_search(query, k=2)
    for i, doc in enumerate(baseline_docs):
        print(f"\n文档 {i+1}:")
        print(doc.page_content)

    print("\n高级检索结果:")
    custom_retriever = CustomRetriever(vectorstore=vectorstore)
    advanced_docs = custom_retriever.get_relevant_documents(query)
    for i, doc in enumerate(advanced_docs):
        print(f"\n文档 {i+1}:")
        print(doc.page_content)


query = "what is the capital of france?"
compare_rag_techniques(query, docs)
```

## 方法 2：Cross-Encoder 模型

<div style="text-align: center;">

<img src="../images/rerank_cross_encoder.svg" alt="rerank cross encoder" style="width:40%; height:auto;">
</div>

### 定义 Cross-Encoder 类

```python
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

class CrossEncoderRetriever(BaseRetriever, BaseModel):
    vectorstore: Any = Field(description="用于初始检索的向量存储")
    cross_encoder: Any = Field(description="用于重排序的 Cross-encoder 模型")
    k: int = Field(default=5, description="初始检索的文档数量")
    rerank_top_k: int = Field(default=3, description="重排序后返回的文档数量")

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        # 初始检索
        initial_docs = self.vectorstore.similarity_search(query, k=self.k)

        # 为 cross-encoder 准备配对
        pairs = [[query, doc.page_content] for doc in initial_docs]

        # 获取 cross-encoder 得分
        scores = self.cross_encoder.predict(pairs)

        # 按得分对文档排序
        scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)

        # 返回重排序后的前 k 个文档
        return [doc for doc, _ in scored_docs[:self.rerank_top_k]]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("异步检索未实现")
```

### 创建实例并通过示例展示

```python
# 创建 cross-encoder 检索器
cross_encoder_retriever = CrossEncoderRetriever(
    vectorstore=vectorstore,
    cross_encoder=cross_encoder,
    k=10,  # 初始检索 10 个文档
    rerank_top_k=5  # 重排序后返回前 5 个
)

# 设置 LLM
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

# 创建带有 cross-encoder 检索器的 RetrievalQA 链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=cross_encoder_retriever,
    return_source_documents=True
)

# 示例查询
query = "What are the impacts of climate change on biodiversity?"
result = qa_chain({"query": query})

print(f"\n问题：{query}")
print(f"答案：{result['result']}")
print("\n相关源文档:")
for i, doc in enumerate(result["source_documents"]):
    print(f"\n文档 {i+1}:")
    print(doc.page_content[:200] + "...")  # 打印每个文档的前 200 个字符
```

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--reranking)
