<!-- ![](https://europe-west1-atp-views-tracker.cloudfunctions.net/working-analytics?notebook=adaptive-retrieval) -->



# 自适应检索增强生成（RAG）系统

## 概述

本系统实现了一种先进的检索增强生成（RAG）方法，可根据查询类型自适应调整检索策略。通过在各个阶段利用语言模型（LLM），旨在为用户查询提供更准确、相关和上下文感知的响应。

## 动机

传统 RAG 系统通常使用一刀切的检索方法，这对于不同类型的查询可能不是最优的。我们的自适应系统的动机源于这样一种理解：不同类型的问题需要不同的检索策略。例如，事实性查询可能受益于精确、聚焦的检索，而分析性查询可能需要更广泛、更多样化的信息集。

## 关键组件

1. **查询分类器**：确定查询类型（事实性、分析性、观点性或情境性）。

2. **自适应检索策略**：针对不同查询类型量身定制的四种独特策略：
   - 事实性策略
   - 分析性策略
   - 观点性策略
   - 情境性策略

3. **LLM 集成**：在整个过程中使用 LLM 来增强检索和排序。

4. **OpenAI GPT 模型**：使用检索到的文档作为上下文生成最终响应。

## 方法详情

### 1. 查询分类

系统首先将用户查询分类为四个类别之一：
- 事实性：寻求特定、可验证信息的查询。
- 分析性：需要全面分析或解释的查询。
- 观点性：关于主观事项或寻求不同观点的查询。
- 情境性：依赖于用户特定上下文的查询。

### 2. 自适应检索策略

每种查询类型触发特定的检索策略：

#### 事实性策略
- 使用 LLM 增强原始查询以提高精度。
- 基于增强后的查询检索文档。
- 使用 LLM 按相关性对文档进行排序。

#### 分析性策略
- 使用 LLM 生成多个子查询以覆盖主查询的不同方面。
- 为每个子查询检索文档。
- 使用 LLM 确保最终文档选择的多样性。

#### 观点性策略
- 使用 LLM 识别主题的不同观点。
- 检索代表每个观点的文档。
- 使用 LLM 从检索到的文档中选择多样化的观点范围。

#### 情境性策略
- 使用 LLM 将用户特定上下文纳入查询。
- 基于情境化查询执行检索。
- 在排序文档时同时考虑相关性和用户上下文。

### 3. LLM 增强排序

检索后，每个策略使用 LLM 对文档进行最终排序。此步骤确保为下一阶段选择最相关和最适当的文档。

### 4. 响应生成

最终检索到的文档集传递给 OpenAI GPT 模型，该模型基于查询和提供的上下文生成响应。

## 此方法的优势

1. **提高准确性**：通过根据查询类型定制检索策略，系统可以提供更准确和相关的信息。

2. **灵活性**：系统适应不同类型的查询，处理广泛的用户需求。

3. **上下文感知**：特别是对于情境性查询，系统可以纳入用户特定信息以提供更个性化的响应。

4. **多样化视角**：对于基于观点的查询，系统主动寻找并呈现多种观点。

5. **全面分析**：分析性策略确保对复杂主题的彻底探索。

## 结论

这种自适应 RAG 系统代表了相对于传统 RAG 方法的显著进步。通过动态调整其检索策略并在整个过程中利用 LLM，它旨在为各种用户查询提供更准确、相关和细致的响应。

<div style="text-align: center;">

<img src="../images/adaptive_retrieval.svg" alt="adaptive retrieval" style="width:100%; height:auto;">
</div>

# 包安装和导入

下面的单元格安装运行此 notebook 所需的所有必要包。

```python
# 安装所需的包
!pip install faiss-cpu langchain langchain-openai python-dotenv
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
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate

from langchain_core.retrievers import BaseRetriever
from typing import Dict, Any
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field


# 原始路径追加已替换为 Colab 兼容性
from helper_functions import *
from evaluation.evalute_rag import *

# 从.env 文件加载环境变量
load_dotenv()

# 设置 OpenAI API 密钥环境变量
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
```

### 定义查询分类器类

```python
class categories_options(BaseModel):
        category: str = Field(description="The category of the query, the options are: Factual, Analytical, Opinion, or Contextual", example="Factual")


class QueryClassifier:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
        self.prompt = PromptTemplate(
            input_variables=["query"],
            template="将以下查询分类为以下类别之一：Factual（事实性）、Analytical（分析性）、Opinion（观点性）或 Contextual（情境性）。\n查询：{query}\n类别:"
        )
        self.chain = self.prompt | self.llm.with_structured_output(categories_options)


    def classify(self, query):
        print("分类查询")
        return self.chain.invoke(query).category
```

### 定义基础检索器类，复杂检索器将继承自它

```python
class BaseRetrievalStrategy:
    def __init__(self, texts):
        self.embeddings = OpenAIEmbeddings()
        text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=0)
        self.documents = text_splitter.create_documents(texts)
        self.db = FAISS.from_documents(self.documents, self.embeddings)
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)


    def retrieve(self, query, k=4):
        return self.db.similarity_search(query, k=k)
```

### 定义事实性检索策略

```python
class relevant_score(BaseModel):
        score: float = Field(description="The relevance score of the document to the query", example=8.0)

class FactualRetrievalStrategy(BaseRetrievalStrategy):
    def retrieve(self, query, k=4):
        print("检索事实性")
        # 使用 LLM 增强查询
        enhanced_query_prompt = PromptTemplate(
            input_variables=["query"],
            template="增强此事实性查询以获得更好的信息检索：{query}"
        )
        query_chain = enhanced_query_prompt | self.llm
        enhanced_query = query_chain.invoke(query).content
        print(f'增强后的查询：{enhanced_query}')

        # 使用增强后的查询检索文档
        docs = self.db.similarity_search(enhanced_query, k=k*2)

        # 使用 LLM 对检索到的文档进行相关性排序
        ranking_prompt = PromptTemplate(
            input_variables=["query", "doc"],
            template="在 1-10 的评分标准上，此文​​档与查询 '{query}' 的相关性如何？\n文档：{doc}\n相关性得分："
        )
        ranking_chain = ranking_prompt | self.llm.with_structured_output(relevant_score)

        ranked_docs = []
        print("对文档排序")
        for doc in docs:
            input_data = {"query": enhanced_query, "doc": doc.page_content}
            score = float(ranking_chain.invoke(input_data).score)
            ranked_docs.append((doc, score))

        # 按相关性分数排序并返回前 k 个
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_docs[:k]]
```

### 定义分析性检索策略

```python
class SelectedIndices(BaseModel):
    indices: List[int] = Field(description="Indices of selected documents", example=[0, 1, 2, 3])

class SubQueries(BaseModel):
    sub_queries: List[str] = Field(description="List of sub-queries for comprehensive analysis", example=["What is the population of New York?", "What is the GDP of New York?"])

class AnalyticalRetrievalStrategy(BaseRetrievalStrategy):
    def retrieve(self, query, k=4):
        print("检索分析性")
        # 使用 LLM 生成子查询以进行全面分析
        sub_queries_prompt = PromptTemplate(
            input_variables=["query", "k"],
            template="为以下查询生成{k}个子问题：{query}"
        )

        llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
        sub_queries_chain = sub_queries_prompt | llm.with_structured_output(SubQueries)

        input_data = {"query": query, "k": k}
        sub_queries = sub_queries_chain.invoke(input_data).sub_queries
        print(f'用于全面分析的子查询：{sub_queries}')

        all_docs = []
        for sub_query in sub_queries:
            all_docs.extend(self.db.similarity_search(sub_query, k=2))

        # 使用 LLM 确保多样性和相关性
        diversity_prompt = PromptTemplate(
            input_variables=["query", "docs", "k"],
            template="""为查询 '{query}' 选择最多样化和最相关的{k}个文档\n文档：{docs}\n
            仅返回所选文档的索引作为整数列表。"""
        )
        diversity_chain = diversity_prompt | self.llm.with_structured_output(SelectedIndices)
        docs_text = "\n".join([f"{i}: {doc.page_content[:50]}..." for i, doc in enumerate(all_docs)])
        input_data = {"query": query, "docs": docs_text, "k": k}
        selected_indices_result = diversity_chain.invoke(input_data).indices
        print(f'选择了多样化和相关的文档')

        return [all_docs[i] for i in selected_indices_result if i < len(all_docs)]
```

### 定义观点性检索策略

```python
class OpinionRetrievalStrategy(BaseRetrievalStrategy):
    def retrieve(self, query, k=3):
        print("检索观点性")
        # 使用 LLM 识别潜在观点
        viewpoints_prompt = PromptTemplate(
            input_variables=["query", "k"],
            template="识别主题 {query} 的{k}个不同观点或视角"
        )
        viewpoints_chain = viewpoints_prompt | self.llm
        input_data = {"query": query, "k": k}
        viewpoints = viewpoints_chain.invoke(input_data).content.split('\n')
        print(f'观点：{viewpoints}')

        all_docs = []
        for viewpoint in viewpoints:
            all_docs.extend(self.db.similarity_search(f"{query} {viewpoint}", k=2))

        # 使用 LLM 分类并选择多样化的观点
        opinion_prompt = PromptTemplate(
            input_variables=["query", "docs", "k"],
            template="将这些文档分类为关于 '{query}' 的不同观点，并选择{k}个最具代表性和多样化的观点:\n文档：{docs}\n所选索引:"
        )
        opinion_chain = opinion_prompt | self.llm.with_structured_output(SelectedIndices)

        docs_text = "\n".join([f"{i}: {doc.page_content[:100]}..." for i, doc in enumerate(all_docs)])
        input_data = {"query": query, "docs": docs_text, "k": k}
        selected_indices = opinion_chain.invoke(input_data).indices
        print(f'选择了多样化和相关的文档')

        return [all_docs[int(i)] for i in selected_indices.split() if i.isdigit() and int(i) < len(all_docs)]
```

### 定义情境性检索策略

```python
class ContextualRetrievalStrategy(BaseRetrievalStrategy):
    def retrieve(self, query, k=4, user_context=None):
        print("检索情境性")
        # 使用 LLM 将用户上下文融入查询
        context_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="给定用户上下文：{context}\n重新构建查询以最好地满足用户需求：{query}"
        )
        context_chain = context_prompt | self.llm
        input_data = {"query": query, "context": user_context or "No specific context provided"}
        contextualized_query = context_chain.invoke(input_data).content
        print(f'情境化查询：{contextualized_query}')

        # 使用情境化查询检索文档
        docs = self.db.similarity_search(contextualized_query, k=k*2)

        # 使用 LLM 对检索到的文档进行相关性排序，同时考虑用户上下文
        ranking_prompt = PromptTemplate(
            input_variables=["query", "context", "doc"],
            template="给定查询：'{query}' 和用户上下文：'{context}'，在 1-10 的评分标准上评价此文​​档的相关性:\n文档：{doc}\n相关性得分:"
        )
        ranking_chain = ranking_prompt | self.llm.with_structured_output(relevant_score)
        print("对文档排序")

        ranked_docs = []
        for doc in docs:
            input_data = {"query": contextualized_query, "context": user_context or "No specific context provided", "doc": doc.page_content}
            score = float(ranking_chain.invoke(input_data).score)
            ranked_docs.append((doc, score))


        # 按相关性分数排序并返回前 k 个
        ranked_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in ranked_docs[:k]]
```

### 定义自适应检索器类

```python
class AdaptiveRetriever:
    def __init__(self, texts: List[str]):
        self.classifier = QueryClassifier()
        self.strategies = {
            "Factual": FactualRetrievalStrategy(texts),
            "Analytical": AnalyticalRetrievalStrategy(texts),
            "Opinion": OpinionRetrievalStrategy(texts),
            "Contextual": ContextualRetrievalStrategy(texts)
        }

    def get_relevant_documents(self, query: str) -> List[Document]:
        category = self.classifier.classify(query)
        strategy = self.strategies[category]
        return strategy.retrieve(query)
```

### 定义继承自 langchain BaseRetriever 的附加检索器

```python
class PydanticAdaptiveRetriever(BaseRetriever):
    adaptive_retriever: AdaptiveRetriever = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.adaptive_retriever.get_relevant_documents(query)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)
```

### 定义自适应 RAG 类

```python
class AdaptiveRAG:
    def __init__(self, texts: List[str]):
        adaptive_retriever = AdaptiveRetriever(texts)
        self.retriever = PydanticAdaptiveRetriever(adaptive_retriever=adaptive_retriever)
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)

        # 创建自定义提示
        prompt_template = """使用以下上下文片段回答最后的问题。
        如果您不知道答案，只需说您不知道即可，不要试图编造答案。

        {context}

        问题：{question}
        答案："""
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        # 创建 LLM 链
        self.llm_chain = prompt | self.llm



    def answer(self, query: str) -> str:
        docs = self.retriever.get_relevant_documents(query)
        input_data = {"context": "\n".join([doc.page_content for doc in docs]), "question": query}
        return self.llm_chain.invoke(input_data)
```

### 演示此模型的使用

```python
# 使用示例
texts = [
    "The Earth is the third planet from the Sun and the only astronomical object known to harbor life."
    ]
rag_system = AdaptiveRAG(texts)
```

### 展示四种不同类型的查询

```python
factual_result = rag_system.answer("What is the distance between the Earth and the Sun?").content
print(f"答案：{factual_result}")

analytical_result = rag_system.answer("How does the Earth's distance from the Sun affect its climate?").content
print(f"答案：{analytical_result}")

opinion_result = rag_system.answer("What are the different theories about the origin of life on Earth?").content
print(f"答案：{opinion_result}")

contextual_result = rag_system.answer("How does the Earth's position in the Solar System influence its habitability?").content
print(f"答案：{contextual_result}")
```

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--adaptive-retrieval)
