# 命题分块 (Propositions Chunking)

### 概述

本代码实现了命题分块方法，基于 [Tony Chen 等人的研究](https://doi.org/10.48550/arXiv.2312.06648)。该系统将输入文本分解为原子性、事实性、自包含且简洁的命题，然后将这些命题编码到向量存储中，以便后续用于检索。

### 关键组件

1. **文档分块 (Document Chunking)：** 将文档分割成易于管理的片段进行分析。
2. **命题生成 (Proposition Generation)：** 使用 LLM 将文档块分解为事实性、自包含的命题。
3. **命题质量检查 (Proposition Quality Check)：** 基于准确性、清晰度、完整性和简洁性评估生成的命题。
4. **Embedding 和向量存储 (Embedding and Vector Store)：** 将命题和较大的文档块嵌入到向量存储中，以实现高效检索。
5. **检索与比较 (Retrieval and Comparison)：** 使用不同大小的查询测试检索系统，比较基于命题的模型与较大块模型的结果。

<img src="../images/proposition_chunking.svg" alt="Reliable-RAG" width="600">

### 动机

命题分块方法的动机是构建一个系统，将文本文档分解为简洁、事实性的命题，以实现更细粒度和精确的信息检索。使用命题可以实现更精细的控制，更好地处理特定查询，特别是从详细或复杂的文本中提取知识。通过比较较小的命题块和较大的文档块，旨在评估细粒度信息检索的有效性。

### 方法详情

1. **加载环境变量：** 代码首先加载环境变量（例如 LLM 服务的 API 密钥），以确保系统可以访问必要的资源。

2. **文档分块：**
   - 使用 `RecursiveCharacterTextSplitter` 将输入文档分割成较小的片段（块）。这确保每个块的大小适合 LLM 处理。

3. **命题生成：**
   - 使用 LLM（本例中使用 "llama-3.1-70b-versatile"）从每个块生成命题。输出结构为事实性、自包含陈述的列表，无需额外上下文即可理解。

4. **质量检查：**
   - 第二个 LLM 通过对准确性、清晰度、完整性和简洁性进行评分来评估命题的质量。在所有类别中满足所需阈值的命题将被保留。

5. **命题 Embedding：**
   - 通过质量检查的命题使用 `OllamaEmbeddings` 模型嵌入到向量存储中。这允许在查询时进行基于相似性的命题检索。

6. **检索与比较：**
   - 构建两个检索系统：一个使用基于命题的块，另一个使用较大的文档块。两者都使用多个查询进行测试，以比较其性能和返回结果的精度。

### 优势

- **细粒度：** 通过将文档分解为小型事实性命题，系统允许高度特定的检索，使得从大型或复杂文档中提取精确答案更加容易。
- **质量保证：** 使用质量检查 LLM 确保生成的命题符合特定标准，提高检索信息的可靠性。
- **检索灵活性：** 通过比较基于命题和较大块的检索，可以评估搜索结果中细粒度与更广泛上下文之间的权衡。

### 实现

1. **命题生成：** LLM 与自定义提示配合使用，从文档块生成事实性陈述。
2. **质量检查：** 生成的命题通过评分系统，评估准确性、清晰度、完整性和简洁性。
3. **向量存储集成：** 命题在使用预训练 embedding 模型嵌入后存储在 FAISS 向量存储中，允许高效的基于相似性的搜索和检索。
4. **查询测试：** 对向量存储（基于命题和较大块）进行多个测试查询，以比较检索性能。

### 总结

本代码展示了一种使用 LLM 将文档分解为自包含命题的稳健方法。系统对每个命题进行质量检查，将其嵌入向量存储，并根据用户查询检索最相关的信息。比较细粒度命题与较大文档块的能力，提供了关于哪种方法对不同类型查询产生更准确或有用结果的洞察。该方法强调了高质量命题生成和检索对于从复杂文档中精确提取信息的重要性。

# 包安装与导入

下面的单元格安装运行此 notebook 所需的所有必要包。


```python
# 安装所需的包
!pip install faiss-cpu langchain langchain-community python-dotenv
```

```python
### LLMs
import os
from dotenv import load_dotenv

# 从 '.env' 文件加载环境变量
load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY') # 用于 LLM
```

### 测试文档

```python
sample_content = """Paul Graham 的文章"Founder Mode"发表于 2024 年 9 月，挑战了关于初创企业规模扩展的传统智慧，认为创始人应该保持独特的管理风格，而不是随着公司成长采用传统的企业管理实践。
传统智慧 vs 创始人模式
文章认为，给成长型公司的传统建议——雇佣优秀人才并给予他们自主权——在应用于初创企业时往往失败。
这种方法虽然适合成熟公司，但对创始人愿景和直接参与至关重要的初创企业可能有害。"Founder Mode"被描述为一个尚未完全理解或记录的新兴范式，与商学院和职业经理人经常建议的传统"经理人模式"形成对比。
独特的创始人能力
创始人拥有职业经理人没有的独特见解和能力，主要是因为他们对公司的愿景和文化有深刻理解。
Graham 建议创始人应该利用这些优势，而不是 conform 传统管理实践。"Founder Mode"是一个尚未完全理解或记录的新兴范式，Graham 希望随着时间的推移，它能像传统经理人模式一样被充分理解，使创始人即使在公司规模扩展时也能保持独特的方法。
扩展初创企业的挑战
随着初创企业成长，人们普遍认为必须过渡到更结构化的管理方法。然而，许多创始人发现这种过渡有问题，因为它经常导致失去推动初创企业最初成功的创新和敏捷精神。
Airbnb 联合创始人 Brian Chesky 分享了他的经验，他被告知以传统管理风格运营公司，导致了糟糕的结果。他最终通过采用不同的方法取得了成功，这个方法受到 Steve Jobs 管理 Apple 方式的启发。
Steve Jobs 的管理风格
Steve Jobs 在 Apple 的管理方法成为 Brian Chesky 在 Airbnb 实施"Founder Mode"的灵感来源。一个值得注意的做法是 Jobs 每年为 Apple 最重要的 100 人举办 retreat，无论他们在组织结构图上的位置如何。这种非常规方法使 Jobs 即使在 Apple 成长时也能保持初创企业般的环境，培养跨层级的创新和直接沟通。这些实践强调了创始人深入参与公司运营的重要性，挑战了随着公司规模扩大将责任委托给职业经理人的传统观念。
"""
```

### 分块 (Chunking)

```python
### 构建索引
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

# 设置 embeddings
embedding_model = OllamaEmbeddings(model='nomic-embed-text:v1.5', show_progress=True)

# 文档列表
docs_list = [Document(page_content=sample_content, metadata={"Title": "Paul Graham 的 Founder Mode 文章", "Source": "https://www.perplexity.ai/page/paul-graham-s-founder-mode-ess-t9TCyvkqRiyMQJWsHr0fnQ"})]

# 分割
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=200, chunk_overlap=50
)

doc_splits = text_splitter.split_documents(docs_list)
```

```python
for i, doc in enumerate(doc_splits):
    doc.metadata['chunk_id'] = i+1 ### 添加块 id
```

### 生成命题 (Generate Propositions)

```python
from typing import List
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq

# 数据模型
class GeneratePropositions(BaseModel):
    """给定文档中的所有命题列表"""

    propositions: List[str] = Field(
        description="命题列表（事实性、自包含且简洁的信息）"
    )


# 使用函数调用的 LLM
llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
structured_llm= llm.with_structured_output(GeneratePropositions)

# Few shot 提示 --- 我们可以添加更多示例使其更好
proposition_examples = [
    {"document":
        "1969 年，Neil Armstrong 在 Apollo 11 任务期间成为第一个在月球上行走的人。",
     "propositions":
        "['Neil Armstrong 是一名宇航员。', 'Neil Armstrong 于 1969 年在月球上行走。', 'Neil Armstrong 是第一个在月球上行走的人。', 'Neil Armstrong 在 Apollo 11 任务期间在月球上行走。', 'Apollo 11 任务发生于 1969 年。']"
    },
]

example_proposition_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{document}"),
        ("ai", "{propositions}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt = example_proposition_prompt,
    examples = proposition_examples,
)

# 提示
system = """请将以下文本分解为简单、自包含的命题。确保每个命题符合以下标准：

    1. 表达单一事实：每个命题应陈述一个具体事实或主张。
    2. 无需上下文即可理解：命题应是自包含的，意味着无需额外上下文即可理解。
    3. 使用全名，不使用代词：避免代词或模糊引用；使用全实体名称。
    4. 包含相关日期/限定词：如果适用，包含必要的日期、时间和限定词以使事实精确。
    5. 包含一个主谓关系：专注于单一主体及其对应的动作或属性，不使用连词或多个从句。"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        few_shot_prompt,
        ("human", "{document}"),
    ]
)

proposition_generator = prompt | structured_llm
```

```python
propositions = [] # 存储文档中的所有命题

for i in range(len(doc_splits)):
    response = proposition_generator.invoke({"document": doc_splits[i].page_content}) # 创建命题
    for proposition in response.propositions:
        propositions.append(Document(page_content=proposition, metadata={"Title": "Paul Graham 的 Founder Mode 文章", "Source": "https://www.perplexity.ai/page/paul-graham-s-founder-mode-ess-t9TCyvkqRiyMQJWsHr0fnQ", "chunk_id": i+1}))
```

### 质量检查 (Quality Check)

```python
# 数据模型
class GradePropositions(BaseModel):
    """对给定命题的准确性、清晰度、完整性和简洁性进行评分"""

    accuracy: int = Field(
        description="根据命题反映原文的程度，评分 1-10。"
    )

    clarity: int = Field(
        description="根据命题在无额外上下文情况下的易理解程度，评分 1-10。"
    )

    completeness: int = Field(
        description="根据命题是否包含必要细节（如日期、限定词），评分 1-10。"
    )

    conciseness: int = Field(
        description="根据命题是否简洁且不丢失重要信息，评分 1-10。"
    )

# 使用函数调用的 LLM
llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
structured_llm= llm.with_structured_output(GradePropositions)

# 提示
evaluation_prompt_template = """
请根据以下标准评估以下命题：
- **准确性**：根据命题反映原文的程度，评分 1-10。
- **清晰度**：根据命题在无额外上下文情况下的易理解程度，评分 1-10。
- **完整性**：根据命题是否包含必要细节（如日期、限定词），评分 1-10。
- **简洁性**：根据命题是否简洁且不丢失重要信息，评分 1-10。

示例：
Docs: 1969 年，Neil Armstrong 在 Apollo 11 任务期间成为第一个在月球上行走的人。

Propositons_1: Neil Armstrong 是一名宇航员。
Evaluation_1: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

Propositons_2: Neil Armstrong 于 1969 年在月球上行走。
Evaluation_2: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

Propositons_3: Neil Armstrong 是第一个在月球上行走的人。
Evaluation_3: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

Propositons_4: Neil Armstrong 在 Apollo 11 任务期间在月球上行走。
Evaluation_4: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

Propositons_5: Apollo 11 任务发生于 1969 年。
Evaluation_5: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

格式：
Proposition: "{proposition}"
Original Text: "{original_text}"
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", evaluation_prompt_template),
        ("human", "{proposition}, {original_text}"),
    ]
)

proposition_evaluator = prompt | structured_llm
```

```python
# 定义评估类别和阈值
evaluation_categories = ["accuracy", "clarity", "completeness", "conciseness"]
thresholds = {"accuracy": 7, "clarity": 7, "completeness": 7, "conciseness": 7}

# 评估命题的函数
def evaluate_proposition(proposition, original_text):
    response = proposition_evaluator.invoke({"proposition": proposition, "original_text": original_text})

    # 解析响应以提取分数
    scores = {"accuracy": response.accuracy, "clarity": response.clarity, "completeness": response.completeness, "conciseness": response.conciseness}
    return scores

# 检查命题是否通过质量检查
def passes_quality_check(scores):
    for category, score in scores.items():
        if score < thresholds[category]:
            return False
    return True

evaluated_propositions = [] # 存储文档中所有评估后的命题

# 遍历生成的命题并评估它们
for idx, proposition in enumerate(propositions):
    scores = evaluate_proposition(proposition.page_content, doc_splits[proposition.metadata['chunk_id'] - 1].page_content)
    if passes_quality_check(scores):
        # 命题通过质量检查，保留它
        evaluated_propositions.append(proposition)
    else:
        # 命题未通过，丢弃或标记以进一步审查
        print(f"{idx+1}) 命题：{proposition.page_content} \n 分数：{scores}")
        print("未通过")
```

### 将命题嵌入向量存储

```python
# 添加到向量存储
vectorstore_propositions = FAISS.from_documents(evaluated_propositions, embedding_model)
retriever_propositions = vectorstore_propositions.as_retriever(
                search_type="similarity",
                search_kwargs={'k': 4}, # 要检索的文档数量
            )
```

```python
query = "谁的管理方法成为 Brian Chesky 在 Airbnb 实施'Founder Mode'的灵感来源？"
res_proposition = retriever_propositions.invoke(query)
```

```python
for i, r in enumerate(res_proposition):
    print(f"{i+1}) 内容：{r.page_content} --- 块 ID: {r.metadata['chunk_id']}")
```

### 与较大块大小的性能比较

```python
# 添加到向量存储_larger_
vectorstore_larger = FAISS.from_documents(doc_splits, embedding_model)
retriever_larger = vectorstore_larger.as_retriever(
                search_type="similarity",
                search_kwargs={'k': 4}, # 要检索的文档数量
            )
```

```python
res_larger = retriever_larger.invoke(query)
```

```python
for i, r in enumerate(res_larger):
    print(f"{i+1}) 内容：{r.page_content} --- 块 ID: {r.metadata['chunk_id']}")
```

### 测试

#### 测试 - 1

```python
test_query_1 = "文章'Founder Mode'是关于什么的？"
res_proposition = retriever_propositions.invoke(test_query_1)
res_larger = retriever_larger.invoke(test_query_1)
```

```python
for i, r in enumerate(res_proposition):
    print(f"{i+1}) 内容：{r.page_content} --- 块 ID: {r.metadata['chunk_id']}")
```

```python
for i, r in enumerate(res_larger):
    print(f"{i+1}) 内容：{r.page_content} --- 块 ID: {r.metadata['chunk_id']}")
```

#### 测试 - 2

```python
test_query_2 = "谁是 Airbnb 的联合创始人？"
res_proposition = retriever_propositions.invoke(test_query_2)
res_larger = retriever_larger.invoke(test_query_2)
```

```python
for i, r in enumerate(res_proposition):
    print(f"{i+1}) 内容：{r.page_content} --- 块 ID: {r.metadata['chunk_id']}")
```

```python
for i, r in enumerate(res_larger):
    print(f"{i+1}) 内容：{r.page_content} --- 块 ID: {r.metadata['chunk_id']}")
```

#### 测试 - 3

```python
test_query_3 = "文章'founder mode'是什么时候发表的？"
res_proposition = retriever_propositions.invoke(test_query_3)
res_larger = retriever_larger.invoke(test_query_3)
```

```python
for i, r in enumerate(res_proposition):
    print(f"{i+1}) 内容：{r.page_content} --- 块 ID: {r.metadata['chunk_id']}")
```

```python
for i, r in enumerate(res_larger):
    print(f"{i+1}) 内容：{r.page_content} --- 块 ID: {r.metadata['chunk_id']}")
```

### 对比

| **方面**                | **基于命题的检索**                                         | **简单块检索**                                              |
|---------------------------|--------------------------------------------------------------------------|--------------------------------------------------------------------------|
| **响应精度**  | 高：提供聚焦且直接的答案。                              | 中：提供更多上下文但可能包含无关信息。    |
| **清晰度和简洁性**    | 高：清晰简洁，避免不必要的细节。                    | 中：更全面但可能令人不知所措。                      |
| **上下文丰富度**    | 低：可能缺乏上下文，专注于特定命题。               | 高：提供额外的上下文和细节。                           |
| **全面性**      | 低：可能省略更广泛的上下文或补充细节。                 | 高：提供更完整的视图和丰富的信息。            |
| **叙述流畅性**         | 中：可能碎片化或不连贯。                                | 高：保持原文的逻辑流畅性和连贯性。 |
| **信息过载**   | 低：不太可能因过多信息而令人不知所措。                  | 高：有过多的信息让用户不知所措的风险。           |
| **用例适用性**   | 最适合快速、事实性查询。                                        | 最适合需要深入理解的复杂查询。               |
| **效率**             | 高：提供快速、有针对性的响应。                               | 中：可能需要更多精力筛选额外内容。      |
| **特异性**            | 高：精确且有针对性的响应。                                   | 中：由于包含更广泛的上下文，答案可能针对性较低。|


![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--proposition-chunking)
