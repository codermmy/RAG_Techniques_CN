### 可视化表示

<img src="../images/reliable_rag.svg" alt="Reliable-RAG" width="300">

# 可靠 RAG（Reliable RAG）系统

# 包安装和导入

下面的单元格安装运行此笔记本所需的所有必要包。


```python
# 安装所需的包
!pip install langchain langchain-community python-dotenv
```

```python
### LLMs
import os
from dotenv import load_dotenv

# 从 '.env' 文件加载环境变量
load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY') # 用于 LLM -- llama-3.1-8b (small) & mixtral-8x7b-32768 (large)
os.environ['COHERE_API_KEY'] = os.getenv('COHERE_API_KEY') # 用于 embedding
```

### 创建向量存储

```python
### 构建索引
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings

# 设置 embeddings
embedding_model = CohereEmbeddings(model="embed-english-v3.0")

# 要索引的文档
urls = [
    "https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-2-reflection/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-3-tool-use/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-4-planning/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-5-multi-agent-collaboration/?ref=dl-staging-website.ghost.io"
]

# 加载
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# 分割
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# 添加到向量存储
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag",
    embedding=embedding_model,
)

retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={'k': 4}, # 要检索的文档数量
            )
```

### 问题

```python
question = "不同类型的 Agent 设计模式有哪些？"
```

### 检索文档

```python
docs = retriever.invoke(question)
```

### 检查文档外观

```python
print(f"标题：{docs[0].metadata['title']}\n\n来源：{docs[0].metadata['source']}\n\n内容：{docs[0].page_content}\n")
```

### 检查文档相关性

```python
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

# 数据模型
class GradeDocuments(BaseModel):
    """对检索到的文档进行相关性检查的二进制评分。"""

    binary_score: str = Field(
        description="文档与问题相关，'yes' 或 'no'"
    )


# 使用函数调用的 LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# 提示
system = """你是一个评估检索到的文档与用户问题相关性的评分者。\n
    如果文档包含与用户问题相关的关键词或语义含义，则将其评为相关。\n
    不需要是严格的测试。目标是过滤掉错误的检索结果。\n
    给出二进制评分 'yes' 或 'no' 以表示文档是否与问题相关。"""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "检索到的文档：\n\n {document} \n\n 用户问题：{question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
```

### 过滤不相关文档

```python
docs_to_use = []
for doc in docs:
    print(doc.page_content, '\n', '-'*50)
    res = retrieval_grader.invoke({"question": question, "document": doc.page_content})
    print(res,'\n')
    if res.binary_score == 'yes':
        docs_to_use.append(doc)
```

### 生成结果

```python
from langchain_core.output_parsers import StrOutputParser

# 提示
system = """你是一个问答任务的助手。根据你的知识回答问题。
最多使用三到五句话，保持答案简洁。"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "检索到的文档：\n\n <docs>{documents}</docs> \n\n 用户问题：<question>{question}</question>"),
    ]
)

# LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# 后处理
def format_docs(docs):
    return "\n".join(f"<doc{i+1}>:\n标题:{doc.metadata['title']}\n来源:{doc.metadata['source']}\n内容:{doc.page_content}\n</doc{i+1}>\n" for i, doc in enumerate(docs))

# 链
rag_chain = prompt | llm | StrOutputParser()

# 运行
generation = rag_chain.invoke({"documents":format_docs(docs_to_use), "question": question})
print(generation)
```

### 检查幻觉

```python
# 数据模型
class GradeHallucinations(BaseModel):
    """对 'generation' 答案中是否存在幻觉的二进制评分。"""

    binary_score: str = Field(
        ...,
        description="答案基于事实，'yes' 或 'no'"
    )

# 使用函数调用的 LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# 提示
system = """你是一个评估 LLM 生成是否基于/支持一组检索到的事实的评分者。\n
    给出二进制评分 'yes' 或 'no'。'Yes' 表示答案基于/支持这组事实。"""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "事实集合：\n\n <facts>{documents}</facts> \n\n LLM 生成：<generation>{generation}</generation>"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader

response = hallucination_grader.invoke({"documents": format_docs(docs_to_use), "generation": generation})
print(response)
```

### 高亮使用的文档

```python
from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

# 数据模型
class HighlightDocuments(BaseModel):
    """返回用于回答问题的文档特定部分。"""

    id: List[str] = Field(
        ...,
        description="用于回答问题的文档 ID 列表"
    )

    title: List[str] = Field(
        ...,
        description="用于回答问题的标题列表"
    )

    source: List[str] = Field(
        ...,
        description="用于回答问题的来源列表"
    )

    segment: List[str] = Field(
        ...,
        description="用于回答问题的文档直接片段列表"
    )

# LLM
llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

# 解析器
parser = PydanticOutputParser(pydantic_object=HighlightDocuments)

# 提示
system = """你是一个用于文档搜索和检索的高级助手。提供以下信息：
1. 一个问题。
2. 基于问题生成的答案。
3. 在生成答案时引用的一组文档。

你的任务是从提供的文档中识别和提取完全内联的片段，这些片段直接对应于用于生成给定答案的内容。提取的片段必须是文档的逐字摘录，确保与提供的文档中的文本逐字匹配。

确保：
- （重要）每个片段与文档的一部分完全匹配，并完全包含在文档文本中。
- 每个片段与生成答案的相关性清晰，并直接支持提供的答案。
- （重要）如果你没有使用特定文档，不要提及它。

使用的文档：<docs>{documents}</docs> \n\n 用户问题：<question>{question}</question> \n\n 生成的答案：<answer>{generation}</answer>

<format_instruction>
{format_instructions}
</format_instruction>
"""


prompt = PromptTemplate(
    template= system,
    input_variables=["documents", "question", "generation"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# 链
doc_lookup = prompt | llm | parser

# 运行
lookup_response = doc_lookup.invoke({"documents":format_docs(docs_to_use), "question": question, "generation": generation})
```

```python
for id, title, source, segment in zip(lookup_response.id, lookup_response.title, lookup_response.source, lookup_response.segment):
    print(f"ID: {id}\n标题：{title}\n来源：{source}\n文本片段：{segment}\n")
```

### 源文本片段

![image.png](attachment:image.png)

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--reliable-rag)
