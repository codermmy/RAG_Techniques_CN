# 简单 RAG（检索增强生成）系统 - CSV 文件

## 概述

此代码实现了一个基本的检索增强生成 (RAG) 系统，用于处理和查询 CSV 文档。系统将文档内容编码到向量存储中，然后可以查询以检索相关信息。

# CSV 文件结构和用例
CSV 文件包含虚拟客户数据，包括名字、姓氏、公司等各种属性。此数据集将用于 RAG 用例，便于创建客户信息问答系统。

## 关键组件

1. 加载和拆分 CSV 文件。
2. 向量存储创建使用 [FAISS](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) 和 OpenAI embeddings
3. 检索器设置用于查询处理后的文档
4. 在 CSV 数据上创建问答。

## 方法详情

### 文档预处理

1. 使用 langchain Csvloader 加载 CSV
2. 数据被分割成块。


### 向量存储创建

1. OpenAI embeddings 用于创建文本块的向量表示。
2. 从这些 embeddings 创建 FAISS 向量存储以进行高效的相似性搜索。

### 检索器设置

1. 配置检索器以获取给定查询的最相关块。

## 此方法的优势

1. 可扩展性：可以通过分块处理大型文档。
2. 灵活性：易于调整块大小和检索结果数量等参数。
3. 效率：利用 FAISS 在高维空间中进行快速相似性搜索。
4. 与高级 NLP 集成：使用 OpenAI embeddings 进行最先进的文本表示。

## 结论

这个简单的 RAG 系统为构建更复杂的信息检索和问答系统提供了坚实的基础。通过将文档内容编码到可搜索的向量存储中，它能够高效地检索相关信息以响应查询。这种方法对于需要在 CSV 文件中快速访问特定信息的应用程序特别有用。

## 导入库

# 包安装和导入

下面的单元格安装运行此笔记本所需的所有必要包。


```python
# 安装所需的包
!pip install faiss-cpu langchain langchain-community langchain-openai pandas python-dotenv
```

```python
from langchain_community.document_loaders.csv_loader import CSVLoader
from pathlib import Path
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
import os
from dotenv import load_dotenv

# 从 .env 文件加载环境变量
load_dotenv()

# 设置 OpenAI API 密钥环境变量
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
```

# CSV 文件结构和用例
CSV 文件包含虚拟客户数据，包括名字、姓氏、公司等各种属性。此数据集将用于 RAG 用例，便于创建客户信息问答系统。

```python
# 下载所需的数据文件
import os
os.makedirs('data', exist_ok=True)

# 下载此笔记本中使用的 PDF 文档
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
!wget -O data/customers-100.csv https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/customers-100.csv

```

```python
import pandas as pd

file_path = ('data/customers-100.csv') # 插入 csv 文件的路径
data = pd.read_csv(file_path)

# 预览 csv 文件
data.head()
```

## 加载和处理 CSV 数据

```python
loader = CSVLoader(file_path=file_path)
docs = loader.load_and_split()
```

## 初始化 FAISS 向量存储和 OpenAI embedding

```python
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings()
index = faiss.IndexFlatL2(len(OpenAIEmbeddings().embed_query(" ")))
vector_store = FAISS(
    embedding_function=OpenAIEmbeddings(),
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)
```

## 将拆分的 CSV 数据添加到向量存储

```python
vector_store.add_documents(documents=docs)
```

## 创建检索链

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

retriever = vector_store.as_retriever()

# 设置系统提示词
system_prompt = (
    "你是一个问答任务的助手。"
    "使用以下检索到的上下文片段来回答"
    "问题。如果你不知道答案，就说你"
    "不知道。最多使用三句话，保持"
    "答案简洁。"
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),

])

# 创建问答链
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
```

## 基于 CSV 数据向 RAG 机器人提问

```python
answer= rag_chain.invoke({"input": "Sheryl Baxter 在哪家公司工作？"})
answer['answer']
```

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--simple-csv-rag)
