# 文档检索中的上下文压缩

## 概述

此代码演示了使用 LangChain 和 OpenAI 语言模型在文档检索系统中实现上下文压缩。该技术旨在通过压缩和提取与给定查询上下文相关的文档最重要部分，来改善检索信息的相关性和简洁性。

## 动机

传统文档检索系统通常返回整个块或文档，其中可能包含不相关的信息。上下文压缩通过智能地提取和压缩检索文档中最相关的部分来解决此问题，从而实现更集中、更高效的信息检索。

## 关键组件

1. 从 PDF 文档创建向量存储
2. 基础检索器设置
3. 基于 LLM 的上下文压缩器
4. 上下文压缩检索器
5. 集成压缩检索器的问答链

## 方法细节

### 文档预处理和向量存储创建

1. 使用自定义的 `encode_pdf` 函数处理 PDF 并将其编码为向量存储。

### 检索器和压缩器设置

1. 从向量存储创建基础检索器。
2. 使用 OpenAI 的 GPT-4 模型初始化基于 LLM 的上下文压缩器（LLMChainExtractor）。

### 上下文压缩检索器

1. 将基础检索器和压缩器组合成 ContextualCompressionRetriever。
2. 此检索器首先使用基础检索器获取文档，然后应用压缩器提取最相关的信息。

### 问答链

1. 创建一个 RetrievalQA 链，集成压缩检索器。
2. 此链使用压缩和提取的信息来生成查询的答案。

## 此方法的优势

1. 提高相关性：系统仅返回与查询最相关的信息。
2. 提高效率：通过压缩和提取相关部分，减少了 LLM 需要处理的文本量。
3. 增强上下文理解：基于 LLM 的压缩器可以理解查询的上下文并相应地提取信息。
4. 灵活性：系统可以轻松适应不同类型的文档和查询。

## 结论

文档检索中的上下文压缩为增强信息检索系统的质量和效率提供了一种强大的方法。通过智能地提取和压缩相关信息，它为查询提供了更集中、更具上下文意识的响应。此方法在需要从大型文档集合中进行高效准确信息检索的各个领域都有潜在应用。

<div style="text-align: center;">

<img src="../images/contextual_compression.svg" alt="上下文压缩" style="width:70%; height:auto;">
</div>

# 包安装和导入

下面的单元格安装运行此笔记本所需的所有必要包。

```python
# 安装所需的包
!pip install langchain python-dotenv
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
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains import RetrievalQA


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
vector_store = encode_pdf(path)
```

### 创建检索器 + 上下文压缩器 + 组合它们

```python
# 创建检索器
retriever = vector_store.as_retriever()


#创建上下文压缩器
llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=4000)
compressor = LLMChainExtractor.from_llm(llm)

#将检索器与压缩器组合
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

# 创建带有压缩检索器的 QA 链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=compression_retriever,
    return_source_documents=True
)
```

### 示例用法

```python
query = "What is the main topic of the document?"
result = qa_chain.invoke({"query": query})
print(result["result"])
print("Source documents:", result["source_documents"])
```

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--contextual-compression)
