# 文档处理的语义分块

## 概述

此代码实现了一种用于处理和从 PDF 文档检索信息的语义分块方法，[由 Greg Kamradt 首次提出](https://youtu.be/8OJC21T2SL4?t=1933)，随后 [在 LangChain 中实现](https://python.langchain.com/docs/how_to/semantic-chunker/)。与基于固定字符或单词计数分割文本的传统方法不同，语义分块旨在创建更有意义和上下文感知的文本片段。

## 动机

传统的文本分割方法通常在任意点分割文档，可能会打断信息流和上下文。语义分块通过尝试在更自然的断点分割文本来解决这个问题，保持每个块内的语义连贯性。

## 关键组件

1. PDF 处理和文本提取
2. 使用 LangChain 的 SemanticChunker 进行语义分块
3. 使用 FAISS 和 OpenAI embeddings 创建向量存储
4. 用于查询处理文档的检索器设置

## 方法细节

### 文档预处理

1. 使用自定义的 `read_pdf_to_string` 函数将 PDF 读取并转换为字符串。

### 语义分块

1. 使用 LangChain 的 `SemanticChunker` 配合 OpenAI embeddings。
2. 提供三种断点类型：
   - 'percentile'：在大于 X 百分位的差异处分割。
   - 'standard_deviation'：在大于 X 个标准差的差异处分割。
   - 'interquartile'：使用四分位距离确定分割点。
3. 在此实现中，使用'percentile'方法，阈值为 90。

### 向量存储创建

1. 使用 OpenAI embeddings 为语义块创建向量表示。
2. 从这些 embeddings 创建 FAISS 向量存储以进行高效的相似性搜索。

### 检索器设置

1. 检索器配置为获取给定查询的前 2 个最相关块。

## 关键特性

1. 上下文感知分割：尝试保持块内的语义连贯性。
2. 灵活配置：允许不同的断点类型和阈值。
3. 与高级 NLP 工具集成：在分块和检索中都使用 OpenAI embeddings。

## 此方法的优势

1. 改善连贯性：块更可能包含完整的想法或概念。
2. 更好的检索相关性：通过保留上下文，检索准确性可能会提高。
3. 适应性：可以根据文档的性质和检索需求调整分块方法。
4. 更好理解的潜力：LLM 或下游任务可能在更连贯的文本片段上表现更好。

## 实现细节

1. 在语义分块过程和最终向量表示中都使用 OpenAI 的 embeddings。
2. 采用 FAISS 创建块的高效可搜索索引。
3. 检索器设置为返回前 2 个最相关的块，可以根据需要调整。

## 示例用法

代码包含一个测试查询："What is the main cause of climate change?"（气候变化的主要原因是什么？）。这演示了如何使用语义分块和检索系统从处理后的文档中查找相关信息。

## 结论

语义分块代表了检索系统文档处理的高级方法。通过尝试在文本片段内保持语义连贯性，它有可能改善检索信息的质量并增强下游 NLP 任务的性能。此技术对于处理保持上下文至关重要的长篇、复杂文档（如科学论文、法律文件或综合报告）特别有价值。

<div style="text-align: center;">

<img src="../images/semantic_chunking_comparison.svg" alt="Self RAG" style="width:100%; height:auto;">
</div>

# 包安装和导入

下面的单元格安装运行此笔记本所需的所有必要包。

```python
# 安装所需的包
!pip install langchain-experimental langchain-openai python-dotenv
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

# 原始路径追加已替换为 Colab 兼容性
from helper_functions import *
from evaluation.evalute_rag import *

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

# 从 .env 文件加载环境变量
load_dotenv()

# 设置 OpenAI API 密钥环境变量
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
```

### 定义文件路径

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

### 将 PDF 读取为字符串

```python
content = read_pdf_to_string(path)
```

### 断点类型：
* 'percentile'：计算句子之间的所有差异，然后任何大于 X 百分位的差异都会触发分割。
* 'standard_deviation'：任何大于 X 个标准差的差异都会触发分割。
* 'interquartile'：使用四分位距离来分割块。

```python
text_splitter = SemanticChunker(OpenAIEmbeddings(), breakpoint_threshold_type='percentile', breakpoint_threshold_amount=90) # 选择要使用的嵌入和断点类型及阈值
```

### 将原始文本分割为语义块

```python
docs = text_splitter.create_documents([content])
```

### 创建向量存储和检索器

```python
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
chunks_query_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
```

### 测试检索器

```python
test_query = "What is the main cause of climate change?"
context = retrieve_context_per_question(test_query, chunks_query_retriever)
show_context(context)
```

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--semantic-chunking)
