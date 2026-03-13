# 文档检索中的假设性文档嵌入 (HyDE)

## 概述

本代码实现了一个假设性文档嵌入 (HyDE) 系统用于文档检索。HyDE 是一种创新方法，将查询问题转换为包含答案的假设性文档，旨在弥合向量空间中查询与文档分布之间的差距。

## 动机

传统检索方法常难以处理短查询与较长、更详细文档之间的语义差距。HyDE 通过将查询扩展为完整的假设性文档来解决这一问题，通过使查询表示更类似于向量空间中的文档表示，潜在地提高检索相关性。

## 关键组件

1. PDF 处理和文本分块
2. 使用 FAISS 和 OpenAI embeddings 创建向量存储
3. 用于生成假设性文档的语言模型
4. 实现 HyDE 技术的自定义 HyDERetriever 类

## 方法详情

### 文档预处理和向量存储创建

1. 处理 PDF 并将其分割成块。
2. 使用 OpenAI embeddings 创建 FAISS 向量存储，以实现高效的相似性搜索。

### 假设性文档生成

1. 使用语言模型 (GPT-4) 生成回答给定查询的假设性文档。
2. 生成过程由提示模板引导，确保假设性文档详细且与向量存储中使用的块大小匹配。

### 检索过程

`HyDERetriever` 类实现以下步骤：

1. 使用语言模型从查询生成假设性文档。
2. 将假设性文档用作向量存储中的搜索查询。
3. 检索与此假设性文档最相似的文档。

## 关键特性

1. 查询扩展：将短查询转换为详细的假设性文档。
2. 灵活配置：允许调整块大小、重叠和检索文档数量。
3. 与 OpenAI 模型集成：使用 GPT-4 进行假设性文档生成，使用 OpenAI embeddings 进行向量表示。

## 此方法的优势

1. 提高相关性：通过将查询扩展为完整文档，HyDE 可以潜在地捕获更细致和相关的匹配。
2. 处理复杂查询：特别适用于可能难以直接匹配的复杂或多面查询。
3. 适应性：假设性文档生成可以适应不同类型的查询和文档领域。
4. 更好理解上下文的潜力：扩展的查询可能更好地捕获原始问题背后的上下文和意图。

## 实现细节

1. 使用 OpenAI 的 ChatGPT 模型进行假设性文档生成。
2. 采用 FAISS 在向量空间中进行高效的相似性搜索。
3. 允许轻松可视化假设性文档和检索结果。

## 结论

假设性文档嵌入 (HyDE) 代表了一种创新的文档检索方法，解决了查询与文档之间的语义差距。通过利用先进的语言模型将查询扩展为假设性文档，HyDE 有可能显著提高检索相关性，特别是对于复杂或细微的查询。这种技术在理解查询意图和上下文至关重要的领域（如法律研究、学术文献综述或高级信息检索系统）中特别有价值。

<div style="text-align: center;">

<img src="../images/HyDe.svg" alt="HyDe" style="width:40%; height:auto;">
</div>

<div style="text-align: center;">

<img src="../images/hyde-advantages.svg" alt="HyDe" style="width:100%; height:auto;">
</div>

# 包安装与导入

下面的单元格安装运行此 notebook 所需的所有必要包。


```python
# 安装所需的包
!pip install python-dotenv
```

```python
# 克隆仓库以访问辅助函数和评估模块
!git clone https://github.com/NirDiamant/RAG_TECHNIQUES.git
import sys
sys.path.append('RAG_TECHNIQUES')
# 如果需要运行最新数据
# !cp -r RAG_TECHNIQUES/data .
```

```python
import os
import sys
from dotenv import load_dotenv


# 原始路径追加已替换为 Colab 兼容版本
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

# 下载此笔记本中使用的 PDF 文档
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf

```

```python
path = "data/Understanding_Climate_Change.pdf"
```

### 定义 HyDE 检索器类 - 创建向量存储、生成假设性文档并检索

```python
class HyDERetriever:
    def __init__(self, files_path, chunk_size=500, chunk_overlap=100):
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=4000)

        self.embeddings = OpenAIEmbeddings()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore = encode_pdf(files_path, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)


        self.hyde_prompt = PromptTemplate(
            input_variables=["query", "chunk_size"],
            template="""给定问题'{query}'，生成直接回答此问题的假设性文档。文档应该详细且深入。
            文档大小必须恰好是{chunk_size}个字符。""",
        )
        self.hyde_chain = self.hyde_prompt | self.llm

    def generate_hypothetical_document(self, query):
        input_variables = {"query": query, "chunk_size": self.chunk_size}
        return self.hyde_chain.invoke(input_variables).content

    def retrieve(self, query, k=3):
        hypothetical_doc = self.generate_hypothetical_document(query)
        similar_docs = self.vectorstore.similarity_search(hypothetical_doc, k=k)
        return similar_docs, hypothetical_doc

```

### 创建 HyDE 检索器实例

```python
retriever = HyDERetriever(path)
```

### 用例演示

```python
test_query = "气候变化的主要原因是什么？"
results, hypothetical_doc = retriever.retrieve(test_query)
```

### 展示假设性文档和检索到的文档

```python
docs_content = [doc.page_content for doc in results]

print("假设性文档:\n")
print(text_wrap(hypothetical_doc)+"\n")
show_context(docs_content)
```

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--hyde-hypothetical-document-embedding)
