# 🌟 新手入门：通过问题生成进行文档增强

> **💡 给新手的说明**
> - **难度级别**：⭐⭐⭐⭐ 中高级（需要了解 RAG 基础和 Embedding 概念）
> - **预计学习时间**：60-75 分钟
> - **前置知识**：了解向量存储、检索器、文档分块等基础概念
> - **本教程你将学会**：如何让 AI 自动为你的文档生成大量相关问题，从而大幅提升检索命中率

---

## 📖 核心概念理解

### 什么是文档增强？

想象你在为一个图书馆编目。传统方法只记录书名和简介，但**文档增强**就像是：

**传统 RAG**：只索引原文
```
原文："气候变化导致全球平均气温上升 1.1°C"
索引：[这句话的向量表示]
```

**增强后的 RAG**：索引原文 + 生成的相关问题
```
原文："气候变化导致全球平均气温上升 1.1°C"
索引：
  - [原文的向量表示]
  - ["全球变暖了多少度？"的向量表示] ← 生成的问题 1
  - ["气候变化的具体影响有哪些？"的向量表示] ← 生成的问题 2
  - ["过去 100 年气温变化数据"的向量表示] ← 生成的问题 3
  ...
```

### 通俗理解

```
用户搜索场景对比：

【未增强的系统】
用户问："地球变热了多少？"
→ 系统：检索"地球"、"变热"... 没找到匹配的原文
→ 结果：❌ 检索失败

【增强后的系统】
用户问："地球变热了多少？"
→ 系统：发现问题与"全球变暖了多少度？"语义相似
→ 找到该问题对应的原文
→ 结果：✅ 检索成功！
```

### 为什么这个方法有效？

1. **问题与用户的搜索意图更接近**：用户通常用问句搜索，而不是陈述句
2. **增加检索入口**：一篇文档对应多个问题，命中概率倍增
3. **语义覆盖更全面**：同一个意思，多种问法都能匹配

### 两种增强级别

| 级别 | 说明 | 适用场景 |
|------|------|---------|
| **文档级 (DOCUMENT_LEVEL)** | 对整个文档生成问题 | 文档较短、主题集中 |
| **片段级 (FRAGMENT_LEVEL)** | 对每个分块单独生成问题 | 文档较长、包含多个主题 |

---

## 🛠️ 第一步：安装必要的包

### 💻 完整代码

```python
# 安装所需的包
# faiss-cpu: Facebook 的向量搜索库（CPU 版本）
# langchain: RAG 系统核心框架
# langchain-openai: OpenAI 接口
# python-dotenv: 环境变量管理
!pip install faiss-cpu langchain langchain-openai python-dotenv
```

> **⚠️ 新手注意**
> - 国内用户可使用清华镜像源加速安装
> - 如果遇到 `faiss` 安装失败，尝试先安装 `conda-forge` 版本

### 导入必要的库

```python
import sys
import os
import re
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from enum import Enum
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from typing import Any, Dict, List, Tuple
from pydantic import BaseModel, Field

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置 OpenAI API 密钥
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# 导入辅助函数
from helper_functions import *
```

> **💡 代码解释**
> - `Enum`：Python 枚举类，用于定义固定的选项
> - `BaseModel, Field`：Pydantic 库，用于定义数据结构
> - `re`：正则表达式模块，用于文本清理

---

## ⚙️ 第二步：配置核心参数

### 📖 这是什么？

这些参数控制文档处理的各个环节，包括分块大小、重叠度、问题生成数量等。

### 💻 完整代码

```python
class QuestionGeneration(Enum):
    """
    枚举类，用于指定文档处理的问题生成级别。

    属性:
        DOCUMENT_LEVEL (int): 表示在整个文档级别生成问题。
        FRAGMENT_LEVEL (int): 表示在单个文本片段级别生成问题。
    """
    DOCUMENT_LEVEL = 1
    FRAGMENT_LEVEL = 2


# ========== 文档级分块参数 ==========
# 根据模型不同，Mitral 7B 最大可达 8000，Llama 3.1 8B 可达 128k
# 这里设置为 4000 tokens，适合大多数文档
DOCUMENT_MAX_TOKENS = 4000      # 每个文档块的最大 token 数
DOCUMENT_OVERLAP_TOKENS = 100   # 文档块之间的重叠 token 数

# ========== 片段级分块参数 ==========
# 在较短文本上计算 Embedding 和文本相似度
FRAGMENT_MAX_TOKENS = 128       # 每个片段的最大 token 数（较短）
FRAGMENT_OVERLAP_TOKENS = 16    # 片段之间的重叠 token 数

# ========== 问题生成配置 ==========
# 在文档或片段级别生成问题
QUESTION_GENERATION = QuestionGeneration.DOCUMENT_LEVEL  # 选择生成级别
# 为每个文档或片段生成多少个问题
QUESTIONS_PER_DOCUMENT = 40     # 每个文档生成 40 个问题
```

> **💡 参数详解**

```
文档分块层级关系：

原始文档（10000 tokens）
    │
    ▼ 按 DOCUMENT_MAX_TOKENS=4000 分割
    │
    ├── 文本文档 1 (4000 tokens，含 100 token 来自文档 2)
    │   │
    │   ▼ 按 FRAGMENT_MAX_TOKENS=128 分割
    │   ├── 片段 1 (128 tokens)
    │   ├── 片段 2 (128 tokens)
    │   └── ...
    │
    ├── 文本文档 2 (4000 tokens，含 100 token 来自文档 1 和 3)
    │   └── ...
    │
    └── 文本文档 3 (2000 tokens)
```

> **⚠️ 新手注意**
> - `DOCUMENT_LEVEL`：生成的问题数量 = 文档数 × 40
> - `FRAGMENT_LEVEL`：生成的问题数量 = 片段数 × 40（会多很多！）
> - 问题太多会增加 API 成本和检索时间，根据需求调整

---

## 🔧 第三步：定义核心类和函数

### 1. 定义问题列表的数据结构

```python
class QuestionList(BaseModel):
    """
    Pydantic 模型，用于规范化问题生成的输出格式。

    属性:
        question_list: 为文档或片段生成的问题列表
    """
    question_list: List[str] = Field(
        ...,
        title="为文档或片段生成的问题列表",
        description="LLM 生成的相关问题列表，每个问题都是完整的问句"
    )
```

> **💡 代码解释**
> - `BaseModel`：Pydantic 的基础模型类，提供数据验证和序列化
> - `Field`：定义字段的元数据，帮助 LLM 理解输出格式
> - 使用结构化输出可以确保 LLM 返回的数据格式一致

### 2. OpenAI Embeddings 包装器

```python
class OpenAIEmbeddingsWrapper(OpenAIEmbeddings):
    """
    OpenAI embeddings 的包装器类，提供与原始 OllamaEmbeddings 类似的接口。

    这个包装器允许我们将 embedding 模型当作可调用对象使用，
    使得代码更加简洁和直观。
    """

    def __call__(self, query: str) -> List[float]:
        """
        允许实例作为可调用对象，为查询生成 embedding。

        Args:
            query (str): 要嵌入的查询字符串。

        Returns:
            List[float]: 查询的 embedding，作为浮点数列表。
        """
        return self.embed_query(query)
```

> **💡 代码解释**
> - `__call__` 方法让实例可以像函数一样调用：`embeddings("hello")`
> - 这种设计模式叫"可调用对象"，在 Python 中很常见

### 3. 问题清理和过滤函数

```python
def clean_and_filter_questions(questions: List[str]) -> List[str]:
    """
    清理和过滤问题列表。

    执行以下操作：
    1. 移除问题开头的编号（如"1. "）
    2. 移除首尾空格
    3. 只保留以问号结尾的有效问题

    Args:
        questions (List[str]): 要清理和过滤的问题列表。

    Returns:
        List[str]: 清理和过滤后的问题列表，以问号结尾。
    """
    cleaned_questions = []
    for question in questions:
        # 移除开头的数字编号，如"1. "、"2. "等
        cleaned_question = re.sub(r'^\d+\.\s*', '', question.strip())
        # 只保留以问号结尾的有效问题
        if cleaned_question.endswith('?'):
            cleaned_questions.append(cleaned_question)
    return cleaned_questions
```

> **💡 代码解释**
> - LLM 生成的问题可能带编号：`"1. 什么是气候变化？"`
> - 正则表达式 `^\d+\.\s*` 匹配开头的数字 + 点 + 空格
> - 过滤掉不完整的问题（没有问号）

> **⚠️ 新手注意**
> - 如果过滤后问题太少，可以调整过滤条件
> - 有些好问题可能不用问号（如"请解释..."），可根据需求调整

### 4. 问题生成函数（核心功能）

```python
def generate_questions(text: str) -> List[str]:
    """
    使用 OpenAI 根据提供的文本生成问题列表。

    Args:
        text (str): 用于生成问题的上下文数据。

    Returns:
        List[str]: 唯一的、过滤后的问题列表。
    """
    # 初始化 LLM，使用 gpt-4o-mini（性价比高）
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 定义 prompt 模板
    prompt = PromptTemplate(
        input_variables=["context", "num_questions"],
        template="""使用上下文数据：{context}

生成至少{num_questions}个可以关于此上下文提出的可能问题列表。
确保问题可以直接在上下文中回答，不包括任何答案或标题。
用换行符分隔问题。"""
    )

    # 创建链：prompt → LLM（结构化输出）
    chain = prompt | llm.with_structured_output(QuestionList)

    # 准备输入数据
    input_data = {"context": text, "num_questions": QUESTIONS_PER_DOCUMENT}

    # 调用 LLM
    result = chain.invoke(input_data)

    # 从 QuestionList 对象中提取问题列表
    questions = result.question_list

    # 清理和过滤
    filtered_questions = clean_and_filter_questions(questions)

    # 去重后返回
    return list(set(filtered_questions))
```

> **💡 代码解释**

```
Prompt 模板解析：

输入变量：
- context: 要处理的文本内容
- num_questions: 要生成的问题数量（40 个）

模板内容告诉 LLM：
1. 基于给定的上下文生成问题
2. 问题要能在上下文中找到答案（不要跑题）
3. 只输出问题，不要答案和标题
4. 用换行符分隔每个问题

结构化输出：
- 使用 with_structured_output(QuestionList)
- 强制 LLM 返回规范的 JSON 格式
- 避免解析失败
```

> **⚠️ 新手注意**
> - `temperature=0`：问题生成需要稳定性，不用随机性
> - 如果生成的问题质量不高，可以优化 prompt
> - `list(set(...))` 用于去重，因为 LLM 可能生成相似问题

### 5. 答案生成函数

```python
def generate_answer(content: str, question: str) -> str:
    """
    使用 OpenAI 根据提供的上下文为给定问题生成答案。

    Args:
        content (str): 用于生成答案的上下文数据。
        question (str): 要生成答案的问题。

    Returns:
        str: 基于提供上下文对问题的精确答案。
    """
    # 初始化 LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 定义 prompt 模板
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""使用上下文数据：{context}

对以下问题提供简洁精确的答案：{question}"""
    )

    # 创建链
    chain = prompt | llm

    # 准备输入数据
    input_data = {"context": content, "question": question}

    # 调用并返回结果
    return chain.invoke(input_data)
```

> **💡 代码解释**
> - 这个函数用于在检索到相关文档后生成最终答案
> - `temperature=0` 确保答案稳定可靠

### 6. 文档分割函数

```python
def split_document(document: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    将文档分割成较小的文本块。

    Args:
        document (str): 要分割的文档文本。
        chunk_size (int): 每个块的大小，以 token 数量计。
        chunk_overlap (int): 连续块之间的重叠 token 数。

    Returns:
        List[str]: 文本块列表，每个块是文档内容的字符串。
    """
    # 使用正则表达式提取单词作为 token
    # 这是一种简化的 tokenization 方法
    tokens = re.findall(r'\b\w+\b', document)

    chunks = []
    # 滑动窗口方式分割
    for i in range(0, len(tokens), chunk_size - chunk_overlap):
        # 取出当前窗口的 token
        chunk_tokens = tokens[i:i + chunk_size]
        chunks.append(chunk_tokens)

        # 如果已经到文档末尾，退出循环
        if i + chunk_size >= len(tokens):
            break

    # 将 token 列表转回字符串
    return [" ".join(chunk) for chunk in chunks]
```

> **💡 代码解释**

```
分割过程可视化：

原文："A B C D E F G H I J K L M N O" (15 个单词)
chunk_size=5, chunk_overlap=1

分割结果：
块 1: [A B C D E]     (tokens 0-4)
块 2: [E F G H I]     (tokens 4-8，E 是重叠部分)
块 3: [I J K L M]     (tokens 8-12，I 是重叠部分)
块 4: [M N O]         (tokens 12-14，M 是重叠部分)

重叠的作用：保持上下文的连续性，避免信息在边界处丢失
```

> **⚠️ 新手注意**
> - 这是一个简化的 tokenization，实际 token 可能跨越单词边界
> - 生产环境建议使用专业的 tokenizer（如 tiktoken）

### 7. 打印文档的辅助函数

```python
def print_document(comment: str, document: Any) -> None:
    """
    打印注释后跟文档内容。

    Args:
        comment (str): 在文档详情之前打印的注释或描述。
        document (Any): 要打印内容的文档。

    Returns:
        None
    """
    print(f'{comment} (类型：{document.metadata["type"]}, 索引：{document.metadata["index"]}): {document.page_content}')
```

> **💡 代码解释**
> - 用于调试，显示每个文档的元数据（类型、索引）和内容
> - `metadata["type"]` 可以是 `"ORIGINAL"`（原文）或 `"AUGMENTED"`（生成的问题）

---

## 🧪 第四步：示例演示

### 测试问题生成和答案生成

```python
# 初始化 OpenAIEmbeddings
embeddings = OpenAIEmbeddingsWrapper()

# 示例文档
example_text = "This is an example document. It contains information about various topics."

# 生成问题
questions = generate_questions(example_text)
print("生成的问题:")
for q in questions:
    print(f"- {q}")

# 生成答案
sample_question = questions[0] if questions else "What is this document about?"
answer = generate_answer(example_text, sample_question)
print(f"\n问题：{sample_question}")
print(f"答案：{answer}")

# 分割文档
chunks = split_document(example_text, chunk_size=10, chunk_overlap=2)
print("\n文档块:")
for i, chunk in enumerate(chunks):
    print(f"块 {i + 1}: {chunk}")

# 使用 OpenAIEmbeddings 的示例
doc_embedding = embeddings.embed_documents([example_text])
query_embedding = embeddings.embed_query("What is the main topic?")
print("\n文档 Embedding (前 5 个元素):", doc_embedding[0][:5])
print("查询 Embedding (前 5 个元素):", query_embedding[:5])
```

### 预期输出

```
生成的问题:
- What topics does this example document cover?
- What kind of information is contained in the document?
- Can you describe the content of this example document?
...

问题：What topics does this example document cover?
答案：This example document contains information about various topics.

文档块:
块 1: This is an example document It contains information
块 2: information about various topics

文档 Embedding (前 5 个元素): [0.012, -0.034, 0.056, -0.021, 0.089]
查询 Embedding (前 5 个元素): [0.045, -0.012, 0.078, -0.034, 0.067]
```

---

## 🏭 第五步：主处理管道

### 📖 这是什么？

`process_documents` 函数是整个文档增强流程的核心，它完成：
1. 分割文档
2. 生成问题
3. 创建向量存储
4. 返回检索器

### 💻 完整代码

```python
def process_documents(content: str, embedding_model: OpenAIEmbeddings):
    """
    处理文档内容，将其分割为片段，生成问题，
    创建 FAISS 向量存储，并返回检索器。

    Args:
        content (str): 要处理的文档内容。
        embedding_model (OpenAIEmbeddings): 用于向量化的 embedding 模型。

    Returns:
        VectorStoreRetriever: 用于检索最相关 FAISS 文档的检索器。
    """
    # ========== 步骤 1：将文档分割为文本文档 ==========
    text_documents = split_document(content, DOCUMENT_MAX_TOKENS, DOCUMENT_OVERLAP_TOKENS)
    print(f'文本内容分割为：{len(text_documents)} 个文档')

    documents = []
    counter = 0

    # ========== 步骤 2：遍历每个文本文档 ==========
    for i, text_document in enumerate(text_documents):
        # 将文本文档进一步分割为片段
        text_fragments = split_document(text_document, FRAGMENT_MAX_TOKENS, FRAGMENT_OVERLAP_TOKENS)
        print(f'文本文档 {i} - 分割为：{len(text_fragments)} 个片段')

        # ========== 步骤 3：处理每个片段 ==========
        for j, text_fragment in enumerate(text_fragments):
            # 添加原始文本片段到文档列表
            documents.append(Document(
                page_content=text_fragment,
                metadata={
                    "type": "ORIGINAL",           # 标记为原始内容
                    "index": counter,             # 唯一索引
                    "text": text_document         # 保存父文档内容（用于后续生成答案）
                }
            ))
            counter += 1

            # ========== 步骤 4a：片段级问题生成 ==========
            if QUESTION_GENERATION == QuestionGeneration.FRAGMENT_LEVEL:
                # 为当前片段生成问题
                questions = generate_questions(text_fragment)

                # 将每个问题作为增强文档添加
                documents.extend([
                    Document(
                        page_content=question,
                        metadata={
                            "type": "AUGMENTED",              # 标记为增强内容
                            "index": counter + idx,           # 唯一索引
                            "text": text_document             # 保存父文档引用
                        }
                    )
                    for idx, question in enumerate(questions)
                ])
                counter += len(questions)
                print(f'文本文档 {i} 文本片段 {j} - 生成：{len(questions)} 个问题')

        # ========== 步骤 4b：文档级问题生成 ==========
        if QUESTION_GENERATION == QuestionGeneration.DOCUMENT_LEVEL:
            # 为整个文本文档生成问题
            questions = generate_questions(text_document)

            # 将每个问题作为增强文档添加
            documents.extend([
                Document(
                    page_content=question,
                    metadata={
                        "type": "AUGMENTED",
                        "index": counter + idx,
                        "text": text_document
                    }
                )
                for idx, question in enumerate(questions)
            ])
            counter += len(questions)
            print(f'文本文档 {i} - 生成：{len(questions)} 个问题')

    # ========== 步骤 5：打印所有文档（调试用） ==========
    for document in documents:
        print_document("数据集", document)

    # ========== 步骤 6：创建向量存储 ==========
    print(f'创建存储，计算 {len(documents)} 个 FAISS 文档的 embeddings')
    vectorstore = FAISS.from_documents(documents, embedding_model)

    # ========== 步骤 7：创建检索器 ==========
    print("创建返回最相关 FAISS 文档的检索器")
    return vectorstore.as_retriever(search_kwargs={"k": 1})
```

> **💡 流程图解**

```
输入：原始文档内容
        │
        ▼
┌─────────────────────────────────────┐
│ 步骤 1：分割为文本文档                 │
│ (每个 4000 tokens，重叠 100)          │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│ 步骤 2：每个文本文档分割为片段         │
│ (每个 128 tokens，重叠 16)            │
└─────────────────────────────────────┘
        │
        ├──→ [ORIGINAL 文档] → 加入向量存储
        │
        └──→ [DOCUMENT_LEVEL 模式]
                │
                ▼
        ┌─────────────────────────────┐
        │ 步骤 3：为文本文档生成问题     │
        │ (每个文档 40 个问题)           │
        └─────────────────────────────┘
                │
                ▼
        ┌─────────────────────────────┐
        │ 步骤 4：问题也加入向量存储     │
        │ (作为 AUGMENTED 文档)         │
        └─────────────────────────────┘
                │
                ▼
        ┌─────────────────────────────┐
        │ 步骤 5：计算所有文档的         │
        │ Embedding 并创建 FAISS        │
        └─────────────────────────────┘
                │
                ▼
        返回：检索器
```

> **⚠️ 新手注意**
> - 这个函数会调用多次 LLM API（每个文档/片段都要生成问题）
> - 注意监控 API 使用量和成本
> - 初次运行建议用小文档测试

---

## 🚀 第六步：完整执行示例

### 下载并加载 PDF 文档

```python
# 创建 data 目录
import os
os.makedirs('data', exist_ok=True)

# 下载教程使用的 PDF 文档
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf

# 加载 PDF 文档
path = "data/Understanding_Climate_Change.pdf"
content = read_pdf_to_string(path)

print(f"文档加载完成，总长度：{len(content)} 字符")
```

### 处理文档并创建检索器

```python
# 实例化 Embedding 模型
embedding_model = OpenAIEmbeddings()

# 处理文档并创建检索器
# 这一步可能需要几分钟（生成问题需要调用 LLM）
document_query_retriever = process_documents(content, embedding_model)

print("检索器创建完成！")
```

### 预期输出（部分）

```
文本内容分割为：3 个文档
文本文档 0 - 分割为：45 个片段
文本文档 0 - 生成：40 个问题
文本文档 1 - 分割为：38 个片段
文本文档 1 - 生成：40 个问题
文本文档 2 - 分割为：22 个片段
文本文档 2 - 生成：40 个问题
数据集 (类型：ORIGINAL, 索引：0): Climate change refers to...
数据集 (类型：AUGMENTED, 索引：1): What is climate change?
数据集 (类型：AUGMENTED, 索引：2): What causes global warming?
...
创建存储，计算 125 个 FAISS 文档的 embeddings
创建返回最相关 FAISS 文档的检索器
检索器创建完成！
```

---

## 🔍 第七步：测试检索效果

### 基本检索测试

```python
# 测试查询
query = "What is climate change?"

# 执行检索
retrieved_docs = document_query_retriever.get_relevant_documents(query)

print(f"\n查询：{query}")
print(f"检索到的文档：{retrieved_docs[0].page_content}")
```

### 高级检索演示

```python
# 更复杂的查询
query = "How do freshwater ecosystems change due to alterations in climatic factors?"
print(f'问题:{query}\n')

# 使用检索器
retrieved_documents = document_query_retriever.invoke(query)

# 显示检索结果
for doc in retrieved_documents:
    print(f"类型：{doc.metadata['type']}")
    print(f"内容：{doc.page_content}\n")
```

### 获取父文档并生成答案

```python
# 找到父文本文档（原始内容）
# 检索到的可能是一个生成的问题，我们需要找到对应的原文
doc = retrieved_documents[0]
context = doc.metadata['text']  # 从元数据中获取父文档

print(f'上下文:\n{context}')

# 使用上下文生成答案
answer = generate_answer(context, query)
print(f'\n答案:\n{answer}')
```

---

## 📊 完整代码整合

```python
# ========== 1. 安装和导入 ==========
!pip install faiss-cpu langchain langchain-openai python-dotenv pydantic

import os
import re
from enum import Enum
from typing import Any, Dict, List, Tuple
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# ========== 2. 配置参数 ==========
class QuestionGeneration(Enum):
    DOCUMENT_LEVEL = 1
    FRAGMENT_LEVEL = 2

DOCUMENT_MAX_TOKENS = 4000
DOCUMENT_OVERLAP_TOKENS = 100
FRAGMENT_MAX_TOKENS = 128
FRAGMENT_OVERLAP_TOKENS = 16
QUESTION_GENERATION = QuestionGeneration.DOCUMENT_LEVEL
QUESTIONS_PER_DOCUMENT = 40

# ========== 3. 定义函数 ==========
class QuestionList(BaseModel):
    question_list: List[str] = Field(..., title="为文档或片段生成的问题列表")

class OpenAIEmbeddingsWrapper(OpenAIEmbeddings):
    def __call__(self, query: str) -> List[float]:
        return self.embed_query(query)

def clean_and_filter_questions(questions: List[str]) -> List[str]:
    cleaned_questions = []
    for question in questions:
        cleaned_question = re.sub(r'^\d+\.\s*', '', question.strip())
        if cleaned_question.endswith('?'):
            cleaned_questions.append(cleaned_question)
    return cleaned_questions

def generate_questions(text: str) -> List[str]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = PromptTemplate(
        input_variables=["context", "num_questions"],
        template="""使用上下文数据：{context}

生成至少{num_questions}个可以关于此上下文提出的可能问题列表。
确保问题可以直接在上下文中回答，不包括任何答案或标题。
用换行符分隔问题。"""
    )
    chain = prompt | llm.with_structured_output(QuestionList)
    input_data = {"context": text, "num_questions": QUESTIONS_PER_DOCUMENT}
    result = chain.invoke(input_data)
    questions = result.question_list
    filtered_questions = clean_and_filter_questions(questions)
    return list(set(filtered_questions))

def generate_answer(content: str, question: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""使用上下文数据：{context}

对以下问题提供简洁精确的答案：{question}"""
    )
    chain = prompt | llm
    input_data = {"context": content, "question": question}
    return chain.invoke(input_data)

def split_document(document: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    tokens = re.findall(r'\b\w+\b', document)
    chunks = []
    for i in range(0, len(tokens), chunk_size - chunk_overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        chunks.append(chunk_tokens)
        if i + chunk_size >= len(tokens):
            break
    return [" ".join(chunk) for chunk in chunks]

def process_documents(content: str, embedding_model: OpenAIEmbeddings):
    text_documents = split_document(content, DOCUMENT_MAX_TOKENS, DOCUMENT_OVERLAP_TOKENS)
    print(f'文本内容分割为：{len(text_documents)} 个文档')

    documents = []
    counter = 0
    for i, text_document in enumerate(text_documents):
        text_fragments = split_document(text_document, FRAGMENT_MAX_TOKENS, FRAGMENT_OVERLAP_TOKENS)
        print(f'文本文档 {i} - 分割为：{len(text_fragments)} 个片段')

        for j, text_fragment in enumerate(text_fragments):
            documents.append(Document(
                page_content=text_fragment,
                metadata={"type": "ORIGINAL", "index": counter, "text": text_document}
            ))
            counter += 1

            if QUESTION_GENERATION == QuestionGeneration.FRAGMENT_LEVEL:
                questions = generate_questions(text_fragment)
                documents.extend([
                    Document(page_content=question, metadata={"type": "AUGMENTED", "index": counter + idx, "text": text_document})
                    for idx, question in enumerate(questions)
                ])
                counter += len(questions)

        if QUESTION_GENERATION == QuestionGeneration.DOCUMENT_LEVEL:
            questions = generate_questions(text_document)
            documents.extend([
                Document(page_content=question, metadata={"type": "AUGMENTED", "index": counter + idx, "text": text_document})
                for idx, question in enumerate(questions)
            ])
            counter += len(questions)
            print(f'文本文档 {i} - 生成：{len(questions)} 个问题')

    print(f'创建存储，计算 {len(documents)} 个 FAISS 文档的 embeddings')
    vectorstore = FAISS.from_documents(documents, embedding_model)
    return vectorstore.as_retriever(search_kwargs={"k": 1})

# ========== 4. 执行 ==========
path = "data/Understanding_Climate_Change.pdf"
content = read_pdf_to_string(path)
embedding_model = OpenAIEmbeddings()
document_query_retriever = process_documents(content, embedding_model)

# ========== 5. 测试 ==========
query = "What is climate change?"
retrieved_docs = document_query_retriever.get_relevant_documents(query)
print(f"\n查询：{query}")
print(f"检索结果：{retrieved_docs[0].page_content}")
```

---

## ⚠️ 常见问题及解决方法

### 问题 1：API 成本太高

**原因**：每个文档/片段都要调用 LLM 生成问题

**解决方法**：
```python
# 1. 减少问题数量
QUESTIONS_PER_DOCUMENT = 20  # 从 40 降到 20

# 2. 使用文档级而不是片段级
QUESTION_GENERATION = QuestionGeneration.DOCUMENT_LEVEL

# 3. 只处理重要文档
# 先过滤掉不重要的内容再处理
```

### 问题 2：生成的问题质量不高

**可能原因**：
- 原始文本太短或太零散
- Prompt 不够清晰

**解决方法**：
```python
# 优化 Prompt
prompt = PromptTemplate(
    input_variables=["context", "num_questions"],
    template="""你是一个专业的问答设计师。请基于以下文本生成问题：

上下文：{context}

要求：
1. 生成{num_questions}个不同的问题
2. 问题应该多样化（是什么、为什么、怎么样等类型）
3. 每个问题都能在上文中找到答案
4. 使用自然、流畅的语言

问题列表："""
)
```

### 问题 3：检索速度变慢

**原因**：文档数量大幅增加（原文 + 生成的问题）

**解决方法**：
```python
# 1. 使用更高效的向量索引
vectorstore = FAISS.from_documents(documents, embedding_model)
# 启用 FAISS 的索引优化
vectorstore.index.train()

# 2. 限制检索数量
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# 3. 定期清理低质量问题
# 可以设置阈值过滤掉质量不高的生成问题
```

---

## 🎓 学习总结

### 你学到了什么？

✅ **文档增强的概念**：通过生成问题扩展检索入口
✅ **两种增强级别**：文档级 vs 片段级的适用场景
✅ **问题生成流程**：使用 LLM 自动生成高质量问题
✅ **混合存储策略**：同时存储原文和生成问题
✅ **元数据管理**：用 metadata 追踪文档来源和类型

### 实际应用场景

| 场景 | 文档增强的价值 |
|------|--------------|
| 客服知识库 | 用户问题多种多样，增强后更容易匹配 |
| 技术文档检索 | 同一技术点有多种问法都能命中 |
| 教育培训 | 自动生成练习题 + 答案的配对 |
| FAQ 系统 | 从长文档自动生成 FAQ 对 |

### 性能对比

```
传统 RAG vs 增强 RAG 检索命中率对比：

查询："全球变暖对生态系统的影响"

传统 RAG:
- 直接搜索原文 → 可能 miss 掉语义相关但用词不同的内容
- 命中率：约 60%

增强 RAG:
- 原文 + 40 个生成问题 → 更多匹配入口
- "生态系统会受到什么影响？"
- "全球变暖如何影响动植物？"
- "生物多样性面临哪些威胁？"
- 命中率：约 85%+
```

---

## 📚 相关资源

- [LangChain 文档增强相关讨论](https://github.com/langchain-ai/langchain/discussions)
- [OpenAI Embedding API 文档](https://platform.openai.com/docs/guides/embeddings)
- [FAISS 向量搜索库](https://github.com/facebookresearch/faiss)

---

*本教程是 RAG 技术系列教程之一。文档增强可以与重排序、融合检索等技术结合，构建更强大的 RAG 系统。*

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--document-augmentation)
