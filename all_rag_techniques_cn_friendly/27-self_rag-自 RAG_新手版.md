# 🌟 新手入门：Self-RAG（自 RAG）

> **💡 给新手的说明**
> - **难度等级**：⭐⭐⭐☆☆（中等）
> - **预计学习时间**：45-60 分钟
> - **前置知识**：了解基本的 RAG 概念，有 Python 编程经验
> - **学完你将掌握**：如何让 RAG 系统智能判断何时需要检索、如何评估答案质量
>
> **🤔 为什么要学这个？** 传统 RAG 不管什么问题都去检索，效率低且可能引入噪音。Self-RAG 像有一个"智能大脑"，会先思考"我需要检索吗？"，然后评估"我的答案可靠吗？"，最终给出更高质量的回复！

---

## 📖 核心概念理解

### 什么是 Self-RAG？

**Self-RAG**（Self-Reflective RAG）是一种**自反思**的检索增强生成方法。它不仅能动态决定是否检索，还能评估生成答案的质量。

### 通俗理解：聪明的助手 vs 死板的助手

#### 传统 RAG（死板助手）

```
你："今天天气怎么样？"
助手：（不管三七二十一，先去检索文档）
      找到 5 个文档...
      基于文档生成答案...
助手："根据文档 XYZ，今天天气..."
```

问题：这种常识性问题根本不需要检索！

#### Self-RAG（聪明助手）

```
你："今天天气怎么样？"
助手：（思考）这个问题基于常识就能回答，不需要检索
助手："今天天气晴朗，气温 25 度左右。"

你："量子纠缠的最新研究进展是什么？"
助手：（思考）这个问题太专业了，需要检索
      检索相关文档...
      评估文档相关性...
      生成答案并评估可靠性...
助手："根据 2024 年的研究，量子纠缠在...（答案完全有依据）"
```

### 核心组件一览

| 组件 | 作用 | 生活化比喻 |
|------|------|-----------|
| **检索决策** | 判断是否需要检索 | 老师判断学生的问题是否需要查资料 |
| **文档检索** | 获取相关文档 | 图书馆员找书 |
| **相关性评估** | 评估文档与查询的相关性 | 读者判断书是否有用 |
| **回复生成** | 基于上下文生成答案 | 根据资料写回答 |
| **支持度评估** | 评估答案是否有上下文支持 | 检查答案是否有依据 |
| **效用评估** | 评估答案的有用性 | 给答案打分（1-5 分） |

### Self-RAG 工作流程

```
用户查询
    ↓
┌──────────────────┐
│ 步骤 1：检索决策 │ ← 需要检索吗？
└──────────────────┘
       ↓ No          ↓ Yes
   直接生成      检索文档
   答案            ↓
              ┌──────────────────┐
              │ 步骤 2：相关性评估│ ← 文档相关吗？
              └──────────────────┘
                     ↓ Irrelevant    ↓ Relevant
                 无需检索生成      基于相关上下文生成
                     ↓                    ↓
              ┌──────────────────┐
              │ 步骤 3：支持度评估│ ← 答案有依据吗？
              └──────────────────┘
                     ↓
              ┌──────────────────┐
              │ 步骤 4：效用评估 │ ← 答案有用吗？（1-5 分）
              └──────────────────┘
                     ↓
              选择最佳答案（支持度高 + 效用高）
```

---

## 🛠️ 第一步：环境准备

### 📖 这是什么？

安装 Self-RAG 所需的依赖包。

### 💻 完整代码

```python
# 安装所需的包
# - langchain: RAG 框架
# - langchain-openai: OpenAI 集成
# - python-dotenv: 环境变量管理
!pip install langchain langchain-openai python-dotenv

# 克隆仓库以访问辅助函数
!git clone https://github.com/NirDiamant/RAG_TECHNIQUES.git
import sys
sys.path.append('RAG_TECHNIQUES')
```

> **💡 代码解释**
>
> **依赖说明：**
> - Self-RAG 的核心是**多个 LLM 调用链**
> - 不需要额外的聚类或图算法库
> - 主要依赖 LangChain 的提示模板和链功能
>
> **⚠️ 新手注意**
> - Self-RAG 会多次调用 LLM（6 个不同的链）
> - 每次查询的成本比普通 RAG 高
> - 但答案质量也更高

### 导入库和设置环境

```python
import os
import sys
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field

# 从 .env 文件加载环境变量
load_dotenv()

# 设置 OpenAI API 密钥
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
```

> **💡 代码解释**
>
> **BaseModel 和 Field 的作用：**
> - 用于定义结构化输出
> - 确保 LLM 返回格式正确的结果
> - 例如：检索决策只返回 "Yes" 或 "No"
>
> **⚠️ 新手注意**
> - `langchain_core.pydantic_v1` 是 Pydantic 的兼容版本
> - 如果用最新版 Pydantic，导入路径可能不同

---

## 🛠️ 第二步：准备测试数据

### 📖 这是什么？

下载并加载用于测试的 PDF 文档。

### 💻 完整代码

```python
from langchain.document_loaders import PyPDFLoader

# 创建数据目录
os.makedirs('data', exist_ok=True)

# 下载气候变化的 PDF 文档
!wget -O data/Understanding_Climate_Change.pdf \
    https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf

# 加载 PDF
path = "data/Understanding_Climate_Change.pdf"
loader = PyPDFLoader(path)
documents = loader.load()

print(f"✓ 已加载 {len(documents)} 页文档")
```

> **💡 代码解释**
> - `PyPDFLoader` 自动提取 PDF 文本
> - 每页作为一个 Document 对象
> - `page_content` 属性存储文本内容

---

## 🛠️ 第三步：创建向量存储

### 📖 这是什么？

将文档转换成可以进行相似度搜索的向量存储。

### 💻 完整代码

```python
# 假设 encode_pdf 是辅助函数（来自 helper_functions）
# 如果不存在，可以手动创建：
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 创建文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# 分割文档
splits = text_splitter.split_documents(documents)

# 创建嵌入模型和向量存储
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(splits, embeddings)

print(f"✓ 向量存储已创建，包含 {len(splits)} 个文本块")
```

> **💡 代码解释**
>
> **向量存储的作用：**
> ```
> 文本块 → 嵌入模型 → 向量
> 查询 → 嵌入模型 → 查询向量
> 相似度搜索 → 找到最相似的向量 → 返回对应文本
> ```
>
> **⚠️ 新手注意**
> - `chunk_size=1000` 表示每块最多 1000 字符
> - `chunk_overlap=200` 保持上下文连贯

---

## 🛠️ 第四步：初始化语言模型

### 📖 这是什么？

初始化用于所有步骤的 LLM。

### 💻 完整代码

```python
llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1000, temperature=0)

print("✓ 语言模型已初始化")
```

> **💡 代码解释**
> - `temperature=0`：输出最确定、最稳定
> - `max_tokens=1000`：限制输出长度
> - `gpt-4o-mini`：性价比高

---

## 🛠️ 第五步：定义提示模板和输出结构

### 📖 这是什么？

Self-RAG 的核心是为每个步骤定义专门的提示模板和输出格式。

### 💻 完整代码

```python
# ============ 步骤 1：检索决策 ============
class RetrievalResponse(BaseModel):
    """检索决策的输出结构"""
    response: str = Field(
        ...,
        description="Determines if retrieval is necessary",
        description="仅输出 'Yes' 或 'No'."
    )

retrieval_prompt = PromptTemplate(
    input_variables=["query"],
    template="Given the query '{query}', determine if retrieval is needed. Output only 'Yes' or 'No'."
)

# ============ 步骤 2：相关性评估 ============
class RelevanceResponse(BaseModel):
    """相关性评估的输出结构"""
    response: str = Field(
        ...,
        description="Determines if context is relevant",
        description="仅输出 'Relevant' 或 'Irrelevant'."
    )

relevance_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="Given the query '{query}' and the context '{context}', determine if the context is relevant. Output only 'Relevant' or 'Irrelevant'."
)

# ============ 步骤 3：回复生成 ============
class GenerationResponse(BaseModel):
    """回复生成的输出结构"""
    response: str = Field(
        ...,
        title="Generated response",
        description="The generated response."
    )

generation_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="Given the query '{query}' and the context '{context}', generate a response."
)

# ============ 步骤 4：支持度评估 ============
class SupportResponse(BaseModel):
    """支持度评估的输出结构"""
    response: str = Field(
        ...,
        title="Determines if response is supported",
        description="Output 'Fully supported', 'Partially supported', or 'No support'."
    )

support_prompt = PromptTemplate(
    input_variables=["response", "context"],
    template="Given the response '{response}' and the context '{context}', determine if the response is supported by the context. Output 'Fully supported', 'Partially supported', or 'No support'."
)

# ============ 步骤 5：效用评估 ============
class UtilityResponse(BaseModel):
    """效用评估的输出结构"""
    response: int = Field(
        ...,
        title="Utility rating",
        description="Rate the utility of the response from 1 to 5."
    )

utility_prompt = PromptTemplate(
    input_variables=["query", "response"],
    template="Given the query '{query}' and the response '{response}', rate the utility of the response from 1 to 5."
)

# ============ 创建 LLM 链 ============
# 每个链 = 提示模板 + LLM（带结构化输出）
retrieval_chain = retrieval_prompt | llm.with_structured_output(RetrievalResponse)
relevance_chain = relevance_prompt | llm.with_structured_output(RelevanceResponse)
generation_chain = generation_prompt | llm.with_structured_output(GenerationResponse)
support_chain = support_prompt | llm.with_structured_output(SupportResponse)
utility_chain = utility_prompt | llm.with_structured_output(UtilityResponse)

print("✓ 所有提示模板和链已创建")
```

> **💡 代码解释**
>
> **每个步骤的作用：**
>
> | 链 | 输入 | 输出 | 作用 |
> |---|------|------|------|
> | retrieval_chain | query | Yes/No | 判断是否需要检索 |
> | relevance_chain | query, context | Relevant/Irrelevant | 评估文档相关性 |
> | generation_chain | query, context | 答案文本 | 生成回复 |
> | support_chain | response, context | 支持度等级 | 评估答案依据 |
> | utility_chain | query, response | 1-5 分 | 评估答案效用 |
>
> **with_structured_output 的作用：**
> - 强制 LLM 返回指定格式
> - 避免解析错误
> - 例如：检索决策只返回 "Yes" 或 "No"，不会有其他内容
>
> **⚠️ 新手注意**
> - 提示词用英文是因为 GPT 对英文理解更准确
> - 可以改成中文提示词（需要调整输出描述）
> - Pydantic 的 `Field` 用于定义字段约束

---

## 🛠️ 第六步：实现 Self-RAG 核心逻辑

### 📖 这是什么？

整合所有步骤，实现完整的 Self-RAG 流程。

### 💻 完整代码

```python
def self_rag(query, vectorstore, top_k=3):
    """
    Self-RAG 主函数

    参数:
        query: 用户查询
        vectorstore: FAISS 向量存储
        top_k: 检索文档数量（默认 3）

    返回:
        最终答案字符串

    流程:
        1. 检索决策 → 是否需要检索？
        2. 如果 Yes:
           a. 检索文档
           b. 评估每个文档的相关性
           c. 基于相关文档生成多个候选答案
           d. 评估每个答案的支持度
           e. 评估每个答案的效用
           f. 选择最佳答案
        3. 如果 No:
           直接生成答案
    """
    print(f"\n处理查询：{query}")

    # ========== 步骤 1：检索决策 ==========
    print("步骤 1：判断是否需要检索...")
    input_data = {"query": query}
    retrieval_decision = retrieval_chain.invoke(input_data).response.strip().lower()
    print(f"检索决策：{retrieval_decision}")

    if retrieval_decision == 'yes':
        # ========== 步骤 2：检索相关文档 ==========
        print("步骤 2：检索相关文档...")
        docs = vectorstore.similarity_search(query, k=top_k)
        contexts = [doc.page_content for doc in docs]
        print(f"检索到 {len(contexts)} 个文档")

        # ========== 步骤 3：评估文档相关性 ==========
        print("步骤 3：评估检索到的文档的相关性...")
        relevant_contexts = []

        for i, context in enumerate(contexts):
            input_data = {"query": query, "context": context}
            relevance = relevance_chain.invoke(input_data).response.strip().lower()
            print(f"文档 {i+1} 相关性：{relevance}")

            if relevance == 'relevant':
                relevant_contexts.append(context)

        print(f"相关上下文数量：{len(relevant_contexts)}")

        # 如果没有找到相关上下文，无需检索直接生成
        if not relevant_contexts:
            print("未找到相关上下文。无需检索直接生成...")
            input_data = {"query": query, "context": "No relevant context found."}
            return generation_chain.invoke(input_data).response

        # ========== 步骤 4：使用相关上下文生成响应 ==========
        print("步骤 4：使用相关上下文生成响应...")
        responses = []  # 存储（答案，支持度，效用）三元组

        for i, context in enumerate(relevant_contexts):
            print(f"正在为上下文 {i+1} 生成响应...")

            # 生成答案
            input_data = {"query": query, "context": context}
            response = generation_chain.invoke(input_data).response

            # ========== 步骤 5：评估支持度 ==========
            print(f"步骤 5：评估响应 {i+1} 的支持度...")
            input_data = {"response": response, "context": context}
            support = support_chain.invoke(input_data).response.strip().lower()
            print(f"支持度评估：{support}")

            # ========== 步骤 6：评估效用 ==========
            print(f"步骤 6：评估响应 {i+1} 的效用...")
            input_data = {"query": query, "response": response}
            utility = int(utility_chain.invoke(input_data).response)
            print(f"效用评分：{utility}")

            # 存储结果
            responses.append((response, support, utility))

        # ========== 步骤 7：选择最佳响应 ==========
        print("选择最佳响应...")

        # 排序规则：
        # 1. 优先考虑"fully supported"
        # 2. 其次考虑效用分数
        best_response = max(
            responses,
            key=lambda x: (x[1] == 'fully supported', x[2])
        )

        print(f"最佳响应支持度：{best_response[1]}，效用：{best_response[2]}")
        return best_response[0]

    else:
        # 无需检索直接生成
        print("无需检索直接生成...")
        input_data = {"query": query, "context": "No retrieval needed."}
        return generation_chain.invoke(input_data).response
```

> **💡 代码解释**
>
> **排序逻辑详解：**
>
> ```python
> best_response = max(
>     responses,
>     key=lambda x: (x[1] == 'fully supported', x[2])
> )
> ```
>
> 这个排序的 key 是一个元组：
> - 第一个元素：`x[1] == 'fully supported'`（布尔值，True=1，False=0）
> - 第二个元素：`x[2]`（效用分数 1-5）
>
> **排序结果示例：**
> | 答案 | 支持度 | 效用 | 排序 key |
> |------|--------|------|---------|
> | A | fully supported | 4 | (True, 4) = (1, 4) |
> | B | partially supported | 5 | (False, 5) = (0, 5) |
> | C | fully supported | 3 | (True, 3) = (1, 3) |
>
> 排序结果：A > C > B（优先考虑支持度）
>
> **⚠️ 新手注意**
> - `.strip().lower()` 去除空格并转小写，方便比较
> - `int()` 转换效用分数为整数
> - 如果没有相关上下文，回退到无检索生成

---

## 🛠️ 第七步：测试 Self-RAG

### 📖 这是什么？

用不同类型的查询测试 Self-RAG 系统。

### 💻 完整代码

```python
# ============ 测试 1：高相关性的简单查询 ============
print("=" * 60)
print("测试 1：与文档高度相关的查询")
print("=" * 60)

query = "气候变化对环境的影响是什么？"
response = self_rag(query, vectorstore)

print("\n最终响应：")
print(response)

# ============ 测试 2：低相关性的挑战性查询 ============
print("\n" + "=" * 60)
print("测试 2：与文档相关性低的查询")
print("=" * 60)

query = "哈利是如何打败奇洛的？"
response = self_rag(query, vectorstore)

print("\n最终响应：")
print(response)
```

> **💡 代码解释**
>
> **测试设计思路：**
>
> **测试 1（相关查询）：**
> - 问题关于气候变化
> - 文档也是关于气候变化
> - 预期：需要检索，能找到相关文档，生成高质量答案
>
> **测试 2（无关查询）：**
> - 问题关于哈利波特（与文档无关）
> - 文档是关于气候变化的
> - 预期：可能判断无需检索，或检索后发现不相关，回退到无检索生成
>
> **📊 预期输出示例：**
>
> ```
> ============================================================
> 测试 1：与文档高度相关的查询
> ============================================================
>
> 处理查询：气候变化对环境的影响是什么？
> 步骤 1：判断是否需要检索...
> 检索决策：yes
> 步骤 2：检索相关文档...
> 检索到 3 个文档
> 步骤 3：评估检索到的文档的相关性...
> 文档 1 相关性：relevant
> 文档 2 相关性：relevant
> 文档 3 相关性：irrelevant
> 相关上下文数量：2
> 步骤 4：使用相关上下文生成响应...
> 正在为上下文 1 生成响应...
> 步骤 5：评估响应 1 的支持度...
> 支持度评估：fully supported
> 步骤 6：评估响应 1 的效用...
> 效用评分：5
> ...
> 最佳响应支持度：fully supported，效用：5
>
> 最终响应：
> 气候变化对环境的影响包括：全球气温上升、极端天气事件增加、
> 海平面上升、生态系统破坏等...
>
> ============================================================
> 测试 2：与文档相关性低的查询
> ============================================================
>
> 处理查询：哈利是如何打败奇洛的？
> 步骤 1：判断是否需要检索...
> 检索决策：no
> 无需检索直接生成...
>
> 最终响应：
> 在《哈利·波特与魔法石》中，哈利通过母亲爱的保护击败了奇洛...
> ```

---

## 🎯 完整代码整合

### 一站式完整流程

```python
# ============== 环境准备 ==============
!pip install langchain langchain-openai python-dotenv

import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# ============== 加载文档 ==============
path = "data/Understanding_Climate_Change.pdf"
loader = PyPDFLoader(path)
documents = loader.load()

# ============== 创建向量存储 ==============
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(splits, embeddings)

# ============== 初始化 LLM ==============
llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1000, temperature=0)

# ============== 定义输出结构 ==============
class RetrievalResponse(BaseModel):
    response: str = Field(description="Only output 'Yes' or 'No'.")

class RelevanceResponse(BaseModel):
    response: str = Field(description="Only output 'Relevant' or 'Irrelevant'.")

class GenerationResponse(BaseModel):
    response: str = Field(description="The generated response.")

class SupportResponse(BaseModel):
    response: str = Field(description="Output 'Fully supported', 'Partially supported', or 'No support'.")

class UtilityResponse(BaseModel):
    response: int = Field(description="Rate utility from 1 to 5.")

# ============== 创建提示模板和链 ==============
retrieval_chain = PromptTemplate(
    input_variables=["query"],
    template="Given the query '{query}', determine if retrieval is needed. Output only 'Yes' or 'No'."
) | llm.with_structured_output(RetrievalResponse)

relevance_chain = PromptTemplate(
    input_variables=["query", "context"],
    template="Given the query '{query}' and the context '{context}', determine relevance. Output 'Relevant' or 'Irrelevant'."
) | llm.with_structured_output(RelevanceResponse)

generation_chain = PromptTemplate(
    input_variables=["query", "context"],
    template="Given the query '{query}' and the context '{context}', generate a response."
) | llm.with_structured_output(GenerationResponse)

support_chain = PromptTemplate(
    input_variables=["response", "context"],
    template="Given the response '{response}' and the context '{context}', evaluate support. Output 'Fully supported', 'Partially supported', or 'No support'."
) | llm.with_structured_output(SupportResponse)

utility_chain = PromptTemplate(
    input_variables=["query", "response"],
    template="Given the query '{query}' and the response '{response}', rate utility 1-5."
) | llm.with_structured_output(UtilityResponse)

# ============== Self-RAG 主函数 ==============
def self_rag(query, vectorstore, top_k=3):
    print(f"\n处理查询：{query}")

    # 步骤 1：检索决策
    retrieval_decision = retrieval_chain.invoke({"query": query}).response.strip().lower()
    print(f"检索决策：{retrieval_decision}")

    if retrieval_decision == 'yes':
        # 步骤 2：检索文档
        docs = vectorstore.similarity_search(query, k=top_k)
        contexts = [doc.page_content for doc in docs]
        print(f"检索到 {len(contexts)} 个文档")

        # 步骤 3：评估相关性
        relevant_contexts = []
        for i, context in enumerate(contexts):
            relevance = relevance_chain.invoke(
                {"query": query, "context": context}
            ).response.strip().lower()
            print(f"文档 {i+1} 相关性：{relevance}")
            if relevance == 'relevant':
                relevant_contexts.append(context)

        if not relevant_contexts:
            print("未找到相关上下文")
            return generation_chain.invoke(
                {"query": query, "context": "No relevant context"}
            ).response

        # 步骤 4-6：生成 + 评估
        responses = []
        for i, context in enumerate(relevant_contexts):
            response = generation_chain.invoke(
                {"query": query, "context": context}
            ).response

            support = support_chain.invoke(
                {"response": response, "context": context}
            ).response.strip().lower()

            utility = int(utility_chain.invoke(
                {"query": query, "response": response}
            ).response)

            responses.append((response, support, utility))
            print(f"答案 {i+1}: 支持度={support}, 效用={utility}")

        # 步骤 7：选择最佳
        best_response = max(
            responses,
            key=lambda x: (x[1] == 'fully supported', x[2])
        )
        print(f"最佳响应：支持度={best_response[1]}, 效用={best_response[2]}")
        return best_response[0]
    else:
        print("无需检索")
        return generation_chain.invoke(
            {"query": query, "context": "No retrieval needed"}
        ).response

# ============== 测试 ==============
query = "气候变化对环境的影响是什么？"
response = self_rag(query, vectorstore)
print(f"\n最终答案：{response}")
```

---

## 📚 Self-RAG 的优势总结

| 优势 | 说明 | 实际价值 |
|------|------|---------|
| **动态检索** | 智能判断是否需要检索 | 节省成本，减少噪音 |
| **相关性过滤** | 只使用相关文档生成答案 | 提高答案质量 |
| **质量保证** | 支持度和效用双重评估 | 答案更可靠 |
| **灵活性** | 可以在有/无检索下生成 | 适应不同场景 |
| **提高准确性** | 基于相关信息并评估支持度 | 减少胡编乱造 |

---

## 🎓 术语解释表

| 术语 | 英文 | 解释 |
|------|------|------|
| 自反思 | Self-Reflective | 系统能评估自己的输出 |
| 检索决策 | Retrieval Decision | 判断是否需要检索 |
| 相关性评估 | Relevance Assessment | 评估文档与查询的相关性 |
| 支持度 | Support | 答案是否有上下文依据 |
| 效用 | Utility | 答案对用户的有用程度 |
| 结构化输出 | Structured Output | 强制 LLM 返回指定格式 |
| 链 | Chain | 多个组件的顺序执行 |

---

## ❓ 常见问题 FAQ

### Q1: Self-RAG 和普通 RAG 的主要区别是什么？

**A:** 关键区别：

| 特性 | 普通 RAG | Self-RAG |
|------|---------|---------|
| 检索决策 | 总是检索 | 智能判断 |
| 文档过滤 | 无 | 相关性评估 |
| 质量评估 | 无 | 支持度 + 效用 |
| 成本 | 低 | 高（多次 LLM 调用） |
| 答案质量 | 一般 | 更高 |

### Q2: Self-RAG 的成本高吗？

**A:** 每次查询的 LLM 调用次数：
- 检索决策：1 次
- 相关性评估：k 次（k=检索文档数）
- 答案生成：≤k 次
- 支持度评估：≤k 次
- 效用评估：≤k 次

总计：1 + 4k 次左右（k=3 时约 13 次）

建议：适合对质量要求高的场景，成本敏感场景慎用。

### Q3: 如何降低 Self-RAG 的成本？

**A:** 可以：
1. 减少检索文档数（top_k=1 或 2）
2. 跳过效用评估（只评估支持度）
3. 使用更便宜的模型（如 GPT-3.5-Turbo）
4. 对简单问题直接回答（设置规则跳过检索决策）

### Q4: 支持度评估和效用评估有什么区别？

**A:**
- **支持度**：答案是否基于上下文（客观）
  - "Fully supported" = 答案完全有依据
  - "No support" = 答案可能是编的

- **效用**：答案对用户有多大帮助（主观）
  - 5 分 = 完美回答
  - 1 分 = 毫无用处

### Q5: 如果所有答案的支持度都很低怎么办？

**A:** 这可能意味着：
1. 检索到的文档不相关
2. 文档中没有答案所需信息
3. 需要扩大检索范围

处理策略：
- 回退到无检索生成
- 告知用户"未找到相关信息"
- 建议用户换一种问法

---

## ✅ 学习检查清单

- [ ] 我理解了 Self-RAG 的核心思想
- [ ] 我知道 5 个步骤各自的作用
- [ ] 我能解释检索决策的意义
- [ ] 我理解支持度评估和效用评估的区别
- [ ] 我能创建自己的 Self-RAG 系统
- [ ] 我知道 Self-RAG 的优缺点

---

## 🚀 下一步学习建议

1. **调整评估标准**：修改提示词，自定义支持度和效用的评估标准
2. **优化成本**：尝试减少 LLM 调用次数的策略
3. **比较多代答案**：观察不同文档生成的答案差异
4. **学习 C-RAG**：继续学习校正 RAG 技术

---

> **💪 恭喜！** 你已经完成了 Self-RAG 的新手教程！现在你掌握了如何构建自反思的智能 RAG 系统，这是构建高质量问答应用的重要技能！
