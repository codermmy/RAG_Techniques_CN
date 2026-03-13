# 🌟 新手入门：可靠 RAG（Reliable RAG）系统

> **💡 给新手的说明**
> - **难度等级**：⭐⭐⭐☆☆（进阶级）
> - **预计时间**：45-60 分钟
> - **前置知识**：基础 RAG 系统知识，了解 LangChain 框架
> - **学习目标**：学会构建更可靠的 RAG 系统，包含质量检查和幻觉检测

---

## 📖 核心概念理解

### 什么是可靠 RAG 系统？

**可靠 RAG**（Reliable RAG）是在基础 RAG 系统上增加了**质量检查机制**的高级系统。它能自动检查检索到的文档是否相关，生成的答案是否有依据。

### 🍕 通俗理解：严谨的学霸助手

想象两个助手：

1. **普通 RAG** 就像一个热心的助手，看到问题就立刻回答，但有时会答非所问或瞎编
2. **可靠 RAG** 就像一个严谨的学霸：
   - 先检查找到的资料是否真的相关（**文档相关性检查**）
   - 回答后再检查答案是否有依据（**幻觉检测**）
   - 最后还会告诉你答案是从哪段资料来的（**来源标注**）

### 🔄 可靠 RAG 的工作流程

```
用户提问
    ↓
检索文档
    ↓
【检查 1】文档是否相关？→ 不相关则过滤
    ↓
生成答案
    ↓
【检查 2】答案有幻觉吗？→ 有幻觉则重新生成
    ↓
【检查 3】标注来源
    ↓
返回可靠答案
```

### 🔑 核心组件解释

| 组件 | 作用 | 生活比喻 |
|------|------|----------|
| **文档检索** | 查找相关文档 | 图书馆找书 |
| **相关性评分器** | 检查文档是否相关 | 检查书是否对题 |
| **生成器** | 基于文档生成答案 | 根据资料写答案 |
| **幻觉检测器** | 检查答案是否有依据 | 检查答案是否瞎编 |
| **来源标注器** | 标注答案来源 | 标注引用出处 |

### 📊 术语解释

- **幻觉（Hallucination）**：AI 生成的内容看起来合理，但实际上没有事实依据，甚至完全错误
- **相关性（Relevance）**：检索到的文档与用户问题的相关程度
- **结构化输出**：让 AI 按照特定格式（如 JSON）输出结果

---

## 🛠️ 第一步：环境准备

### 📖 这是什么？

安装可靠 RAG 系统所需的 Python 库。

### 💻 完整代码

```python
# ============================================
# 安装所需的包
# ============================================
# langchain: RAG 框架核心
# langchain-community: 社区扩展组件
# python-dotenv: 环境变量管理

!pip install langchain langchain-community python-dotenv
```

> **💡 代码解释**
> - 这些是可靠 RAG 系统的核心依赖
> - 后续还会用到一些额外的库，会在使用前安装
>
> **⚠️ 新手注意**
> - 某些包可能需要额外安装（如 langchain_groq、langchain_cohere）
> - 安装过程中如有警告通常可以忽略

---

## 🔑 第二步：配置 API 密钥

### 📖 这是什么？

可靠 RAG 系统使用多个 API 服务：
- **Groq**：提供快速的 LLM 服务（用于文档评分和答案生成）
- **Cohere**：提供 Embedding 服务（用于创建向量）

### 💻 完整代码

```python
# ============================================
# 导入必要的库并配置 API 密钥
# ============================================
import os
from dotenv import load_dotenv

# 从 '.env' 文件加载环境变量
load_dotenv()

# 设置 API 密钥
# GROQ_API_KEY: 用于 LLM（llama-3.1-8b 小型模型 和 mixtral-8x7b 大型模型）
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

# COHERE_API_KEY: 用于 Embedding（文本向量化）
os.environ['COHERE_API_KEY'] = os.getenv('COHERE_API_KEY')
```

> **💡 代码解释**
> - `load_dotenv()` 从 `.env` 文件加载配置
> - `os.getenv()` 读取环境变量
>
> **⚠️ 新手注意**
> - **API 密钥获取**：
>   - Groq: 访问 https://console.groq.com 注册获取
>   - Cohere: 访问 https://dashboard.cohere.com 注册获取
> - **免费额度**：两个服务都提供一定的免费额度，适合学习使用
>
> **📄 推荐的 .env 文件格式**
> ```
> # .env 文件内容
> GROQ_API_KEY=gsk_your_groq_key_here
> COHERE_API_KEY=your_cohere_key_here
> ```

---

## 🗄️ 第三步：创建向量存储

### 📖 这是什么？

从网页加载文档，分割成小块，然后创建向量存储。

### 💻 完整代码

```python
# ============================================
# 构建索引
# ============================================
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings

# ========== 步骤 1：设置 Embedding 模型 ==========
# 使用 Cohere 的 embed-english-v3.0 模型
embedding_model = CohereEmbeddings(model="embed-english-v3.0")

# ========== 步骤 2：准备要索引的文档 ==========
# 这里是 5 篇关于 AI Agent 设计模式的文章
urls = [
    "https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-2-reflection/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-3-tool-use/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-4-planning/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-5-multi-agent-collaboration/?ref=dl-staging-website.ghost.io"
]

# ========== 步骤 3：加载文档 ==========
# 从每个 URL 加载内容
docs = [WebBaseLoader(url).load() for url in urls]
# 展开成一维列表
docs_list = [item for sublist in docs for item in sublist]

# ========== 步骤 4：分割文档 ==========
# 使用 tiktoken 编码器的分块器
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500,    # 每块 500 个 token
    chunk_overlap=0    # 块之间不重叠
)
doc_splits = text_splitter.split_documents(docs_list)

# ========== 步骤 5：创建向量存储 ==========
vectorstore = Chroma.from_documents(
    documents=doc_splits,           # 分割后的文档
    collection_name="rag",          # 集合名称
    embedding=embedding_model,      # 使用的嵌入模型
)

# ========== 步骤 6：创建检索器 ==========
retriever = vectorstore.as_retriever(
    search_type="similarity",       # 相似度搜索
    search_kwargs={'k': 4},         # 每次检索返回 4 个文档
)
```

> **💡 代码解释**
>
> **Chroma 是什么？**
> - 一个轻量级的向量数据库
> - 适合快速原型开发和小项目
>
> **分块参数说明**：
> - `chunk_size=500`：每块 500 个 token（token 是文本的单位，约等于 4/3 个英文单词）
> - `chunk_overlap=0`：块之间不重叠（这个设置可以根据需要调整）
>
> **检索器说明**：
> - `search_type="similarity"`：使用相似度搜索
> - `search_kwargs={'k': 4}`：返回最相似的 4 个文档
>
> **⚠️ 新手注意**
> - 如果无法访问这些 URL，可以替换成其他网页
> - 也可以改用本地文件加载
>
> **📊 术语解释**
> - **Token**：文本处理的基本单位，英文中约等于 3/4 个单词
> - **RecursiveCharacterTextSplitter**：递归地按字符分割文本的分块器

---

## ❓ 第四步：准备问题

### 📖 这是什么？

准备一个测试问题来验证系统。

### 💻 完整代码

```python
# 设置测试问题
question = "不同类型的 Agent 设计模式有哪些？"
```

> **💡 预期**
> - 这个问题应该能从刚才加载的 5 篇文章中找到答案
> - 因为这些文章就是关于 Agent 设计模式的系列教程

---

## 🔍 第五步：检索文档

### 📖 这是什么？

使用检索器查找与问题相关的文档。

### 💻 完整代码

```python
# 执行检索
docs = retriever.invoke(question)
```

> **💡 代码解释**
> - `invoke()` 是调用检索器的方法
> - 返回的是与问题最相似的文档列表
>
> **⚠️ 新手注意**
> - 检索结果可能不完美，这就是为什么需要后面的质量检查

---

## 📋 第六步：检查文档外观

### 📖 这是什么？

查看检索到的第一个文档的内容，了解数据格式。

### 💻 完整代码

```python
# 查看第一个文档的详细信息
print(f"标题：{docs[0].metadata['title']}\n")
print(f"来源：{docs[0].metadata['source']}\n")
print(f"内容：{docs[0].page_content}\n")
```

> **💡 预期输出示例**
> ```
> 标题：Agentic Design Patterns Part 2: Reflection
>
> 来源：https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-2-reflection/
>
> 内容：AI agents that can reflect on their actions and improve...
> ```
>
> **📊 术语解释**
> - `metadata`：文档的元数据，如标题、来源等
> - `page_content`：文档的实际内容

---

## ✅ 第七步：检查文档相关性（核心功能 1）

### 📖 这是什么？

这是可靠 RAG 的**第一个关键检查点**：过滤掉不相关的文档。

### 💻 完整代码

```python
# ============================================
# 创建文档相关性评分器
# ============================================
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

# ========== 定义数据模型 ==========
# 这告诉 AI 输出什么格式
class GradeDocuments(BaseModel):
    """对检索到的文档进行相关性检查的二进制评分。"""
    binary_score: str = Field(
        description="文档与问题相关，'yes' 或 'no'"
    )

# ========== 创建 LLM 实例 ==========
# 使用 Groq 的 llama-3.1-8b-instant 模型（快速且便宜）
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# 让 LLM 输出结构化数据
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# ========== 创建提示词 ==========
system = """你是一个评估检索到的文档与用户问题相关性的评分者。
    如果文档包含与用户问题相关的关键词或语义含义，则将其评为相关。
    不需要是严格的测试。目标是过滤掉错误的检索结果。
    给出二进制评分 'yes' 或 'no' 以表示文档是否与问题相关。"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "检索到的文档：\n\n {document} \n\n 用户问题：{question}"),
    ]
)

# ========== 创建评分器 ==========
# 将提示词和 LLM 连接起来
retrieval_grader = grade_prompt | structured_llm_grader
```

> **💡 代码解释**
>
> **Pydantic 是什么？**
> - 一个数据验证库
> - 这里用来定义 AI 输出的格式
>
> **with_structured_output 是什么？**
> - 让 AI 按照指定格式输出
> - 确保输出可以被程序处理
>
> **temperature=0 是什么？**
> - 控制 AI 的随机性
> - 0 表示最确定、最一致的输出
>
> **⚠️ 新手注意**
> - `|` 是 LangChain 的链式操作符，把组件连接起来
> - 这行代码等价于先设置提示词，再调用 LLM

### 执行文档过滤

```python
# ============================================
# 过滤不相关文档
# ============================================
docs_to_use = []

for doc in docs:
    # 打印文档内容（用于调试）
    print(doc.page_content, '\n', '-'*50)

    # 让评分器判断文档是否相关
    res = retrieval_grader.invoke({"question": question, "document": doc.page_content})
    print(res, '\n')

    # 如果相关，保留这个文档
    if res.binary_score == 'yes':
        docs_to_use.append(doc)
```

> **💡 预期输出**
> ```
> [文档内容...]
> --------------------------------------------------
> binary_score='yes'
>
> [文档内容...]
> --------------------------------------------------
> binary_score='no'  <- 这个文档会被过滤掉
> ```
>
> **⚠️ 新手注意**
> - 如果所有文档都被过滤掉，可能需要调整评分标准
> - 这个过程需要调用 API，可能需要一点时间

---

## 💬 第八步：生成答案（核心功能 2）

### 📖 这是什么？

基于过滤后的文档生成答案。

### 💻 完整代码

```python
# ============================================
# 生成答案
# ============================================
from langchain_core.output_parsers import StrOutputParser

# ========== 创建提示词 ==========
system = """你是一个问答任务的助手。根据你的知识回答问题。
最多使用三到五句话，保持答案简洁。"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "检索到的文档：\n\n <docs>{documents}</docs> \n\n 用户问题：<question>{question}</question>"),
    ]
)

# ========== 创建 LLM 实例 ==========
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# ========== 定义文档格式化函数 ==========
def format_docs(docs):
    """将文档列表格式化成字符串"""
    return "\n".join(
        f"<doc{i+1}>:\n标题:{doc.metadata['title']}\n来源:{doc.metadata['source']}\n内容:{doc.page_content}\n</doc{i+1}>\n"
        for i, doc in enumerate(docs)
    )

# ========== 创建链 ==========
# 提示词 → LLM → 字符串解析器
rag_chain = prompt | llm | StrOutputParser()

# ========== 执行生成 ==========
generation = rag_chain.invoke({
    "documents": format_docs(docs_to_use),
    "question": question
})

print(generation)
```

> **💡 代码解释**
>
> **StrOutputParser 是什么？**
> - 将 LLM 的输出解析为字符串
> - 确保输出格式一致
>
> **format_docs 函数的作用**：
> - 把多个文档格式化成 LLM 容易理解的格式
> - 每个文档都有编号、标题、来源和内容
>
> **⚠️ 新手注意**
> - `temperature=0` 确保输出一致
> - 如果需要更有创意的答案，可以调高 temperature

---

## 🔎 第九步：检查幻觉（核心功能 3）

### 📖 这是什么？

这是可靠 RAG 的**第二个关键检查点**：检测 AI 是否瞎编了答案。

### 💻 完整代码

```python
# ============================================
# 幻觉检测器
# ============================================

# ========== 定义数据模型 ==========
class GradeHallucinations(BaseModel):
    """对 'generation' 答案中是否存在幻觉的二进制评分。"""
    binary_score: str = Field(
        ...,
        description="答案基于事实，'yes' 或 'no'"
    )

# ========== 创建 LLM 实例 ==========
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# ========== 创建提示词 ==========
system = """你是一个评估 LLM 生成是否基于/支持一组检索到的事实的评分者。
    给出二进制评分 'yes' 或 'no'。'Yes' 表示答案基于/支持这组事实。"""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "事实集合：\n\n <facts>{documents}</facts> \n\n LLM 生成：<generation>{generation}</generation>"),
    ]
)

# ========== 创建幻觉检测器 ==========
hallucination_grader = hallucination_prompt | structured_llm_grader

# ========== 执行检测 ==========
response = hallucination_grader.invoke({
    "documents": format_docs(docs_to_use),
    "generation": generation
})

print(response)
```

> **💡 预期输出**
> ```
> binary_score='yes'  <- 答案有依据，通过检查
> 或
> binary_score='no'   <- 答案可能是瞎编的
> ```
>
> **⚠️ 新手注意**
> - 如果检测到幻觉，可以：
>   - 重新生成答案
>   - 返回更多检索文档
>   - 告知用户答案可能不可靠

---

## 📍 第十步：标注来源（核心功能 4）

### 📖 这是什么？

可靠 RAG 的**第三个关键功能**：标注答案是从哪些文档的哪些部分来的。

### 💻 完整代码

```python
# ============================================
# 来源标注器
# ============================================
from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

# ========== 定义数据模型 ==========
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

# ========== 创建 LLM 实例 ==========
# 使用更大的模型来处理这个复杂任务
llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

# ========== 创建解析器 ==========
parser = PydanticOutputParser(pydantic_object=HighlightDocuments)

# ========== 创建提示词 ==========
system = """你是一个用于文档搜索和检索的高级助手。提供以下信息：
1. 一个问题。
2. 基于问题生成的答案。
3. 在生成答案时引用的一组文档。

你的任务是从提供的文档中识别和提取完全内联的片段，这些片段直接对应于用于生成给定答案的内容。提取的片段必须是文档的逐字摘录，确保与提供的文档中的文本逐字匹配。

确保：
- （重要）每个片段与文档的一部分完全匹配，并完全包含在文档文本中。
- 每个片段与生成答案的相关性清晰，并直接支持提供的答案。
- （重要）如果你没有使用特定文档，不要提及它。

使用的文档：<docs>{documents}</docs>

用户问题：<question>{question}</question>

生成的答案：<answer>{generation}</answer>

<format_instruction>
{format_instructions}
</format_instruction>
"""

prompt = PromptTemplate(
    template=system,
    input_variables=["documents", "question", "generation"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# ========== 创建查找链 ==========
doc_lookup = prompt | llm | parser

# ========== 执行查找 ==========
lookup_response = doc_lookup.invoke({
    "documents": format_docs(docs_to_use),
    "question": question,
    "generation": generation
})
```

> **💡 代码解释**
>
> **PydanticOutputParser 是什么？**
> - 将 LLM 输出解析为 Pydantic 模型
> - 确保输出格式严格符合要求
>
> **partial_variables 是什么？**
> - 给提示词模板预填充一些变量
> - 这里用来插入格式说明
>
> **⚠️ 新手注意**
> - 这个步骤使用更大的模型（mixtral-8x7b），因为任务更复杂
> - 可能需要更长的运行时间

### 显示标注结果

```python
# 打印每个引用的来源
for id, title, source, segment in zip(
    lookup_response.id,
    lookup_response.title,
    lookup_response.source,
    lookup_response.segment
):
    print(f"ID: {id}\n标题：{title}\n来源：{source}\n文本片段：{segment}\n")
```

> **💡 预期输出示例**
> ```
> ID: doc1
> 标题：Agentic Design Patterns Part 2
> 来源：https://...
> 文本片段：Reflection is a key concept where agents can...
> ```

---

## 🎯 完整代码总结

下面是一个简化的可靠 RAG 流程：

```python
# 1. 配置
import os
os.environ['GROQ_API_KEY'] = 'your_key'
os.environ['COHERE_API_KEY'] = 'your_key'

# 2. 创建检索器
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings

embedding_model = CohereEmbeddings(model="embed-english-v3.0")
# ... 加载和分割文档 ...
retriever = vectorstore.as_retriever(search_kwargs={'k': 4})

# 3. 检索
docs = retriever.invoke("你的问题")

# 4. 过滤不相关文档
# ... 使用 LLM 评分 ...

# 5. 生成答案
# ... 基于过滤后的文档生成 ...

# 6. 检查幻觉
# ... 验证答案是否有依据 ...

# 7. 标注来源
# ... 提取引用的原文片段 ...
```

---

## ❓ 常见问题 FAQ

### Q1: 为什么要用多个 API 服务？
**A**:
- **Cohere**：专注于 Embedding，效果好
- **Groq**：提供快速的推理服务，适合实时应用
- 可以根据需求替换成单一服务（如 OpenAI）

### Q2: 幻觉检测准确吗？
**A**:
- 不是 100% 准确，但能过滤大部分明显幻觉
- 更大的模型通常检测更准确

### Q3: 可以只用一个模型完成所有任务吗？
**A**:
- 可以！但分开使用可以让每个任务更专业
- 也可以节省成本（用小模型做简单任务）

### Q4: 如果所有文档都被过滤掉怎么办？
**A**:
- 增加检索数量（k 值）
- 放宽评分标准
- 考虑使用其他检索策略

### Q5: 这个系统比基础 RAG 好在哪里？
**A**:
- **更可靠**：有质量检查机制
- **更透明**：标注答案来源
- **更可信**：检测并过滤幻觉

---

## 🚀 进阶技巧

### 构建完整的工作流

```python
def reliable_rag(question, retriever):
    """完整的可靠 RAG 流程"""
    # 1. 检索
    docs = retriever.invoke(question)

    # 2. 过滤不相关文档
    docs_to_use = []
    for doc in docs:
        res = retrieval_grader.invoke({"question": question, "document": doc.page_content})
        if res.binary_score == 'yes':
            docs_to_use.append(doc)

    # 3. 生成答案
    generation = rag_chain.invoke({
        "documents": format_docs(docs_to_use),
        "question": question
    })

    # 4. 检查幻觉
    hallucination_check = hallucination_grader.invoke({
        "documents": format_docs(docs_to_use),
        "generation": generation
    })

    # 5. 标注来源
    if hallucination_check.binary_score == 'yes':
        sources = doc_lookup.invoke({
            "documents": format_docs(docs_to_use),
            "question": question,
            "generation": generation
        })
        return {
            "answer": generation,
            "sources": sources,
            "reliable": True
        }
    else:
        return {
            "answer": "无法生成可靠答案",
            "reliable": False
        }
```

---

## 📚 关键知识点回顾

| 概念 | 说明 |
|------|------|
| **文档相关性检查** | 过滤掉与问题不相关的检索结果 |
| **幻觉检测** | 检查 AI 生成的答案是否有事实依据 |
| **来源标注** | 标注答案引用的原文片段 |
| **结构化输出** | 让 AI 按照指定格式输出结果 |
| **链（Chain）** | 将多个组件连接成完整流程 |
| **Pydantic 模型** | 定义数据验证和输出格式 |

---

## 🔗 与基础 RAG 的对比

| 特性 | 基础 RAG | 可靠 RAG |
|------|----------|----------|
| **文档过滤** | ❌ 无 | ✅ 有 |
| **幻觉检测** | ❌ 无 | ✅ 有 |
| **来源标注** | ❌ 无 | ✅ 有 |
| **复杂度** | 低 | 高 |
| **可靠性** | 一般 | 高 |
| **运行时间** | 快 | 较慢 |

---

*本教程是 RAG 技术系列教程的进阶级，建议先学习基础 RAG 教程再学习本教程。*

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--reliable-rag)
