# 🌟 新手入门：校正 RAG（Corrective RAG）系统

> **💡 给新手的说明**
> - **难度等级**：⭐⭐⭐⭐☆（进阶级）
> - **预计时间**：60-90 分钟
> - **前置知识**：了解基础 RAG 概念、熟悉 LangChain 框架
> - **学习目标**：理解 CRAG 的核心思想，掌握动态校正检索的策略，能够构建一个智能的 RAG 系统

---

## 📖 核心概念理解

### 什么是 CRAG（校正 RAG）？

**CRAG**（Corrective RAG，校正 RAG）是一种"聪明的"RAG 系统。它不像传统 RAG 那样盲目相信检索到的内容，而是会先**评估**检索结果的质量，然后根据评估结果采取不同的策略。

### 🍕 通俗理解：智能助手比喻

想象一下你有两个助手：

1. **普通 RAG 助手**：你问他问题，他立刻从文件柜里拿出一份文件，不管内容是否相关，就直接照着念给你听
2. **CRAG 助手**：你问他问题后，他会：
   - 先从文件柜里找资料
   - **仔细检查**找到的资料是否真的相关
   - 如果资料质量好 → 基于资料回答
   - 如果资料不够好 → 去网上搜索更多信息
   - 如果资料部分相关 → 把有用的部分提炼出来，再补充搜索
   - 最后综合所有信息给你一个准确的答案

**CRAG 的核心思想**：不是所有检索到的信息都是可靠的，需要先评估再使用！

### 🔑 核心组件解释

| 组件 | 作用 | 生活比喻 |
|------|------|----------|
| **FAISS 索引** | 存储和检索文档的向量数据库 | 公司的内部文件柜 |
| **检索评估器** | 判断检索结果是否相关 | 质量检查员 |
| **知识提炼** | 从文档中提取关键信息 | 秘书做会议摘要 |
| **网络搜索查询重写器** | 优化搜索关键词 | 帮你更好地向搜索引擎提问 |
| **回复生成器** | 综合所有信息生成回答 | 最终整理报告的分析师 |

### 📊 CRAG 工作流程图

```
你提问
   │
   ▼
从内部知识库检索
   │
   ▼
评估检索质量 ←──┐
   │            │
   ├── 相关 ──→ 知识提炼 ──┐
   │                      │
   ├── 无关 ──→ 网络搜索 ──┤
   │                      │
   └── 部分相关 ─→ 提炼 + 搜索 ─┤
                              │
                              ▼
                        综合所有知识
                              │
                              ▼
                        生成最终回答
```

---

## 🛠️ 第一步：环境准备

### 📖 这是什么？

在开始之前，我们需要安装必要的 Python 库。CRAG 系统依赖于向量数据库、LLM 和网络搜索功能。

### 💻 完整代码

```python
# ============================================
# 安装所需的包
# ============================================
# 每个包的作用：
# - langchain: RAG 框架的核心组件
# - langchain-openai: OpenAI 的集成
# - langchain-community: 社区贡献的集成，包含各种工具
# - faiss-cpu: Facebook 的高效相似度搜索库（用于向量检索）
# - python-dotenv: 管理 API 密钥等环境变量

!pip install langchain langchain-openai python-dotenv
!pip install faiss-cpu
```

> **💡 代码解释**
> - `!pip install` 是 Jupyter Notebook 中安装包的方式
> - 如果使用 Google Colab，大部分包已经预装
>
> **⚠️ 新手注意**
> - 如果遇到安装失败，可以尝试逐个安装
> - 如使用国内网络，可添加清华源：`!pip install langchain -i https://pypi.tuna.tsinghua.edu.cn/simple`
> - FAISS 在某些系统上可能需要额外配置，如遇问题请参考官方文档

---

## 🔑 第二步：配置 API 密钥和导入库

### 📖 这是什么？

CRAG 系统需要使用 OpenAI 的 API 来进行文本 Embedding 和 LLM 推理，同时需要搜索工具进行网络搜索。

### 💻 完整代码

```python
# ============================================
# 导入必要的库并配置 API 密钥
# ============================================
import os
import sys
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field

# 从 .env 文件加载环境变量
load_dotenv()

# 设置 OpenAI API 密钥环境变量
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# 导入辅助函数（如果使用原项目代码）
# sys.path.append('RAG_TECHNIQUES')
# from helper_functions import *

# 导入网络搜索工具
from langchain.tools import DuckDuckGoSearchResults
```

> **💡 代码解释**
> - `load_dotenv()` 从 `.env` 文件加载配置
> - `os.getenv()` 读取环境变量
> - `DuckDuckGoSearchResults` 是一个免费的网络搜索工具
>
> **⚠️ 新手注意**
> - **API 密钥安全**：永远不要把你的 API 密钥直接写在代码里提交到 Git！
> - 推荐使用 `.env` 文件：
>   ```
>   # .env 文件内容
>   OPENAI_API_KEY=sk-your-actual-key-here
>   ```
> - `.env` 文件应该添加到 `.gitignore` 中
>
> **❓ 常见问题**
> - **Q: 我不想用 OpenAI 怎么办？**
> - A: 可以使用其他 LLM 服务，如 Anthropic、Google 等，需要修改相应的导入和初始化代码

---

## 📄 第三步：下载示例文档并创建向量存储

### 📖 这是什么？

我们需要一个文档来测试 CRAG 系统。这里使用一个关于气候变化的 PDF 文档作为示例。

### 💻 完整代码

```python
# ============================================
# 创建目录并下载示例 PDF
# ============================================
import os

# 创建 data 目录
os.makedirs('data', exist_ok=True)

# 下载示例 PDF 文档
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf

# 指定 PDF 文件路径
path = "data/Understanding_Climate_Change.pdf"
```

> **💡 代码解释**
> - `os.makedirs('data', exist_ok=True)` 创建 data 目录，如果已存在也不会报错
> - `!wget` 从网络下载文件
> - `-O` 指定下载后的文件名

### 💻 创建向量存储

```python
# ============================================
# 使用辅助函数创建向量存储
# ============================================
# 假设 encode_pdf 是一个已定义的函数，用于处理 PDF 并创建 FAISS 索引
vectorstore = encode_pdf(path)
```

> **💡 代码解释**
> - `encode_pdf()` 函数会：
>   1. 读取 PDF 文件
>   2. 分割文本成块
>   3. 创建 Embedding 向量
>   4. 存储到 FAISS 索引中
>
> **⚠️ 新手注意**
> - 如果你没有 `encode_pdf` 函数，可以参考基础 RAG 教程中的实现
> - 也可以手动实现这个过程（后面会详细说明）

---

## 🤖 第四步：初始化语言模型和搜索工具

### 📖 这是什么？

初始化后续步骤需要用到的工具：LLM 用于评估和生成，搜索工具用于网络检索。

### 💻 完整代码

```python
# ============================================
# 初始化 OpenAI 语言模型
# ============================================
llm = ChatOpenAI(
    model="gpt-4o-mini",  # 使用较小较快的模型
    max_tokens=1000,       # 最大输出长度
    temperature=0          # 温度为 0，输出更稳定
)

# ============================================
# 初始化搜索工具
# ============================================
search = DuckDuckGoSearchResults()
```

> **💡 代码解释**
> - `gpt-4o-mini` 是性价比较高的模型选择
> - `temperature=0` 让输出更可预测，适合评估任务
> - `DuckDuckGoSearchResults` 是免费的搜索工具，无需 API 密钥

---

## 🧠 第五步：定义核心功能组件

### 📖 这是什么？

这是 CRAG 系统的"大脑"部分，包含三个关键功能：
1. **检索评估器** - 判断检索结果是否相关
2. **知识提炼器** - 从文档中提取关键信息
3. **查询重写器** - 优化网络搜索查询

### 💻 完整代码

```python
# ============================================
# 1. 检索评估器 - 评估文档相关性
# ============================================

# 定义 Pydantic 模型用于结构化输出
class RetrievalEvaluatorInput(BaseModel):
    relevance_score: float = Field(
        ..., 
        description="The relevance score of the document to the query. the score should be between 0 and 1."
    )

def retrieval_evaluator(query: str, document: str) -> float:
    """
    评估文档与查询的相关性
    
    参数：
        query: 用户查询
        document: 检索到的文档内容
    
    返回：
        相关性分数（0-1 之间）
    """
    prompt = PromptTemplate(
        input_variables=["query", "document"],
        template="在 0 到 1 的范围内，以下文档与查询的相关性如何？查询：{query}\nDocument: {document}\nRelevance score:"
    )
    chain = prompt | llm.with_structured_output(RetrievalEvaluatorInput)
    input_variables = {"query": query, "document": document}
    result = chain.invoke(input_variables).relevance_score
    return result


# ============================================
# 2. 知识提炼器 - 提取关键信息
# ============================================

class KnowledgeRefinementInput(BaseModel):
    key_points: str = Field(..., description="The document to extract key information from.")

def knowledge_refinement(document: str):
    """
    从文档中提取关键信息
    
    参数：
        document: 需要提炼的文档内容
    
    返回：
        关键信息列表
    """
    prompt = PromptTemplate(
        input_variables=["document"],
        template="以要点形式提取以下文档中的关键信息:\n{document}\nKey points:"
    )
    chain = prompt | llm.with_structured_output(KnowledgeRefinementInput)
    input_variables = {"document": document}
    result = chain.invoke(input_variables).key_points
    return [point.strip() for point in result.split('\n') if point.strip()]


# ============================================
# 3. 查询重写器 - 优化搜索查询
# ============================================

class QueryRewriterInput(BaseModel):
    query: str = Field(..., description="The query to rewrite.")

def rewrite_query(query: str) -> str:
    """
    重写查询以更适合网络搜索
    
    参数：
        query: 原始查询
    
    返回：
        重写后的查询
    """
    prompt = PromptTemplate(
        input_variables=["query"],
        template="重写以下查询以使其更适合网络搜索:\n{query}\nRewritten query:"
    )
    chain = prompt | llm.with_structured_output(QueryRewriterInput)
    input_variables = {"query": query}
    return chain.invoke(input_variables).query.strip()
```

> **💡 代码解释**
> - `with_structured_output()` 让 LLM 返回结构化的 JSON 数据
> - `Pydantic BaseModel` 定义输出格式
> - 三个函数分别对应 CRAG 的三个核心能力
>
> **⚠️ 新手注意**
> - `relevance_score` 分数范围是 0-1，后面会根据分数决定策略
> - 分数阈值：>0.7 高相关，<0.3 低相关，中间为模糊

---

## 🔧 第六步：辅助函数

### 📖 这是什么？

解析搜索结果的工具函数，将 JSON 格式的搜索结果转换为可用格式。

### 💻 完整代码

```python
def parse_search_results(results_string: str):
    """
    解析搜索结果的 JSON 字符串
    
    参数：
        results_string: JSON 格式的搜索结果字符串
    
    返回：
        包含标题和链接的元组列表
    """
    import json
    try:
        # 尝试解析 JSON 字符串
        results = json.loads(results_string)
        # 从每个结果中提取并返回标题和链接
        return [(result.get('title', 'Untitled'), result.get('link', '')) for result in results]
    except json.JSONDecodeError:
        # 通过返回空列表处理 JSON 解码错误
        print("Error parsing search results. Returning empty list.")
        return []
```

---

## 🔄 第七步：实现 CRAG 核心流程

### 📖 这是什么？

这里是 CRAG 的核心逻辑，包含文档检索、评估、知识获取和响应生成。

### 💻 完整代码

```python
# ============================================
# CRAG 子函数
# ============================================

def retrieve_documents(query: str, faiss_index, k: int = 3):
    """
    使用 FAISS 索引根据查询检索文档
    
    参数：
        query: 查询字符串
        faiss_index: FAISS 向量索引
        k: 检索的文档数量，默认 3 个
    
    返回：
        检索到的文档内容列表
    """
    docs = faiss_index.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]


def evaluate_documents(query: str, documents: list):
    """
    评估文档与查询的相关性
    
    参数：
        query: 查询字符串
        documents: 文档内容列表
    
    返回：
        相关性分数列表
    """
    return [retrieval_evaluator(query, doc) for doc in documents]


def perform_web_search(query: str):
    """
    执行网络搜索并提炼结果
    
    参数：
        query: 查询字符串
    
    返回：
        精炼后的知识列表和来源元组列表
    """
    # 重写查询
    rewritten_query = rewrite_query(query)
    # 执行搜索
    web_results = search.run(rewritten_query)
    # 提炼知识
    web_knowledge = knowledge_refinement(web_results)
    # 解析来源
    sources = parse_search_results(web_results)
    return web_knowledge, sources


def generate_response(query: str, knowledge: str, sources: list) -> str:
    """
    基于知识生成最终回答
    
    参数：
        query: 原始查询
        knowledge: 精炼后的知识
        sources: 来源列表
    
    返回：
        生成的回答
    """
    response_prompt = PromptTemplate(
        input_variables=["query", "knowledge", "sources"],
        template="根据以下知识回答问题。在答案末尾包含源及其链接（如果有）:\n查询：{query}\nKnowledge: {knowledge}\n源：{sources}\nAnswer:"
    )
    input_variables = {
        "query": query,
        "knowledge": knowledge,
        "sources": "\n".join([f"{title}: {link}" if link else title for title, link in sources])
    }
    response_chain = response_prompt | llm
    return response_chain.invoke(input_variables).content
```

---

## 🎯 第八步：完整的 CRAG 流程

### 📖 这是什么？

将所有组件组合成完整的 CRAG 处理流程。

### 💻 完整代码

```python
def crag_process(query: str, faiss_index):
    """
    完整的 CRAG 处理流程
    
    参数：
        query: 用户查询
        faiss_index: FAISS 向量索引
    
    返回：
        生成的回答
    """
    print(f"\nProcessing query: {query}")
    
    # ========== 步骤 1：检索并评估文档 ==========
    retrieved_docs = retrieve_documents(query, faiss_index)
    eval_scores = evaluate_documents(query, retrieved_docs)
    
    print(f"\n检索到 {len(retrieved_docs)} 个文档")
    print(f"评估分数：{eval_scores}")
    
    # ========== 步骤 2：基于评估分数确定操作 ==========
    max_score = max(eval_scores)
    sources = []
    
    if max_score > 0.7:
        # 情况 1：高相关性 - 直接使用检索到的文档
        print("\n操作：正确 - 使用检索到的文档")
        best_doc = retrieved_docs[eval_scores.index(max_score)]
        final_knowledge = best_doc
        sources.append(("Retrieved document", ""))
        
    elif max_score < 0.3:
        # 情况 2：低相关性 - 执行网络搜索
        print("\n操作：不正确 - 执行网络搜索")
        final_knowledge, sources = perform_web_search(query)
        
    else:
        # 情况 3：模糊 - 结合检索到的文档和网络搜索
        print("\n操作：模糊 - 结合检索到的文档和网络搜索")
        best_doc = retrieved_docs[eval_scores.index(max_score)]
        # 精炼检索到的知识
        retrieved_knowledge = knowledge_refinement(best_doc)
        # 执行网络搜索
        web_knowledge, web_sources = perform_web_search(query)
        # 合并知识
        final_knowledge = "\n".join(retrieved_knowledge + web_knowledge)
        sources = [("Retrieved document", "")] + web_sources
    
    print("\n最终知识：")
    print(final_knowledge)
    
    print("\n源：")
    for title, link in sources:
        print(f"{title}: {link}" if link else title)
    
    # ========== 步骤 3：生成响应 ==========
    print("\n生成响应中...")
    response = generate_response(query, final_knowledge, sources)
    print("\n响应已生成")
    
    return response
```

> **💡 代码解释**
> - 三种情况对应不同的检索质量：
>   - **>0.7 高相关**：直接用内部知识
>   - **<0.3 低相关**：放弃内部知识，用网络搜索
>   - **0.3-0.7 模糊**：两者结合
> - 打印语句帮助你跟踪处理过程

---

## 🧪 第九步：测试 CRAG 系统

### 💻 完整代码

```python
# ============================================
# 测试 1：与文档高度相关的问题
# ============================================
query = "气候变化的主要原因是什么？"
result = crag_process(query, vectorstore)
print(f"查询：{query}")
print(f"答案：{result}")

# ============================================
# 测试 2：与文档低相关的问题（测试网络搜索）
# ============================================
query = "how did harry beat quirrell?"
result = crag_process(query, vectorstore)
print(f"查询：{query}")
print(f"答案：{result}")
```

> **📊 预期输出示例**
> ```
> Processing query: 气候变化的主要原因是什么？
> 
> 检索到 3 个文档
> 评估分数：[0.85, 0.72, 0.45]
> 
> 操作：正确 - 使用检索到的文档
> 
> 最终知识：
> [文档内容...]
> 
> 响应已生成
> 答案：气候变化的主要原因是温室气体排放...
> ```

---

## ⚠️ 常见问题与调试

### Q1: 评估分数阈值如何选择？

**建议**：
- 0.7 和 0.3 是经验值，可以根据实际效果调整
- 如果系统太多使用网络搜索，可以提高阈值
- 如果系统太依赖内部知识，可以降低阈值

### Q2: 网络搜索速度慢怎么办？

**解决方案**：
- 网络搜索确实需要几秒到十几秒
- 可以添加缓存机制
- 或者限制只在必要时搜索

### Q3: 如何处理多轮对话？

**解决方案**：
- 在查询中包含对话历史
- 使用 LangChain 的 `ConversationBufferMemory`

---

## 📚 总结

### 核心要点回顾

1. **CRAG 核心思想**：检索后评估，动态调整策略
2. **三种策略**：高相关直接用、低相关去搜索、模糊就结合
3. **关键组件**：评估器、提炼器、重写器

### 进阶方向

1. **自定义评估器**：训练专门的分类模型
2. **多源检索**：结合多个知识库
3. **流式输出**：边检索边生成

---

## 🔗 相关资源

- [LangChain 官方文档](https://python.langchain.com/)
- [FAISS 向量库文档](https://faiss.ai/)
- [原始 CRAG 论文](https://arxiv.org/abs/2401.15884)

<div style="text-align: center;">
<img src="../images/crag.svg" alt="Corrective RAG" style="width:80%; height:auto;">
</div>
