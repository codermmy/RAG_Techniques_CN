# 🌟 新手入门：自适应检索（Adaptive RAG）

> **💡 给新手的说明**
> - **难度等级**：⭐⭐⭐⭐（中高级，需要较好的 Python 和 RAG 基础）
> - **预计学习时间**：70-90 分钟
> - **前置知识**：了解 RAG 基本流程、Prompt Engineering、向量检索
> - **本教程特色**：包含查询分类详解、四种策略对比、完整类设计
>
> **📚 什么是自适应检索？** 想象你去图书馆问问题：
> - 问事实性问题（"地球周长多少？"）→ 图书管理员给你查百科全书
> - 问分析性问题（"气候变化如何影响农业？"）→ 图书管理员给你找多本相关书籍
> - 问观点性问题（"人工智能对人类是好事吗？"）→ 图书管理员给你找不同立场的文章
>
> 自适应 RAG 就是这样一位"聪明的图书管理员"，会根据你的问题类型，用不同的方式帮你找答案！

---

## 📖 核心概念理解

### 通俗理解：看人下菜碟的智慧

**传统 RAG 的问题**：
```
不管什么问题，都用同样的方式检索：
❌ "地球周长多少？" → 检索一堆泛泛而谈的文章
❌ "气候变化如何影响经济？" → 只检索到一篇最相关的，不够全面
❌ "AI 是好事还是坏事？" → 只检索到单一观点
```

**自适应 RAG 的解决方案**：
```
先分类问题类型，再用不同策略：

📍 事实性问题 → 精确检索（快速找到准确答案）
📊 分析性问题 → 多角度检索（覆盖多个方面）
💭 观点性问题 → 多观点检索（呈现不同立场）
🎯 情境性问题 → 个性化检索（结合用户背景）
```

### 四种查询类型详解

| 类型 | 特点 | 例子 | 最佳策略 |
|------|------|------|----------|
| **事实性 (Factual)** | 寻求具体、可验证的信息 | "地球到太阳多远？" | 精确检索，重准确性 |
| **分析性 (Analytical)** | 需要全面分析或解释 | "气候变化如何影响农业？" | 多子查询，重全面性 |
| **观点性 (Opinion)** | 主观议题或寻求不同观点 | "AI 对人类有益吗？" | 多观点，重多样性 |
| **情境性 (Contextual)** | 依赖用户特定上下文 | "以我的预算该买什么车？" | 结合上下文，重个性化 |

### 生活化比喻

| 场景 | 传统 RAG | 自适应 RAG |
|------|---------|-----------|
| 🏥 **看病** | 不管什么病都开同样检查 | 根据症状选择不同检查项目 |
| 🍽️ **点餐** | 给每个客人同样的套餐 | 根据客人口味推荐不同菜品 |
| 🎓 **教学** | 对所有学生用同样方法教 | 根据学生特点调整教学方式 |
| 🚗 **导航** | 总是推荐同一条路线 | 根据目的地、路况、偏好推荐 |

### 核心术语解释

| 术语 | 通俗解释 | 技术含义 |
|------|----------|----------|
| **查询分类 (Query Classification)** | 判断问题属于什么类型 | 用 LLM 或分类器判断查询类别 |
| **检索策略 (Retrieval Strategy)** | 找资料的方法 | 不同的检索算法和参数配置 |
| **子查询 (Sub-query)** | 把大问题拆成小问题 | Query Decomposition 技术 |
| **LLM 增强 (LLM Enhancement)** | 用 AI 改进查询或结果 | 用语言模型优化检索过程 |
| **动态调整 (Dynamic Adjustment)** | 根据情况灵活变化 | 自适应系统的核心特性 |

---

## 🛠️ 第一步：安装必要的包

### 📖 这是什么？
构建自适应 RAG 系统需要的基础工具。

### 💻 完整代码

```python
# 安装所需的包
# ⚠️ faiss-cpu 是 Facebook 开源的向量搜索库
!pip install faiss-cpu langchain langchain-openai python-dotenv
```

```python
# 克隆仓库以访问辅助函数
!git clone https://github.com/NirDiamant/RAG_TECHNIQUES.git
import sys
sys.path.append('RAG_TECHNIQUES')
```

```python
# 导入必要的库
import os
import sys
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate  # 提示模板
from langchain.vectorstores import FAISS  # 向量存储
from langchain.embeddings import OpenAIEmbeddings  # OpenAI 嵌入
from langchain.text_splitter import CharacterTextSplitter  # 文本分割器
from langchain_core.retrievers import BaseRetriever  # 检索器基类
from typing import Dict, Any, List  # 类型提示
from langchain.docstore.document import Document  # 文档对象
from langchain_openai import ChatOpenAI  # OpenAI 聊天模型
from langchain_core.pydantic_v1 import BaseModel, Field  # Pydantic 模型

# 导入辅助函数
from helper_functions import *
from evaluation.evalute_rag import *

# 从.env 文件加载环境变量
load_dotenv()

# 设置 OpenAI API 密钥
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
```

> **💡 代码解释**
> - `FAISS`：Facebook AI Similarity Search，高效的向量检索库
> - `PromptTemplate`：用于创建可复用的提示词模板
> - `BaseModel, Field`：Pydantic 库，用于定义数据结构
> - `BaseRetriever`：LangChain 的检索器接口，继承它可以获得标准功能
>
> **⚠️ 新手注意**
> - `faiss-cpu` 适合学习，生产环境可能需要 `faiss-gpu`
> - 确保 `.env` 文件中有 `OPENAI_API_KEY`

---

## 🛠️ 第二步：实现查询分类器

### 📖 这是什么？
**这是系统的"大脑"！** 它负责判断用户的问题属于什么类型，然后选择对应的策略。

### 💡 分类原理

```
用户问题："地球到太阳的距离是多少？"
         │
         ▼
    ┌─────────────┐
    │ 查询分类器   │
    │ (LLM + Prompt)│
    └──────┬──────┘
           │
           ▼
    分析过程：
    - 这个问题有唯一正确答案吗？→ 是
    - 需要多角度分析吗？→ 否
    - 涉及主观观点吗？→ 否
    - 需要用户上下文吗？→ 否
           │
           ▼
    分类结果：Factual（事实性）
```

### 💻 完整代码

```python
# ==================== 定义分类输出的 Pydantic 模型 ====================
class categories_options(BaseModel):
    """
    查询分类的可选类别。
    使用 Pydantic 确保输出格式正确。
    """
    category: str = Field(
        description="查询的类别，可选值：Factual（事实性）、Analytical（分析性）、Opinion（观点性）或 Contextual（情境性）",
        example="Factual"
    )


# ==================== 实现查询分类器类 ====================
class QueryClassifier:
    """
    查询分类器：使用 LLM 将用户查询分类为四种类型之一。

    🎯 四种类型：
    - Factual（事实性）：寻求具体、可验证的信息
    - Analytical（分析性）：需要全面分析或解释
    - Opinion（观点性）：关于主观事项或寻求不同观点
    - Contextual（情境性）：依赖于用户特定上下文
    """

    def __init__(self):
        """初始化分类器。"""
        # 使用 GPT-4o 模型（支持结构化输出）
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)

        # 创建分类提示模板
        self.prompt = PromptTemplate(
            input_variables=["query"],  # 输入变量名
            template="""将以下查询分类为以下类别之一：

- Factual（事实性）：寻求具体、可验证的信息，通常有明确的正确答案
- Analytical（分析性）：需要全面分析、解释或多角度探索的问题
- Opinion（观点性）：关于主观事项、价值判断或需要呈现不同观点的问题
- Contextual（情境性）：依赖于用户特定背景、偏好或上下文的问题

查询：{query}

类别："""
        )

        # 创建 LLM 链（提示 + 结构化输出）
        self.chain = self.prompt | self.llm.with_structured_output(categories_options)

    def classify(self, query: str) -> str:
        """
        对查询进行分类。

        Args:
            query: 用户查询。

        Returns:
            分类结果字符串（"Factual"、"Analytical"、"Opinion"或"Contextual"）。
        """
        print(f"🔍 正在分类查询...")
        result = self.chain.invoke(query)
        category = result.category
        print(f"📋 分类结果：{category}")
        return category


# ==================== 测试分类器 ====================
if __name__ == "__main__":
    classifier = QueryClassifier()

    # 测试不同类型的查询
    test_queries = [
        "地球到太阳的距离是多少？",  # 事实性
        "气候变化如何影响全球经济？",  # 分析性
        "人工智能对人类是好事还是坏事？",  # 观点性
        "以我的预算和喜好应该买什么车？"  # 情境性
    ]

    for query in test_queries:
        print(f"\n查询：{query}")
        category = classifier.classify(query)
        print(f"类别：{category}\n")
        print("-" * 50)
```

> **💡 代码解释**
>
> **Pydantic 模型的作用**：
> ```python
> # 定义期望的输出格式
> class categories_options(BaseModel):
>     category: str  # 必须有一个 category 字段
>
> # LLM 会按照这个格式输出
> # 输出示例：{"category": "Factual"}
> ```
>
> **PromptTemplate 的设计**：
> ```python
> template = """
> 1. 先解释每个类别的含义（帮助 LLM 理解）
> 2. 给出查询
> 3. 让 LLM 填空
> """
> ```
>
> **⚠️ 新手注意**
> - `temperature=0`：让分类结果更稳定、一致
> - `with_structured_output()`：确保 LLM 输出符合 Pydantic 模型
> - 可以添加更多类别，但需要相应修改分类逻辑

### ❓ 常见问题

**Q1: 为什么用 LLM 而不是训练一个分类器？**
```
LLM 分类的优势：
✅ 无需训练数据
✅ 可以处理未见过的查询类型
✅ 可以解释分类理由
✅ 容易调整类别定义

训练分类器的优势：
✅ 速度更快
✅ 成本更低
✅ 可以更精确

选择取决于你的需求！
```

**Q2: 分类准确吗？**
```
在测试中，GPT-4 对这四类的分类准确率约 85-95%。
提高准确率的方法：
1. 提供更详细的类别说明
2. 给 Few-shot 示例
3. 用更好的模型
```

---

## 🛠️ 第三步：实现基础检索器

### 📖 这是什么？
所有高级策略都继承自这个基础检索器。它提供最基本的相似性搜索功能。

### 💻 完整代码

```python
class BaseRetrievalStrategy:
    """
    基础检索策略：所有其他策略的父类。

    📝 提供基本功能：
    - 文本分块
    - 创建向量存储
    - 基础相似性搜索

    子类可以继承并重写 retrieve 方法实现不同策略。
    """

    def __init__(self, texts: List[str]):
        """
        初始化基础检索器。

        Args:
            texts: 文本内容列表，用于构建知识库。
        """
        # 创建 Embeddings（使用 OpenAI）
        self.embeddings = OpenAIEmbeddings()

        # 文本分割器
        text_splitter = CharacterTextSplitter(
            chunk_size=800,      # 每块 800 字符
            chunk_overlap=0       # 无重叠（简单示例）
        )

        # 创建文档
        self.documents = text_splitter.create_documents(texts)

        # 创建向量存储
        self.db = FAISS.from_documents(self.documents, self.embeddings)

        # 初始化 LLM（用于后续增强）
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        基础检索：使用相似性搜索返回最相关的 k 个文档。

        Args:
            query: 查询文本。
            k: 返回的文档数量（默认 4）。

        Returns:
            相关文档列表。
        """
        return self.db.similarity_search(query, k=k)
```

> **💡 代码解释**
>
> **继承的设计模式**：
> ```
> BaseRetrievalStrategy（基础类）
>     ├── FactualRetrievalStrategy（事实性策略）
>     ├── AnalyticalRetrievalStrategy（分析性策略）
>     ├── OpinionRetrievalStrategy（观点性策略）
>     └── ContextualRetrievalStrategy（情境性策略）
>
> 所有子类共享：
> - embeddings
> - documents
> - db (向量存储)
> - llm
>
> 每个子类有自己的 retrieve 实现
> ```
>
> **⚠️ 新手注意**
> - `CharacterTextSplitter`：按字符分割，适合英文
> - 中文建议用 `RecursiveCharacterTextSplitter` 或专门的分词器

---

## 🛠️ 第四步：实现事实性检索策略

### 📖 这是什么？
针对事实性问题的优化策略：**先增强查询，再精确检索**。

### 💡 工作原理

```
原始查询："地球周长多少？"
         │
         ▼
    ┌─────────────┐
    │ LLM 增强查询 │
    └──────┬──────┘
         │
         ▼
增强后："地球的赤道周长和极周长分别是多少公里？"
         │
         ▼
    ┌─────────────┐
    │ 检索文档     │ → 更多候选（k*2）
    └──────┬──────┘
         │
         ▼
    ┌─────────────┐
    │ LLM 评分排序 │ → 按相关性打分
    └──────┬──────┘
         │
         ▼
    返回最相关的前 k 个文档
```

### 💻 完整代码

```python
# 定义评分输出的 Pydantic 模型
class relevant_score(BaseModel):
    """文档相关性评分。"""
    score: float = Field(
        description="文档与查询的相关性得分，范围 1-10",
        example=8.0
    )


class FactualRetrievalStrategy(BaseRetrievalStrategy):
    """
    事实性检索策略：针对事实性问题优化。

    🎯 特点：
    1. 使用 LLM 增强原始查询（更精确）
    2. 检索更多候选文档
    3. 用 LLM 对每个文档评分排序

    适用场景：有明确答案的问题
    例如："地球周长多少？"、"水的沸点是多少？"
    """

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        执行事实性检索。

        Args:
            query: 查询文本。
            k: 返回的文档数量（默认 4）。

        Returns:
            最相关的 k 个文档。
        """
        print("🔍 执行事实性检索...")

        # ==================== 步骤 1: 使用 LLM 增强查询 ====================
        enhanced_query_prompt = PromptTemplate(
            input_variables=["query"],
            template="增强这个事实性查询以获得更好的信息检索效果，使其更精确和具体：{query}"
        )
        query_chain = enhanced_query_prompt | self.llm
        enhanced_query = query_chain.invoke(query).content
        print(f"✨ 增强后的查询：{enhanced_query}")

        # ==================== 步骤 2: 使用增强后的查询检索文档 ====================
        # 检索更多候选（k*2），给后续排序留有余地
        docs = self.db.similarity_search(enhanced_query, k=k*2)
        print(f"📚 检索到 {len(docs)} 个候选文档")

        # ==================== 步骤 3: 使用 LLM 对文档进行相关性排序 ====================
        ranking_prompt = PromptTemplate(
            input_variables=["query", "doc"],
            template="在 1-10 的评分标准上，此文​​档与查询 '{query}' 的相关性如何？\n文档内容：{doc}\n相关性得分（只返回数字）："
        )
        ranking_chain = ranking_prompt | self.llm.with_structured_output(relevant_score)

        ranked_docs = []
        print("📊 正在对文档评分...")
        for doc in docs:
            input_data = {
                "query": enhanced_query,
                "doc": doc.page_content
            }
            try:
                score_result = ranking_chain.invoke(input_data)
                score = float(score_result.score)
                ranked_docs.append((doc, score))
                print(f"  文档得分：{score}")
            except Exception as e:
                print(f"  评分失败：{e}")
                ranked_docs.append((doc, 5.0))  # 默认中等分数

        # ==================== 步骤 4: 按分数排序并返回前 k 个 ====================
        ranked_docs.sort(key=lambda x: x[1], reverse=True)  # 降序排序

        # 只返回文档，不要分数
        result_docs = [doc for doc, _ in ranked_docs[:k]]
        print(f"✅ 返回得分最高的 {k} 个文档")

        return result_docs
```

> **💡 代码解释**
>
> **为什么要增强查询？**
> ```
> 原始查询可能：
> - 太模糊："地球周长"
>
> 增强后更精确：
> - "地球的赤道周长和极周长分别是多少公里或英里？"
>
> 更精确的查询 → 更准确的检索结果
> ```
>
> **为什么检索 k*2 个再排序？**
> ```
> 向量检索可能不够精确，用 LLM 二次筛选：
> 1. 向量检索：快速筛选出 k*2 个候选
> 2. LLM 评分：精确评估每个候选的相关性
> 3. 选 top-k：返回最好的结果
> ```
>
> **⚠️ 新手注意**
> - 这会增加 LLM 调用次数（每个文档一次）
> - 可以批量评分减少调用次数
> - 注意 API 费用

---

## 🛠️ 第五步：实现分析性检索策略

### 📖 这是什么？
针对分析性问题的策略：**把大问题拆成小问题，全面覆盖**。

### 💡 工作原理

```
原始查询："气候变化如何影响农业？"
         │
         ▼
    ┌─────────────┐
    │ LLM 生成子查询 │
    └──────┬──────┘
         │
         ▼
子查询 1: "气候变化对作物产量的影响"
子查询 2: "气候变化对农业水资源的影响"
子查询 3: "气候变化对病虫害的影响"
子查询 4: "气候变化对农业经济的影响"
         │
         ▼
    ┌─────────────┐
    │ 分别检索每个子查询 │
    └──────┬──────┘
         │
         ▼
    ┌─────────────┐
    │ 合并所有结果   │
    └──────┬──────┘
         │
         ▼
    ┌─────────────┐
    │ LLM 选择多样化结果 │
    └──────┬──────┘
         │
         ▼
    返回全面覆盖各方面的文档
```

### 💻 完整代码

```python
# 定义子查询输出的 Pydantic 模型
class SubQueries(BaseModel):
    """子查询列表。"""
    sub_queries: List[str] = Field(
        description="用于全面分析主查询的子问题列表",
        example=["气候变化的原因是什么？", "气候变化的影响有哪些？"]
    )

# 文档选择输出的 Pydantic 模型
class SelectedIndices(BaseModel):
    """选中文档索引列表。"""
    indices: List[int] = Field(
        description="选中文档的索引列表",
        example=[0, 1, 2, 3]
    )


class AnalyticalRetrievalStrategy(BaseRetrievalStrategy):
    """
    分析性检索策略：针对分析性问题优化。

    🎯 特点：
    1. 使用 LLM 生成多个子查询（分解问题）
    2. 分别检索每个子查询
    3. 合并结果并选择最具多样性和相关性的文档

    适用场景：需要全面分析的问题
    例如："气候变化如何影响经济？"、"AI 对就业市场的影响"
    """

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        执行分析性检索。

        Args:
            query: 查询文本。
            k: 返回的文档数量（默认 4）。

        Returns:
            最相关和多样化的 k 个文档。
        """
        print("🔍 执行分析性检索...")

        # ==================== 步骤 1: 使用 LLM 生成子查询 ====================
        sub_queries_prompt = PromptTemplate(
            input_variables=["query", "k"],
            template="为以下查询生成{k}个子问题，以便全面分析这个主题的各个方面：{query}"
        )

        llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
        sub_queries_chain = sub_queries_prompt | llm.with_structured_output(SubQueries)

        input_data = {"query": query, "k": k}
        sub_queries_result = sub_queries_chain.invoke(input_data)
        sub_queries = sub_queries_result.sub_queries

        print(f"✨ 生成的子查询：")
        for i, sq in enumerate(sub_queries, 1):
            print(f"  {i}. {sq}")

        # ==================== 步骤 2: 为每个子查询检索文档 ====================
        all_docs = []
        for sub_query in sub_queries:
            # 每个子查询检索 2 个文档
            docs = self.db.similarity_search(sub_query, k=2)
            all_docs.extend(docs)

        print(f"📚 共检索到 {len(all_docs)} 个文档")

        # ==================== 步骤 3: 使用 LLM 确保多样性和相关性 ====================
        diversity_prompt = PromptTemplate(
            input_variables=["query", "docs", "k"],
            template="""为查询 '{query}' 选择最多样化和最相关的{k}个文档。

多样性意味着文档应该涵盖主题的不同方面，而不是重复相同的内容。

文档列表：
{docs}

只返回所选文档的索引作为整数列表（如 [0, 2, 4, 5]）。"""
        )
        diversity_chain = diversity_prompt | self.llm.with_structured_output(SelectedIndices)

        # 准备文档文本（只显示前 50 字符，避免太长）
        docs_text = "\n".join([
            f"{i}: {doc.page_content[:50]}..."
            for i, doc in enumerate(all_docs)
        ])

        input_data = {"query": query, "docs": docs_text, "k": k}
        selected_indices_result = diversity_chain.invoke(input_data)
        selected_indices = selected_indices_result.indices

        print(f"🎯 选择了 {len(selected_indices)} 个多样化和相关的文档")

        # ==================== 步骤 4: 返回选中的文档 ====================
        # 过滤越界索引（安全处理）
        valid_indices = [i for i in selected_indices if i < len(all_docs)]
        result_docs = [all_docs[i] for i in valid_indices]

        return result_docs
```

> **💡 代码解释**
>
> **为什么要分解成子查询？**
> ```
> 大问题："气候变化如何影响农业？"
>
> 直接检索可能：
> - 只找到泛泛而谈的文章
> - 遗漏某些重要方面
>
> 分解后：
> - 子查询 1: "对作物产量的影响" → 找到具体数据
> - 子查询 2: "对水资源的影响" → 找到灌溉相关信息
> - 子查询 3: "对病虫害的影响" → 找到农业灾害信息
> - 子查询 4: "对农民收入的影响" → 找到经济影响分析
>
> 结果更全面！
> ```
>
> **⚠️ 新手注意**
> - 子查询数量影响检索广度和成本
> - 可以适当调整 k 值
> - 多样性选择可能过滤掉一些相关但相似的文档

---

## 🛠️ 第六步：实现观点性检索策略

### 📖 这是什么？
针对观点性问题的策略：**主动寻找不同立场和观点**。

### 💡 工作原理

```
原始查询："AI 对人类是好事还是坏事？"
         │
         ▼
    ┌─────────────┐
    │ LLM 识别潜在观点 │
    └──────┬──────┘
         │
         ▼
观点 1: "AI 带来经济繁荣和效率提升"（乐观）
观点 2: "AI 威胁就业和隐私"（悲观）
观点 3: "AI 影响取决于如何监管"（中立）
         │
         ▼
    ┌─────────────┐
    │ 为每个观点检索文档 │
    └──────┬──────┘
         │
         ▼
    ┌─────────────┐
    │ 选择代表不同观点的文档 │
    └──────┬──────┘
         │
         ▼
    返回涵盖多种观点的文档
```

### 💻 完整代码

```python
class OpinionRetrievalStrategy(BaseRetrievalStrategy):
    """
    观点性检索策略：针对观点性问题优化。

    🎯 特点：
    1. 使用 LLM 识别主题的不同观点/立场
    2. 为每个观点检索代表性文档
    3. 确保最终结果涵盖多样化的观点

    适用场景：主观议题、价值判断、争议话题
    例如："AI 是好事还是坏事？"、"应该支持全球化吗？"
    """

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """
        执行观点性检索。

        Args:
            query: 查询文本。
            k: 返回的文档数量（默认 3）。

        Returns:
            代表不同观点的 k 个文档。
        """
        print("🔍 执行观点性检索...")

        # ==================== 步骤 1: 使用 LLM 识别潜在观点 ====================
        viewpoints_prompt = PromptTemplate(
            input_variables=["query", "k"],
            template="识别关于主题 '{query}' 的{k}个不同观点或视角。每个观点应该代表一个独特的立场或角度。"
        )
        viewpoints_chain = viewpoints_prompt | self.llm
        input_data = {"query": query, "k": k}
        viewpoints_result = viewpoints_chain.invoke(input_data)
        viewpoints = viewpoints_result.content.split('\n')

        # 清理空行
        viewpoints = [v.strip() for v in viewpoints if v.strip()]

        print(f"✨ 识别的观点：")
        for i, viewpoint in enumerate(viewpoints, 1):
            print(f"  {i}. {viewpoint}")

        # ==================== 步骤 2: 为每个观点检索文档 ====================
        all_docs = []
        for viewpoint in viewpoints:
            # 将查询与观点结合进行检索
            combined_query = f"{query} {viewpoint}"
            docs = self.db.similarity_search(combined_query, k=2)
            all_docs.extend(docs)

        print(f"📚 共检索到 {len(all_docs)} 个文档")

        # ==================== 步骤 3: 使用 LLM 分类并选择多样化观点 ====================
        opinion_prompt = PromptTemplate(
            input_variables=["query", "docs", "k"],
            template="""将这些文档分类为关于 '{query}' 的不同观点，并选择{k}个最具代表性和多样化的观点。

确保选择的文档涵盖不同的立场和视角，而不仅仅是重复相同观点。

文档列表：
{docs}

返回所选文档的索引（整数列表，如 [0, 2, 5]）。"""
        )
        opinion_chain = opinion_prompt | self.llm.with_structured_output(SelectedIndices)

        # 准备文档文本（显示前 100 字符）
        docs_text = "\n".join([
            f"{i}: {doc.page_content[:100]}..."
            for i, doc in enumerate(all_docs)
        ])

        input_data = {"query": query, "docs": docs_text, "k": k}
        selected_indices_result = opinion_chain.invoke(input_data)

        # 处理返回的索引（可能是字符串或列表）
        if isinstance(selected_indices_result.indices, str):
            # 如果是字符串，解析成整数列表
            selected_indices = [
                int(i) for i in selected_indices_result.indices.split()
                if i.isdigit()
            ]
        else:
            selected_indices = selected_indices_result.indices

        print(f"🎯 选择了 {len(selected_indices)} 个代表不同观点的文档")

        # ==================== 步骤 4: 返回选中的文档 ====================
        valid_indices = [i for i in selected_indices if i < len(all_docs)]
        result_docs = [all_docs[i] for i in valid_indices]

        return result_docs
```

> **💡 代码解释**
>
> **为什么观点性检索特殊？**
> ```
> 事实性问题：追求唯一正确答案
> 观点性问题：需要呈现多种立场
>
> 例如"AI 的影响"：
> - 乐观观点：提高效率、创造新机会
> - 悲观观点：失业风险、隐私威胁
> - 中立观点：取决于如何治理
>
> 好的回答应该涵盖多角度！
> ```
>
> **⚠️ 新手注意**
> - 观点数量 k 不宜太多，3-5 个为宜
> - 某些观点可能没有足够代表性的文档
> - 需要平衡观点的代表性和文档质量

---

## 🛠️ 第七步：实现情境性检索策略

### 📖 这是什么？
针对情境性问题的策略：**结合用户特定上下文提供个性化结果**。

### 💡 工作原理

```
用户查询："我应该买什么保险？"
用户上下文：30 岁，已婚，有 1 个孩子，年收入 50 万
         │
         ▼
    ┌─────────────┐
    │ LLM 情境化查询 │
    └──────┬──────┘
         │
         ▼
情境化后："30 岁已婚有孩子的中产家庭应该买什么保险？
          需要考虑家庭责任和收入水平"
         │
         ▼
    ┌─────────────┐
    │ 检索文档     │
    └──────┬──────┘
         │
         ▼
    ┌─────────────┐
    │ LLM 评分（考虑上下文）│
    └──────┬──────┘
         │
         ▼
    返回最符合用户情况的文档
```

### 💻 完整代码

```python
class ContextualRetrievalStrategy(BaseRetrievalStrategy):
    """
    情境性检索策略：针对情境性问题优化。

    🎯 特点：
    1. 使用 LLM 将用户上下文融入查询
    2. 用情境化后的查询检索
    3. 评分时同时考虑相关性和用户上下文匹配度

    适用场景：依赖用户背景的问题
    例如："我该买什么保险？"（需要知道年龄、收入等）
          "怎么学编程？"（需要知道基础、目标等）

    ⚠️ 注意：需要额外传入 user_context 参数
    """

    def retrieve(self, query: str, k: int = 4, user_context: str = None) -> List[Document]:
        """
        执行情境性检索。

        Args:
            query: 查询文本。
            k: 返回的文档数量（默认 4）。
            user_context: 用户特定的上下文信息（可选）。

        Returns:
            最符合用户上下文的 k 个文档。
        """
        print("🔍 执行情境性检索...")
        print(f"📋 用户上下文：{user_context or '未提供'}")

        # ==================== 步骤 1: 使用 LLM 将用户上下文融入查询 ====================
        context_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""给定用户上下文：{context}

重新构建以下查询，使其最好地满足用户在此上下文中的需求：

原始查询：{query}

重构后的查询："""
        )
        context_chain = context_prompt | self.llm

        input_data = {
            "query": query,
            "context": user_context or "未提供具体上下文"
        }
        contextualized_query = context_chain.invoke(input_data).content
        print(f"✨ 情境化后的查询：{contextualized_query}")

        # ==================== 步骤 2: 使用情境化查询检索文档 ====================
        # 检索更多候选（k*2）
        docs = self.db.similarity_search(contextualized_query, k=k*2)
        print(f"📚 检索到 {len(docs)} 个候选文档")

        # ==================== 步骤 3: 使用 LLM 对文档评分（考虑用户上下文）====================
        ranking_prompt = PromptTemplate(
            input_variables=["query", "context", "doc"],
            template="""给定查询：'{query}'
和用户上下文：'{context}'

在 1-10 的评分标准上评价此文档的相关性和实用性：
文档内容：{doc}

相关性得分（只返回数字）："""
        )
        ranking_chain = ranking_prompt | self.llm.with_structured_output(relevant_score)

        print("📊 正在对文档评分（考虑上下文）...")
        ranked_docs = []
        for doc in docs:
            input_data = {
                "query": contextualized_query,
                "context": user_context or "未提供具体上下文",
                "doc": doc.page_content
            }
            try:
                score_result = ranking_chain.invoke(input_data)
                score = float(score_result.score)
                ranked_docs.append((doc, score))
            except Exception as e:
                print(f"  评分失败：{e}")
                ranked_docs.append((doc, 5.0))

        # ==================== 步骤 4: 按分数排序并返回前 k 个 ====================
        ranked_docs.sort(key=lambda x: x[1], reverse=True)

        result_docs = [doc for doc, _ in ranked_docs[:k]]
        print(f"✅ 返回最符合上下文的 {k} 个文档")

        return result_docs
```

> **💡 代码解释**
>
> **情境性检索的价值**：
> ```
> 没有上下文：
> "我该买什么保险？" → 泛泛的保险建议
>
> 有上下文：
> "30 岁已婚有孩子，年收入 50 万" →
> - 定期寿险（家庭责任）
> - 重疾险（收入保障）
> - 医疗险（补充社保）
> - 子女教育金（孩子未来）
>
> 结果更有针对性！
> ```
>
> **⚠️ 新手注意**
> - `user_context` 是关键，需要用户提供
> - 上下文质量直接影响结果质量
> - 可以设计 UI 引导用户提供上下文

---

## 🛠️ 第八步：整合自适应检索器

### 📖 这是什么？
这是系统的"指挥中心"，把分类器和所有策略整合在一起。

### 💻 完整代码

```python
class AdaptiveRetriever:
    """
    自适应检索器：根据查询类型自动选择合适的检索策略。

    🧠 工作原理：
    1. 接收用户查询
    2. 使用分类器判断查询类型
    3. 选择对应的检索策略
    4. 返回检索结果

    支持四种策略：
    - Factual：事实性检索
    - Analytical：分析性检索
    - Opinion：观点性检索
    - Contextual：情境性检索
    """

    def __init__(self, texts: List[str]):
        """
        初始化自适应检索器。

        Args:
            texts: 用于构建知识库的文本列表。
        """
        # 初始化分类器
        self.classifier = QueryClassifier()

        # 初始化四种策略
        self.strategies = {
            "Factual": FactualRetrievalStrategy(texts),       # 事实性
            "Analytical": AnalyticalRetrievalStrategy(texts),  # 分析性
            "Opinion": OpinionRetrievalStrategy(texts),       # 观点性
            "Contextual": ContextualRetrievalStrategy(texts)   # 情境性
        }

        print("✅ 自适应检索器初始化完成！")

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        获取与查询相关的文档。

        Args:
            query: 用户查询。

        Returns:
            相关文档列表。
        """
        # 步骤 1: 分类查询
        category = self.classifier.classify(query)
        print(f"\n📋 查询分类：{category}")

        # 步骤 2: 选择对应策略
        strategy = self.strategies[category]
        print(f"🎯 使用策略：{category}RetrievalStrategy")

        # 步骤 3: 执行检索
        if category == "Contextual":
            # 情境性检索需要额外处理（这里简化，实际应该从某处获取上下文）
            documents = strategy.retrieve(query)
        else:
            documents = strategy.retrieve(query)

        print(f"📚 检索到 {len(documents)} 个文档")
        return documents


# ==================== 定义 LangChain 兼容的检索器 ====================
class PydanticAdaptiveRetriever(BaseRetriever):
    """
    LangChain 兼容的自适应检索器包装类。

    📝 作用：
    - 继承 BaseRetriever 获得 LangChain 集成能力
    - 可以传递给 LangChain 的 Chain、Agent 等使用
    """
    adaptive_retriever: AdaptiveRetriever = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        """同步获取相关文档。"""
        return self.adaptive_retriever.get_relevant_documents(query)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """异步获取相关文档（简单委托给同步方法）。"""
        return self.get_relevant_documents(query)
```

> **💡 代码解释**
>
> **设计模式**：
> ```
> ┌─────────────────────────────────────┐
> │         AdaptiveRetriever           │
> │  (核心逻辑)                         │
> │                                     │
> │  - classifier: QueryClassifier     │
> │  - strategies: Dict[str, Strategy] │
> └─────────────────────────────────────┘
>                  │
>                  │ 包装
>                  ▼
> ┌─────────────────────────────────────┐
> │   PydanticAdaptiveRetriever         │
> │   (LangChain 兼容层)                │
> │                                     │
> │  - 继承 BaseRetriever              │
> │  - 可以传给 LangChain Chain 使用    │
> └─────────────────────────────────────┘
> ```
>
> **⚠️ 新手注意**
> - `Field(exclude=True)`：Pydantic 排除字段，不序列化
> - `arbitrary_types_allowed = True`：允许任意类型
> - 异步方法简单委托给同步方法，实际可以真正异步实现

---

## 🛠️ 第九步：实现自适应 RAG 系统

### 📖 这是什么？
最终的完整系统：结合自适应检索器和 LLM 生成答案。

### 💻 完整代码

```python
class AdaptiveRAG:
    """
    自适应 RAG 系统：完整的问答系统。

    🎯 完整流程：
    1. 接收用户问题
    2. 分类问题类型
    3. 选择合适的检索策略
    4. 检索相关文档
    5. 用 LLM 生成答案

    特点：根据问题类型自动调整检索方式，
    提供更准确、全面、适当的答案。
    """

    def __init__(self, texts: List[str]):
        """
        初始化自适应 RAG 系统。

        Args:
            texts: 用于构建知识库的文本列表。
        """
        # 初始化自适应检索器
        adaptive_retriever = AdaptiveRetriever(texts)

        # 包装成 LangChain 兼容检索器
        self.retriever = PydanticAdaptiveRetriever(adaptive_retriever=adaptive_retriever)

        # 初始化 LLM
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)

        # 创建自定义提示模板
        prompt_template = """使用以下上下文片段回答最后的问题。
如果您不知道答案，只需说您不知道即可，不要试图编造答案。
答案应该准确、简洁，同时涵盖问题的关键方面。

上下文：
{context}

问题：{question}
答案："""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # 创建 LLM 链
        self.llm_chain = prompt | self.llm

        print("✅ 自适应 RAG 系统初始化完成！")

    def answer(self, query: str) -> str:
        """
        回答问题。

        Args:
            query: 用户问题。

        Returns:
            LLM 生成的答案。
        """
        print(f"\n{'='*60}")
        print(f"📝 问题：{query}")
        print(f"{'='*60}")

        # 步骤 1: 检索相关文档
        docs = self.retriever.get_relevant_documents(query)

        # 步骤 2: 组合物上下文
        context = "\n\n".join([doc.page_content for doc in docs])

        # 步骤 3: 生成答案
        input_data = {"context": context, "question": query}
        response = self.llm_chain.invoke(input_data)

        print(f"\n💡 答案：{response.content}")
        print(f"{'='*60}\n")

        return response.content
```

> **💡 代码解释**
>
> **完整流程**：
> ```
> 用户问题
>    │
>    ▼
> ┌─────────────────┐
> │ AdaptiveRAG     │
> │ .answer()       │
> └────────┬────────┘
>          │
>          ▼
> ┌─────────────────┐
> │ 分类查询类型     │ → Factual/Analytical/Opinion/Contextual
> └────────┬────────┘
>          │
>          ▼
> ┌─────────────────┐
> │ 选择检索策略     │
> └────────┬────────┘
>          │
>          ▼
> ┌─────────────────┐
> │ 检索相关文档     │ → [doc1, doc2, doc3, doc4]
> └────────┬────────┘
>          │
>          ▼
> ┌─────────────────┐
> │ 组合物上下文     │ → "doc1\n\ndoc2\n\ndoc3\n\ndoc4"
> └────────┬────────┘
>          │
>          ▼
> ┌─────────────────┐
> │ LLM 生成答案     │
> └────────┬────────┘
>          │
>          ▼
>      返回答案
> ```
>
> **⚠️ 新手注意**
> - `temperature=0`：让答案更稳定、确定
> - 提示词强调"不知道就说不知道"，减少幻觉
> - 可以添加更多后处理（如引用来源）

---

## 🛠️ 第十步：完整演示

### 💻 完整代码

```python
# ==================== 准备示例文本 ====================
texts = [
    "地球是太阳系中第三颗行星，距离太阳约 1.496 亿公里（1 天文单位）。"
    "地球是目前已知唯一存在生命的星球，拥有适宜的温度、大气和液态水。",
    "气候变化的主要原因是人类活动，特别是化石燃料的燃烧，释放大量温室气体。",
    "全球变暖导致极端天气事件增加，包括热浪、干旱、洪水和强风暴。",
    "人工智能正在改变各行各业，从医疗诊断到自动驾驶，从客服到教育。",
    "关于 AI 的影响存在不同观点：乐观者认为将带来繁荣，悲观者担心失业和隐私问题。",
    "学习编程需要根据基础和目标选择合适的方法：自学适合有基础的人，培训班适合快速入门。",
    "保险选择应考虑个人情况：年龄、收入、家庭责任、健康状况等因素。"
]

# ==================== 初始化 RAG 系统 ====================
print("🚀 初始化自适应 RAG 系统...")
rag_system = AdaptiveRAG(texts)
print("✅ 系统就绪！\n")

# ==================== 测试四种不同类型的查询 ====================
print("\n" + "="*60)
print("🧪 测试 1: 事实性问题")
print("="*60)
factual_result = rag_system.answer("地球到太阳的距离是多少？")

print("\n" + "="*60)
print("🧪 测试 2: 分析性问题")
print("="*60)
analytical_result = rag_system.answer("气候变化如何影响天气？")

print("\n" + "="*60)
print("🧪 测试 3: 观点性问题")
print("="*60)
opinion_result = rag_system.answer("人工智能对社会的影响是正面还是负面？")

print("\n" + "="*60)
print("🧪 测试 4: 情境性问题")
print("="*60)
# 注意：情境性查询在实际应用中应该传入 user_context
# 这里简化演示
contextual_result = rag_system.answer("我应该如何学习编程？")

print("\n" + "="*60)
print("🎉 所有测试完成！")
print("="*60)
```

> **💡 预期输出**
> ```
> 🚀 初始化自适应 RAG 系统...
> ✅ 自适应检索器初始化完成！
> ✅ 自适应 RAG 系统初始化完成！
> ✅ 系统就绪！
>
> ============================================================
> 🧪 测试 1: 事实性问题
> ============================================================
> ============================================================
> 📝 问题：地球到太阳的距离是多少？
> ============================================================
> 🔍 正在分类查询...
> 📋 分类结果：Factual
> 🎯 使用策略：FactualRetrievalStrategy
> ...
> 💡 答案：地球到太阳的平均距离约为 1.496 亿公里（1 天文单位）。
> ============================================================
>
> ============================================================
> 🧪 测试 2: 分析性问题
> ============================================================
> ...（类似输出）
>
> ============================================================
> 🧪 测试 3: 观点性问题
> ============================================================
> ...
>
> ============================================================
> 🧪 测试 4: 情境性问题
> ============================================================
> ...
>
> ============================================================
> 🎉 所有测试完成！
> ============================================================
> ```

---

## 📊 可视化理解

### 自适应 RAG 完整架构

```
┌─────────────────────────────────────────────────────────────┐
│                    自适应 RAG 系统架构                       │
└─────────────────────────────────────────────────────────────┘

                         用户查询
                            │
                            ▼
                ┌─────────────────────┐
                │   查询分类器         │
                │  QueryClassifier   │
                └──────────┬──────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   ┌────────┐        ┌─────────┐        ┌─────────┐
   │Factual │        │Analytical│       │ Opinion │
   │事实性   │        │分析性    │        │ 观点性  │
   └────┬───┘        └────┬────┘        └────┬────┘
        │                  │                  │
        │   ┌──────────────┴──────────────┐   │
        │   │                             │   │
        ▼   ▼                             ▼   ▼
   ┌──────────────────────────────────────────┐
   │         情境性 Contextual                │
   │   (需要额外传入 user_context)             │
   └─────────────────┬────────────────────────┘
                     │
                     ▼
           ┌─────────────────┐
           │   检索文档      │
           │  (不同策略)     │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │   组合物上下文   │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │   LLM 生成答案   │
           └────────┬────────┘
                    │
                    ▼
                 返回答案
```

### 四种策略对比

```
┌─────────────────────────────────────────────────────────────┐
│                      四种检索策略对比                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 📍 事实性策略 Factual                                       │
├─────────────────────────────────────────────────────────────┤
│ 目标：精确找到准确答案                                      │
│ 方法：查询增强 → 多候选 → LLM 评分                          │
│ 适用："是什么"、"多少"、"何时"类问题                       │
│ 示例："地球周长多少？"                                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 📊 分析性策略 Analytical                                    │
├─────────────────────────────────────────────────────────────┤
│ 目标：全面覆盖主题各方面                                    │
│ 方法：生成子查询 → 分别检索 → 多样化选择                    │
│ 适用："如何"、"为什么"、"影响"类问题                       │
│ 示例："气候变化如何影响农业？"                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 💭 观点性策略 Opinion                                       │
├─────────────────────────────────────────────────────────────┤
│ 目标：呈现多种不同观点                                      │
│ 方法：识别观点 → 分别检索 → 选择代表观点                    │
│ 适用：争议话题、价值判断类问题                              │
│ 示例："AI 对人类有益吗？"                                   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 🎯 情境性策略 Contextual                                    │
├─────────────────────────────────────────────────────────────┤
│ 目标：提供个性化答案                                        │
│ 方法：融入上下文 → 情境化查询 → 个性化评分                  │
│ 适用：依赖个人情况的问题                                    │
│ 示例："我该买什么保险？"（需要用户背景）                    │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚠️ 避坑指南

### 常见错误及解决方法

**错误 1: 分类不准确**
```
问题：分类器把事实性问题分到分析性
解决：
1. 优化提示词，提供更清晰的类别定义
2. 添加 Few-shot 示例
3. 使用更好的模型
```

**错误 2: LLM 调用过多，费用高**
```
问题：每个文档都要 LLM 评分
解决：
1. 减少候选数量
2. 批量处理（一次评多个）
3. 对简单问题用轻量方法
```

**错误 3: 情境性检索没有上下文**
```
问题：user_context 为空
解决：
1. 设计 UI 引导用户提供
2. 从用户画像/历史中获取
3. 用默认值或追问
```

**错误 4: 某些策略效果差**
```
问题：特定策略不如预期
解决：
1. 检查该策略的 Prompt
2. 调整参数（如子查询数量）
3. 收集反馈持续优化
```

---

## ❓ 新手常见问题

### Q1: 自适应 RAG 比单一策略好吗？

**答**：取决于场景：

| 场景 | 自适应 | 单一策略 |
|------|--------|---------|
| 问题类型多样 | ✅ 更好 | ❌ 不够灵活 |
| 问题类型单一 | ➖ 差不多 | ✅ 更简单 |
| 资源有限 | ❌ 成本高 | ✅ 成本低 |
| 追求最佳体验 | ✅ 推荐 | ➖ 可能不够 |

### Q2: 可以添加更多策略吗？

**答**：当然可以！

```python
# 例如添加"创造性"策略
class CreativeRetrievalStrategy(BaseRetrievalStrategy):
    """创造性检索：用于头脑风暴、创意生成"""
    def retrieve(self, query, k=4):
        # 实现创意检索逻辑
        pass

# 添加到策略字典
self.strategies["Creative"] = CreativeRetrievalStrategy(texts)
```

### Q3: 如何评估自适应 RAG 的效果？

**答**：多维度评估：

```
1. 准确性：答案是否正确
2. 全面性：是否覆盖关键方面
3. 适当性：策略选择是否合适
4. 用户满意度：主观评价
5. 效率：响应时间
6. 成本：API 调用费用
```

---

## 📝 实战练习

### 练习 1: 添加新的查询类型

```python
# 定义新的分类
class categories_options(BaseModel):
    category: str = Field(
        description="Factual, Analytical, Opinion, Contextual, or Creative"
    )

# 实现新策略
class CreativeRetrievalStrategy(BaseRetrievalStrategy):
    def retrieve(self, query, k=4):
        print("🎨 执行创造性检索...")
        # 实现创造性检索逻辑
        # 例如：联想相关概念、跨领域检索
        pass

# 添加到系统
self.strategies["Creative"] = CreativeRetrievalStrategy(texts)
```

### 练习 2: 优化现有策略

```python
# 例如：为事实性策略添加缓存
from functools import lru_cache

class FactualRetrievalStrategy(BaseRetrievalStrategy):
    @lru_cache(maxsize=100)
    def retrieve(self, query, k=4):
        # 相同查询直接返回缓存结果
        pass
```

---

## 📚 总结

恭喜你完成了自适应检索的学习！现在你已经：

✅ **理解了**自适应 RAG 的核心思想和价值
✅ **掌握了**四种检索策略的实现方法
✅ **学会了**查询分类和策略选择
✅ **能够**在自己的项目中应用此技术

**下一步学习建议**：
1. 尝试用自己的数据测试
2. 添加更多查询类型和策略
3. 结合其他技术（如反馈循环）
4. 优化性能和成本

---

> **💪 记住**：没有最好的策略，只有最合适的策略！
>
> 如果本教程对你有帮助，欢迎分享给更多朋友！🌟
