# 🌟 新手入门：可解释检索（Explainable Retrieval）

> **💡 给新手的说明**
> - **难度等级**：⭐⭐☆☆☆（中等偏易）
> - **预计学习时间**：30-45 分钟
> - **前置知识**：了解基本的 Python 编程，对 RAG（检索增强生成）有初步认识
> - **学完你将掌握**：如何让检索系统不仅返回结果，还能解释"为什么"返回这些结果
>
> **🤔 为什么要学这个？** 想象你去图书馆找书，管理员不仅给你书，还告诉你"这本书适合你是因为..."。可解释检索就是这样的智能助手！

---

## 📖 核心概念理解

### 什么是可解释检索？

传统的检索系统就像一个"黑盒子"：你输入问题，它返回一堆文档，但你不知道**为什么**这些文档被选中。

**可解释检索器**则不同，它会：
1. 检索相关文档
2. 对每个文档说清楚："我选你是因为这个原因..."

### 通俗理解：找图书管理员帮忙

想象你在图书馆问管理员：
> **你**："我想了解为什么天空是蓝色的"

**传统检索系统**的做法：
> 给你 5 本书，什么也不说

**可解释检索系统**的做法：
> 给你 5 本书，并且说：
> - "第 1 本书：这本直接解释了光的散射原理，与你的问题高度相关"
> - "第 2 本书：这本讲述了大气层的组成，帮助你理解背景知识"
> - ...

### 核心组件一览

| 组件 | 作用 | 生活化比喻 |
|------|------|-----------|
| 向量存储 (Vector Store) | 存储文档的数学表示 | 图书馆的藏书目录 |
| 基础检索器 (Retriever) | 快速找到相似文档 | 图书管理员查找目录 |
| 语言模型 (LLM) | 生成自然语言解释 | 管理员给你解释选书理由 |
| ExplainableRetriever 类 | 整合检索和解释功能 | 整个智能咨询服务 |

---

## 🛠️ 第一步：环境准备

### 📖 这是什么？

在开始之前，我们需要安装必要的工具包。这就像做饭前要准备好锅碗瓢盆一样。

### 💻 完整代码

```python
# 安装所需的包
# python-dotenv 用于管理 API 密钥等环境变量
!pip install python-dotenv

# 克隆仓库以访问辅助函数和评估模块
# 这就像复制一本参考书到你的手边
!git clone https://github.com/NirDiamant/RAG_TECHNIQUES.git

# 将仓库路径添加到系统路径，让 Python 能找到这些模块
import sys
sys.path.append('RAG_TECHNIQUES')

# 如果需要使用最新数据运行，可以取消下面这行的注释
# !cp -r RAG_TECHNIQUES/data .
```

> **💡 代码解释**
> - `!pip install`：安装 Python 包，就像用手机应用商店下载 APP
> - `!git clone`：从 GitHub 复制整个项目代码
> - `sys.path.append()`：告诉 Python 去哪里找我们需要的工具
>
> **⚠️ 新手注意**
> - 如果你在 Google Colab 中运行，`!`开头的命令可以直接执行
> - 如果在本地 Jupyter Notebook 运行，确保已安装 git
> - 如果网络慢，git clone 可能需要耐心等待

### 设置环境变量

```python
import os
import sys
from dotenv import load_dotenv

# 从 .env 文件加载环境变量
# 这就像从一个安全的地方取出你的 API 钥匙
load_dotenv()

# 设置 OpenAI API 密钥环境变量
# 这是使用 OpenAI 服务的"通行证"
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
```

> **💡 代码解释**
> - `load_dotenv()`：从 `.env` 文件读取配置信息
> - `os.getenv()`：获取环境变量的值
> - `os.environ[]`：设置环境变量，让其他库能使用
>
> **⚠️ 新手注意**
> - 你需要先创建 `.env` 文件并填入你的 OpenAI API 密钥
> - 格式：`OPENAI_API_KEY=your-api-key-here`
> - **千万不要**把 API 密钥直接写在代码里！
>
> **❓ 常见问题**
>
> **Q: 我没有 OpenAI API 密钥怎么办？**
>
> A: 你可以：
> 1. 去 [OpenAI 官网](https://platform.openai.com/) 申请
> 2. 或者使用其他嵌入模型（如 Hugging Face 的免费模型）
> 3. 本教程主要演示原理，理解思路最重要

---

## 🛠️ 第二步：定义可解释检索器类

### 📖 这是什么？

类（Class）是 Python 中创建对象的模板。我们可以把 `ExplainableRetriever` 想象成一个"智能检索机器"的设计图纸。

### 💻 完整代码

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

class ExplainableRetriever:
    def __init__(self, texts):
        """
        初始化可解释检索器

        参数:
            texts: 文本列表，比如 ["文章 1 内容", "文章 2 内容", ...]
        """
        # 初始化嵌入模型
        # 嵌入模型能把文字变成数字向量，方便计算机比较相似度
        self.embeddings = OpenAIEmbeddings()

        # 从文本创建向量存储
        # FAISS 是 Facebook 开发的高效相似度搜索工具
        self.vectorstore = FAISS.from_texts(texts, self.embeddings)

        # 初始化语言模型（LLM）
        # temperature=0 让输出更稳定、确定
        # max_tokens=4000 限制输出长度
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-4o-mini",
            max_tokens=4000
        )

        # 创建基础检索器
        # search_kwargs={"k": 5} 表示每次检索返回 5 个最相关的文档
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )

        # 创建解释生成的提示模板
        # 这就像给 AI 一个固定的"回答格式"
        explain_prompt = PromptTemplate(
            input_variables=["query", "context"],  # 输入变量名
            template="""
分析以下查询和检索到的上下文之间的关系。
解释为什么此次上下文与查询相关，以及它如何帮助回答查询。

查询：{query}

上下文：{context}

解释：
"""
        )

        # 将提示模板和 LLM 连接成一条"链"
        # 数据会按顺序流经这两个组件
        self.explain_chain = explain_prompt | self.llm

    def retrieve_and_explain(self, query):
        """
        执行检索并生成解释

        参数:
            query: 用户的查询问题

        返回:
            包含文档内容和解释的字典列表
        """
        # 使用基础检索器检索相关文档
        # get_relevant_documents 返回 Document 对象列表
        docs = self.retriever.get_relevant_documents(query)

        # 存储解释后的结果
        explained_results = []

        # 遍历每个检索到的文档
        for doc in docs:
            # 准备输入数据
            input_data = {
                "query": query,           # 用户的问题
                "context": doc.page_content  # 文档内容
            }

            # 调用 LLM 生成解释
            # invoke() 方法执行链式调用
            explanation = self.explain_chain.invoke(input_data).content

            # 将结果添加到列表中
            explained_results.append({
                "content": doc.page_content,    # 文档原始内容
                "explanation": explanation      # AI 生成的解释
            })

        return explained_results
```

> **💡 代码解释**
>
> **关键概念解析：**
>
> 1. **嵌入（Embedding）**
>    - 把文字转换成数字向量的过程
>    - 相似的内容在向量空间中距离更近
>    - 比如："猫"和"狗"的向量距离比"猫"和"汽车"更近
>
> 2. **FAISS**
>    - Facebook AI Similarity Search 的缩写
>    - 可以快速从百万级文档中找到最相似的
>    - 比逐个比较快得多
>
> 3. **PromptTemplate（提示模板）**
>    - 预定义的提示词格式
>    - `{query}` 和 `{context}` 会被实际内容替换
>    - 就像填空题的模板
>
> 4. **Chain（链）**
>    - 把多个组件串联起来
>    - 数据像流水线一样依次处理
>
> **⚠️ 新手注意**
> - `__init__` 是构造函数，创建对象时自动调用
> - `self` 代表对象本身，用于访问对象的属性和方法
> - `|` 符号是 LangChain 的链式操作符，相当于"然后"
>
> **❓ 常见问题**
>
> **Q: 为什么要设置 temperature=0？**
>
> A: temperature 控制输出的随机性：
> - 值越高（如 1.0），输出越有创意但不稳定
> - 值越低（如 0），输出越确定和一致
> - 解释任务需要稳定性，所以设为 0

---

## 🛠️ 第三步：创建实例并测试

### 📖 这是什么？

现在我们要使用刚才定义的"设计图纸"创建一个实际的检索器对象，并用一些测试数据来验证它是否工作正常。

### 💻 完整代码

```python
# 创建示例文本数据
# 这里用 3 个简单的句子做演示
texts = [
    "The sky is blue because of the way sunlight interacts with the atmosphere.",
    "Photosynthesis is the process by which plants use sunlight to produce energy.",
    "Global warming is caused by the increase of greenhouse gases in Earth's atmosphere."
]

# 创建可解释检索器实例
# 这就像根据设计图纸造出一台实际工作的机器
explainable_retriever = ExplainableRetriever(texts)

# 测试查询
query = "Why is the sky blue?"

# 执行检索并获取带解释的结果
results = explainable_retriever.retrieve_and_explain(query)

# 打印结果
for i, result in enumerate(results, 1):
    print(f"结果{i}:")
    print(f"内容：{result['content']}")
    print(f"解释：{result['explanation']}")
    print()  # 空行分隔
```

> **💡 代码解释**
> - `enumerate(results, 1)`：遍历列表同时获取索引，从 1 开始计数
> - `result['content']`：访问字典中的内容字段
> - `result['explanation']`：访问字典中的解释字段
>
> **⚠️ 新手注意**
> - 示例文本很简单，实际应用中会是成百上千的文档
> - 每个查询会返回 5 个结果（由 k=5 决定）
> - 如果文档总数少于 5 个，则返回所有文档
>
> **📊 预期输出示例**
>
> ```
> 结果 1:
> 内容：The sky is blue because of the way sunlight interacts with the atmosphere.
> 解释：这个文档直接回答了天空为什么是蓝色的问题。它提到阳光与大气层的相互作用是导致天空呈现蓝色的原因。这是对您查询最直接相关的信息。
>
> 结果 2:
> 内容：Photosynthesis is the process by which plants use sunlight to produce energy.
> 解释：这个文档提到了阳光，但主要讲述的是植物的光合作用过程。它与您的查询相关性较低，因为不涉及天空颜色的成因。
>
> 结果 3:
> 内容：Global warming is caused by the increase of greenhouse gases in Earth's atmosphere.
> 解释：这个文档涉及大气层，但讨论的是全球变暖问题。它与天空颜色的问题关联不大，相关性较低。
> ```
>
> **❓ 常见问题**
>
> **Q: 为什么返回的结果中有不相关的内容？**
>
> A: 这是因为检索器基于向量相似度，不是精确匹配。有时候语义相关但主题不同的文档也会被检索到。解释功能正好可以帮助你判断每个结果的实际相关性。

---

## 🎯 完整代码整合

### 一站式完整代码

```python
# ============== 环境准备 ==============
!pip install python-dotenv
!git clone https://github.com/NirDiamant/RAG_TECHNIQUES.git

import sys
sys.path.append('RAG_TECHNIQUES')

import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# ============== 导入必要的库 ==============
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# ============== 定义可解释检索器类 ==============
class ExplainableRetriever:
    def __init__(self, texts):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.from_texts(texts, self.embeddings)
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-4o-mini",
            max_tokens=4000
        )
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )

        explain_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""
分析以下查询和检索到的上下文之间的关系。
解释为什么此次上下文与查询相关，以及它如何帮助回答查询。

查询：{query}

上下文：{context}

解释：
"""
        )
        self.explain_chain = explain_prompt | self.llm

    def retrieve_and_explain(self, query):
        docs = self.retriever.get_relevant_documents(query)
        explained_results = []

        for doc in docs:
            input_data = {"query": query, "context": doc.page_content}
            explanation = self.explain_chain.invoke(input_data).content
            explained_results.append({
                "content": doc.page_content,
                "explanation": explanation
            })

        return explained_results

# ============== 创建示例并测试 ==============
texts = [
    "The sky is blue because of the way sunlight interacts with the atmosphere.",
    "Photosynthesis is the process by which plants use sunlight to produce energy.",
    "Global warming is caused by the increase of greenhouse gases in Earth's atmosphere."
]

explainable_retriever = ExplainableRetriever(texts)

query = "Why is the sky blue?"
results = explainable_retriever.retrieve_and_explain(query)

for i, result in enumerate(results, 1):
    print(f"结果{i}:")
    print(f"内容：{result['content']}")
    print(f"解释：{result['explanation']}")
    print()
```

---

## 📚 进阶知识

### 这种方法的优势

| 优势 | 说明 | 实际应用场景 |
|------|------|-------------|
| **透明度** | 用户可以理解为什么检索到特定文档 | 法律文档检索，需要知道相关性依据 |
| **信任** | 解释建立用户对系统结果的信心 | 医疗信息系统，决策需要可追溯 |
| **学习** | 用户可以深入了解查询和文档之间的关系 | 教育工具，帮助学生理解信息关联 |
| **调试** | 更容易识别和纠正检索过程中的问题 | 系统开发和维护阶段 |
| **定制化** | 解释提示可以根据不同用例调整 | 针对不同行业定制解释风格 |

### 自定义解释提示

你可以根据需求修改解释的格式和风格：

```python
# 针对法律领域的解释提示
legal_explain_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="""
作为法律助手，分析以下查询和检索到的法律条文之间的关系。
解释：
1. 这个条文与查询的法律问题有何关联
2. 引用的具体条款或原则
3. 可能适用的法律场景

查询：{query}

检索到的法律条文：{context}

法律分析：
"""
)
```

---

## 🎓 术语解释表

| 术语 | 英文 | 解释 |
|------|------|------|
| 嵌入 | Embedding | 将文本转换为数字向量的技术 |
| 向量存储 | Vector Store | 存储和检索向量数据的数据库 |
| 检索器 | Retriever | 负责查找相关文档的组件 |
| 语言模型 | Language Model (LLM) | 能理解和生成自然语言的 AI 模型 |
| 提示模板 | Prompt Template | 预定义的提示词格式 |
| 链 | Chain | 将多个组件串联的处理流程 |
| 相似度搜索 | Similarity Search | 根据向量距离查找相似内容 |
| FAISS | FAISS | Facebook 开发的高效相似度搜索库 |

---

## ✅ 学习检查清单

- [ ] 我理解了什么是可解释检索及其价值
- [ ] 我知道嵌入（Embedding）的作用
- [ ] 我能解释 FAISS 向量存储的功能
- [ ] 我理解了 PromptTemplate 的用途
- [ ] 我能创建自己的 ExplainableRetriever 实例
- [ ] 我知道如何自定义解释提示

---

## 🚀 下一步学习建议

1. **尝试更多文档**：用真实的文档集合（如 PDF、网页）替换示例文本
2. **调整解释风格**：修改提示模板，让解释更适合你的场景
3. **优化检索参数**：尝试改变 `k` 值，观察结果数量的影响
4. **学习下一个技术**：继续学习 GraphRAG，了解如何用知识图谱增强检索

---

> **💪 恭喜！** 你已经完成了可解释检索的新手教程。现在你不仅知道如何检索文档，还能让系统告诉你"为什么"选择这些文档。这是构建可信赖 RAG 系统的重要一步！
