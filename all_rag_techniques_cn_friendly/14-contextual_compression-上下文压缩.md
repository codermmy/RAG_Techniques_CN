# 🌟 新手入门：文档检索中的上下文压缩

> **💡 给新手的说明**
> - **难度级别**：⭐⭐⭐ 中级（需要了解基础的 RAG 和向量检索概念）
> - **预计学习时间**：40-50 分钟
> - **前置知识**：了解向量存储、检索器的基本概念
> - **本教程你将学会**：如何让检索系统只返回"精华部分"，而不是整段整段的原文

---

## 📖 核心概念理解

### 什么是上下文压缩？

想象你去图书馆找资料，图书管理员给你抱来一整本书说："答案在这本书里，自己找吧。"

**传统检索**就像这个管理员——返回整个文档或大块文本，让你自己去筛选。

**上下文压缩**则像一个贴心的助手，它会把书中与你的问题最相关的几段话摘抄出来，直接给你精华内容。

### 通俗理解

```
传统检索：
问题："法国首都哪里有好吃的？"
返回：[整本巴黎旅游指南，500 页]
→ 你：😫 我要自己找多久...

上下文压缩检索：
问题："法国首都哪里有好吃的？"
返回：["巴黎推荐餐厅：1. Le Comptoir... 2. L'Ami Jean..."]
→ 你：😊 太棒了！
```

### 为什么需要上下文压缩？

| 问题 | 传统检索 | 上下文压缩 |
|------|---------|-----------|
| 返回内容太多 | ❌ 整段整段的文本 | ✅ 只提取相关句子 |
| 包含无关信息 | ❌ 噪声干扰 LLM 判断 | ✅ 过滤掉不相关内容 |
| 处理效率 | ❌ LLM 需要处理大量文本 | ✅ 减少输入 token 数量 |
| 答案精准度 | ❌ 可能被无关信息干扰 | ✅ 聚焦核心信息 |

### 工作原理图解

```
用户提问 → 检索器找到相关文档 → 压缩器提取精华 → 返回简洁答案
    ↓              ↓                    ↓              ↓
 "气候变化的    [找到 3 篇相关文章，    [LLM 分析每篇，   "气候变化主要由
  原因是？"      共 5000 字]            提取关键句]        人类燃烧化石燃料引起..."
```

---

## 🛠️ 第一步：安装必要的包

### 📖 这是什么？

安装运行本教程所需的 Python 库。

### 💻 完整代码

```python
# 安装所需的包
# langchain: RAG 系统的核心框架
# python-dotenv: 安全管理 API 密钥
!pip install langchain python-dotenv
```

> **💡 代码解释**
> - `langchain`：提供构建 RAG 系统的全套工具
> - `python-dotenv`：从 `.env` 文件读取环境变量，避免硬编码 API 密钥

> **⚠️ 新手注意**
> - 国内用户可以使用清华镜像源加速安装：
>   ```bash
>   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple langchain python-dotenv
>   ```

### 导入必要的库

```python
import os
import sys
from dotenv import load_dotenv

# LangChain 的核心组件
# LLMChainExtractor: 基于 LLM 的文本提取器，用于压缩上下文
from langchain.retrievers.document_compressors import LLMChainExtractor
# ContextualCompressionRetriever: 带压缩功能的检索器
from langchain.retrievers import ContextualCompressionRetriever
# RetrievalQA: 问答链，整合检索和回答生成
from langchain.chains import RetrievalQA

# 导入项目提供的辅助函数
from helper_functions import *
from evaluation.evalute_rag import *

# 加载 .env 文件中的环境变量
load_dotenv()

# 设置 OpenAI API 密钥
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
```

> **💡 代码解释**
> - `LLMChainExtractor`：使用大语言模型理解查询意图，从文档中提取相关片段
> - `ContextualCompressionRetriever`：包装普通检索器，添加压缩功能
> - `RetrievalQA`：将检索和问答功能整合成一个简单的接口

> **⚠️ 新手注意**
> - 确保 `.env` 文件中有正确的 `OPENAI_API_KEY`
> - 如果导入失败，检查 langchain 版本是否兼容

---

## 📂 第二步：准备和编码 PDF 文档

### 📖 这是什么？

加载 PDF 文档并创建向量存储，为后续的检索做准备。

### 💻 完整代码

```python
# 创建 data 目录
import os
os.makedirs('data', exist_ok=True)

# 下载教程使用的 PDF 文档
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
```

> **⚠️ 新手注意**
> - 如果下载失败，可以手动下载 PDF 放到 `data/` 目录
> - 也可以用自己的 PDF 替换

### 定义文件路径

```python
# PDF 文件路径
path = "data/Understanding_Climate_Change.pdf"
```

### 创建向量存储

```python
# encode_pdf 是辅助函数，完成以下工作：
# 1. 读取 PDF
# 2. 分割成文本块
# 3. 计算每个块的 Embedding
# 4. 创建 FAISS 向量存储
vector_store = encode_pdf(path)

print("向量存储创建完成！")
```

> **💡 代码解释**
> - `encode_pdf()` 封装了 PDF 处理的全流程
> - 返回的 `vector_store` 是一个 FAISS 向量数据库，可以进行语义搜索

> **❓ 常见问题**
> - **Q**: `encode_pdf` 在哪里定义的？
> - **A**: 这是项目 `helper_functions.py` 中提供的辅助函数，内部使用 `RecursiveCharacterTextSplitter` 分块和 `OpenAIEmbeddings` 计算向量

---

## 🔧 第三步：创建检索器和压缩器

### 📖 这是什么？

这是本教程的核心步骤！我们要创建两个组件：
1. **基础检索器**：负责找到相关文档
2. **上下文压缩器**：负责从文档中提取精华

### 💻 完整代码

```python
# ========== 1. 创建基础检索器 ==========
# as_retriever() 把向量存储转成检索器接口
# 默认会返回最相关的 4 个文档块
retriever = vector_store.as_retriever()

# ========== 2. 创建 LLM（大语言模型）实例 ==========
# ChatOpenAI 是 LangChain 提供的 OpenAI 聊天模型接口
# temperature=0: 让模型输出更稳定、确定（适合提取任务）
# model_name="gpt-4o-mini": 使用 GPT-4o-mini 模型（性价比高）
# max_tokens=4000: 允许的最大输出长度
llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=4000)

# ========== 3. 创建上下文压缩器 ==========
# LLMChainExtractor.from_llm() 创建一个基于 LLM 的文本提取器
# 它会理解查询的上下文，智能地从文档中提取相关部分
compressor = LLMChainExtractor.from_llm(llm)

# ========== 4. 组合检索器和压缩器 ==========
# ContextualCompressionRetriever 把两者结合起来
# 工作流程：先检索 → 再压缩提取 → 返回精华
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,  # 使用上面创建的压缩器
    base_retriever=retriever     # 使用上面创建的检索器
)

# ========== 5. 创建问答链 ==========
# RetrievalQA 整合了检索和问答功能
# 输入问题 → 自动检索 → 用检索结果生成答案
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,                        # 使用哪个 LLM 来回答问题
    retriever=compression_retriever, # 使用带压缩功能的检索器
    return_source_documents=True     # 返回源文档（方便查看）
)
```

> **💡 代码解释**

整个流程的工作方式：

```
用户提问
    ↓
┌─────────────────────────────────────┐
│  compression_retriever              │
│  ┌─────────────────────────────┐    │
│  │ 1. base_retriever 检索       │    │
│  │    → 返回 4 个相关文档块      │    │
│  └─────────────────────────────┘    │
│              ↓                       │
│  ┌─────────────────────────────┐    │
│  │ 2. base_compressor 压缩      │    │
│  │    → LLM 分析并提取精华       │    │
│  └─────────────────────────────┘    │
│              ↓                       │
│  ┌─────────────────────────────┐    │
│  │ 3. 返回压缩后的相关片段      │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
    ↓
LLM 用提取的精华内容生成最终答案
```

> **⚠️ 新手注意**
> - `temperature=0` 很重要！提取任务需要稳定的输出，不要用高 temperature
> - `gpt-4o-mini` 是性价比之选，也可以用 `gpt-4o` 获得更好效果但成本更高
> - `return_source_documents=True` 让你可以看到答案是从哪里来的，方便验证

---

## 🧪 第四步：测试上下文压缩检索

### 📖 这是什么？

实际运行我们的压缩检索系统，看看效果如何。

### 💻 完整代码

```python
# 定义测试问题
query = "What is the main topic of the document?"

# 调用问答链获取答案
# invoke() 方法接收一个字典，包含查询字符串
result = qa_chain.invoke({"query": query})

# 打印结果
print("答案：")
print(result["result"])
print("\n" + "="*50)
print("源文档（压缩后的精华片段）：")
print("源文档数量:", len(result["source_documents"]))
for i, doc in enumerate(result["source_documents"], 1):
    print(f"\n--- 片段 {i} ---")
    print(doc.page_content)
```

> **💡 代码解释**
> - `invoke({"query": query})`：运行整个 QA 链
> - `result["result"]`：LLM 生成的最终答案
> - `result["source_documents"]`：压缩后用于生成答案的文档片段列表

### 预期输出示例

```
答案：
The main topic of the document is climate change, including its causes,
effects, and potential solutions. The document discusses how human
activities, particularly the burning of fossil fuels and deforestation,
are driving global warming and its wide-ranging impacts on ecosystems,
weather patterns, and human societies.

==================================================
源文档（压缩后的精华片段）：
源文档数量：2

--- 片段 1 ---
Climate change refers to significant and lasting changes in global
temperatures and weather patterns. While climate change can occur
naturally over geological time scales, human activities since the
Industrial Revolution have been the dominant cause...

--- 片段 2 ---
The primary driver of climate change is the burning of fossil fuels
(coal, oil, and natural gas) for energy and transportation, which
releases large amounts of carbon dioxide and other greenhouse gases...
```

> **💡 观察重点**
>
> 注意看源文档片段——它们不是完整的段落，而是**被提取出来的精华句子**！
>
> 对比原始文档，压缩器智能地：
> - ✅ 保留了与问题直接相关的信息
> - ✅ 去掉了无关的背景和例子
> - ✅ 提取出简洁的核心内容

---

## 🔄 完整流程图解

```
┌─────────────────────────────────────────────────────────────────┐
│                    上下文压缩检索完整流程                         │
└─────────────────────────────────────────────────────────────────┘

1️⃣ 用户提问
   "What is the main topic?"
        │
        ▼
2️⃣ 基础检索器 (base_retriever)
   从向量存储中找到 4 个最相关的文档块
   ┌─────────────────────────────────────────┐
   │ 文档块 1: [完整段落，500 字]               │
   │ 文档块 2: [完整段落，500 字]               │
   │ 文档块 3: [完整段落，500 字]               │
   │ 文档块 4: [完整段落，500 字]               │
   └─────────────────────────────────────────┘
        │
        ▼
3️⃣ 上下文压缩器 (LLMChainExtractor)
   LLM 分析每个文档块，提取与问题相关的部分
   ┌─────────────────────────────────────────┐
   │ 从文档块 1 提取："Climate change is..."   │
   │ 从文档块 2 提取："The main causes are..." │
   │ 文档块 3 提取：（无相关内容，跳过）         │
   │ 从文档块 4 提取："Global warming leads..."│
   └─────────────────────────────────────────┘
        │
        ▼
4️⃣ 压缩后的精华内容
   总字数从 2000 字 → 300 字
        │
        ▼
5️⃣ LLM 生成最终答案
   基于压缩后的精华内容生成连贯的回答
```

---

## 📊 上下文压缩 vs 传统检索对比

让我们通过一个具体例子来看看差异：

```python
# 创建一个普通检索器（无压缩）
normal_retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 创建普通 QA 链
normal_qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=normal_retriever,
    return_source_documents=True
)

# 对比两种方法的结果
query = "What causes climate change?"

print("=" * 60)
print("【传统检索】返回的源文档：")
normal_result = normal_qa_chain.invoke({"query": query})
for i, doc in enumerate(normal_result["source_documents"], 1):
    print(f"\n文档 {i} ({len(doc.page_content)} 字):")
    print(doc.page_content[:300] + "...")

print("\n" + "=" * 60)
print("【上下文压缩】返回的源文档：")
compressed_result = qa_chain.invoke({"query": query})
for i, doc in enumerate(compressed_result["source_documents"], 1):
    print(f"\n文档 {i} ({len(doc.page_content)} 字):")
    print(doc.page_content[:300] + "...")

print("\n" + "=" * 60)
print("字数对比：")
normal_total = sum(len(doc.page_content) for doc in normal_result["source_documents"])
compressed_total = sum(len(doc.page_content) for doc in compressed_result["source_documents"])
print(f"传统检索：{normal_total} 字")
print(f"上下文压缩：{compressed_total} 字")
print(f"压缩率：{compressed_total/normal_total*100:.1f}%")
```

### 预期对比结果

```
【传统检索】返回的源文档：
文档 1 (487 字):
Climate change is a complex issue driven by multiple factors. The Earth's
climate system has experienced changes throughout its history, including
ice ages and warmer periods. However, scientific evidence shows that human
activities since the Industrial Revolution have been the dominant cause...

文档 2 (512 字):
The greenhouse effect is a natural process that warms the Earth's surface.
It occurs when certain gases in the atmosphere trap heat from the sun.
Without the greenhouse effect, Earth would be too cold to support most
forms of life. However, human activities are intensifying this effect...

【上下文压缩】返回的源文档：
文档 1 (89 字):
Human activities since the Industrial Revolution, particularly burning
fossil fuels and deforestation, have been the dominant cause of climate
change.

文档 2 (76 字):
Burning fossil fuels releases greenhouse gases, mainly carbon dioxide,
which trap heat and intensify the greenhouse effect.

字数对比：
传统检索：999 字
上下文压缩：165 字
压缩率：16.5%
```

> **💡 关键观察**
> - 上下文压缩将内容压缩到原来的约 **1/6**
> - 但核心信息完全保留
> - LLM 处理更少的 token = 更低的成本 + 更快的速度

---

## ⚙️ 进阶配置选项

### 调整检索数量

```python
# 增加初始检索的文档数量
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

# 压缩器会从这 10 个文档中提取精华
# 适合需要更全面信息的场景
```

### 使用不同的 LLM 模型

```python
# 使用更强的模型（成本更高）
llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)

# 使用更经济的模型（成本更低）
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=1000)
```

### 自定义压缩器参数

```python
# LLMChainExtractor 支持自定义 prompt
from langchain.prompts import PromptTemplate

custom_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="基于查询：{query}\n\n从以下上下文中提取最相关的信息：{context}\n\n提取内容："
)

compressor = LLMChainExtractor.from_llm(llm, prompt=custom_prompt)
```

---

## ⚠️ 常见问题及解决方法

### 问题 1：压缩后内容为空

```
警告：压缩后没有返回任何内容
```

**可能原因**：
- 检索到的文档与查询不相关
- LLM 没有正确理解提取任务

**解决方法**：
```python
# 1. 增加初始检索数量
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

# 2. 尝试更强的 LLM 模型
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
```

### 问题 2：压缩效果不明显

```
压缩后的内容和原始文档长度差不多
```

**可能原因**：
- 原始文档已经很精炼
- LLM 认为大部分内容都相关

**解决方法**：
```python
# 尝试调整 prompt，让压缩更激进
custom_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="基于查询：{query}\n\n从以下上下文中只提取直接相关的核心信息，尽量简洁：{context}\n\n提取内容："
)
```

### 问题 3：API 调用成本太高

**降低成本的策略**：
```python
# 1. 减少检索数量
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 2. 使用更小的模型
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=500)

# 3. 限制压缩器输出的最大长度
compressor = LLMChainExtractor.from_llm(llm)
# 在应用层限制处理文档的总长度
```

---

## 🧪 完整代码整合

```python
# ========== 1. 安装和导入 ==========
!pip install langchain python-dotenv

import os
from dotenv import load_dotenv
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from helper_functions import *

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# ========== 2. 加载文档并创建向量存储 ==========
os.makedirs('data', exist_ok=True)
# 下载或手动放置 PDF
path = "data/Understanding_Climate_Change.pdf"
vector_store = encode_pdf(path)

# ========== 3. 创建检索器和压缩器 ==========
retriever = vector_store.as_retriever()

llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=4000)
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=compression_retriever,
    return_source_documents=True
)

# ========== 4. 测试 ==========
query = "What is the main topic of the document?"
result = qa_chain.invoke({"query": query})

print("答案：")
print(result["result"])
print("\n源文档片段：")
for doc in result["source_documents"]:
    print(f"- {doc.page_content[:100]}...")
```

---

## 🎓 学习总结

### 你学到了什么？

✅ **上下文压缩的概念**：从检索到的文档中提取精华，而不是返回整段文本
✅ **LLMChainExtractor 的使用**：基于 LLM 的智能文本提取器
✅ **ContextualCompressionRetriever**：将检索和压缩功能整合的检索器
✅ **成本优化技巧**：通过减少处理 token 数量来降低 API 成本

### 实际应用场景

| 场景 | 传统检索的问题 | 上下文压缩的优势 |
|------|---------------|-----------------|
| 长文档问答 | 返回大段无关内容 | 只提取相关句子 |
| 多文档汇总 | 信息冗余重复 | 去重并提取核心 |
| 实时问答系统 | 响应慢、token 消耗大 | 快速、经济 |
| 移动端应用 | 传输数据量大 | 精简内容易于展示 |

### 下一步可以做什么？

1. 📊 对比不同 LLM 模型的压缩效果
2. 🔧 尝试自定义 prompt 来调整压缩风格
3. 🔄 结合其他 RAG 技术（如重排序）进一步提升质量
4. 💰 测试和估算使用上下文压缩后的成本节省

---

## 📚 相关资源

- [LangChain 上下文压缩文档](https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression)
- [LLMChainExtractor API 参考](https://api.python.langchain.com/en/latest/retrievers/langchain.retrievers.document_compressors.LLMChainExtractor.html)
- [RAG 技术最佳实践](https://python.langchain.com/docs/use_cases/question_answering/)

---

*本教程是 RAG 技术系列教程之一。上下文压缩可以与重排序、融合检索等技术结合使用，构建更高效的 RAG 系统。*

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--contextual-compression)
