# 🌟 新手入门：文档处理的语义分块

> **💡 给新手的说明**
> - **难度级别**：⭐⭐⭐ 中级（需要了解基础的 Python 和 RAG 概念）
> - **预计学习时间**：45-60 分钟
> - **前置知识**：了解什么是 Embedding、向量数据库、PDF 处理
> - **本教程你将学会**：如何让 AI 更"聪明"地分割长文档，而不是简单地按字数切割

---

## 📖 核心概念理解

### 什么是语义分块？

想象一下你在读一本厚厚的小说。如果用传统方法分块，就像机械地每 100 页切一刀——结果可能把一章完整的故事拦腰截断，让你看得莫名其妙。

**语义分块**则像一个有经验的图书管理员，它会在"章节结束"或"话题转换"的自然位置分开，确保每一块内容都是完整连贯的。

### 通俗理解

| 传统分块 | 语义分块 |
|---------|---------|
| 🤖 机械地每 500 字切一刀 | 🧠 在语义变化处自然分开 |
| 可能打断完整的句子或概念 | 保持每个块内的话题连贯性 |
| "这个块说到一半就断了..." | "这个块讲的是一个完整的意思" |

### 为什么要用语义分块？

```
传统分块的问题：
"气候变化是由温室气体排放引起的。这些气体包括二氧化碳、甲烷..."
                            ↓ [机械切割在这里]
"甲烷和氮氧化物。另一个重要因素是森林砍伐..."
→ 读者：？？？这句话怎么没头没尾的？

语义分块的效果：
"气候变化是由温室气体排放引起的。这些气体包括二氧化碳、甲烷和氮氧化物。"
→ 完整的意思，容易理解！
```

---

## 🛠️ 第一步：安装必要的包

### 📖 这是什么？

在开始之前，我们需要安装运行这个教程所需的所有 Python 库。就像做饭前要准备好锅碗瓢盆一样。

### 💻 完整代码

```python
# 安装所需的包
# langchain-experimental: 包含语义分块器的实验性功能
# langchain-openai: OpenAI 的 Embedding 接口
# python-dotenv: 用于管理 API 密钥等环境变量
!pip install langchain-experimental langchain-openai python-dotenv
```

> **💡 代码解释**
> - `!pip install` 是 Jupyter Notebook 中安装 Python 包的命令
> - 这些包都是 RAG 系统的基础依赖

> **⚠️ 新手注意**
> - 如果你在国内，pip 安装可能会比较慢，可以配置国内镜像源：
>   ```bash
>   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple langchain-experimental langchain-openai python-dotenv
>   ```
> - 安装完成后建议重启 Jupyter 内核再导入

### 导入必要的库

```python
# 导入 Python 内置库
import os      # 操作系统接口，用于读取环境变量
import sys     # Python 解释器相关功能

# 从 .env 文件加载环境变量（安全地管理 API 密钥）
from dotenv import load_dotenv

# 导入辅助函数（项目提供的工具函数）
from helper_functions import *
from evaluation.evalute_rag import *

# 导入 LangChain 的语义分块器 - 这是本教程的核心工具
from langchain_experimental.text_splitter import SemanticChunker
# 导入 OpenAI 的 Embedding 模型
from langchain_openai.embeddings import OpenAIEmbeddings

# 从 .env 文件加载环境变量
load_dotenv()

# 设置 OpenAI API 密钥环境变量
# 这样代码才能调用 OpenAI 的服务
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
```

> **💡 代码解释**
> - `.env` 文件是一个存放敏感信息（如 API 密钥）的地方，避免把密钥直接写在代码里
> - `SemanticChunker` 是 LangChain 提供的语义分块工具，会自动判断在哪里分割文本最合适

> **⚠️ 新手注意**
> - 你需要一个 OpenAI API 密钥才能运行这段代码
> - 确保在项目根目录创建 `.env` 文件，内容格式：`OPENAI_API_KEY=sk-xxxxx`
> - 如果看到 `ModuleNotFoundError`，说明有包没安装成功，回到第一步重新安装

---

## 📂 第二步：准备和加载 PDF 文档

### 📖 这是什么？

我们需要一个 PDF 文档来演示语义分块的效果。本教程使用的是《理解气候变化》这份科普文档。

### 💻 完整代码

```python
# 创建 data 目录（如果不存在）
import os
os.makedirs('data', exist_ok=True)

# 下载本教程使用的 PDF 文档
# 从 GitHub 仓库下载气候变化相关的 PDF
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
```

> **💡 代码解释**
> - `os.makedirs('data', exist_ok=True)`：创建名为 data 的文件夹，`exist_ok=True`表示如果文件夹已存在也不会报错
> - `!wget`：Linux/Mac 系统下载文件的命令，Windows 用户可以用 `!curl -o` 替代

> **⚠️ 新手注意**
> - 如果下载失败（国内网络问题），可以手动下载 PDF 放到 `data/` 目录下
> - 你也可以用自己的 PDF 文档替换，只需修改文件路径

### 定义文件路径

```python
# 定义 PDF 文件的路径变量
path = "data/Understanding_Climate_Change.pdf"
```

### 将 PDF 读取为字符串

```python
# 使用辅助函数将 PDF 转换为纯文本字符串
content = read_pdf_to_string(path)
```

> **💡 代码解释**
> - `read_pdf_to_string` 是项目提供的辅助函数，内部使用 PyPDFLoader 读取 PDF
> - 返回值 `content` 是一个长字符串，包含 PDF 的全部文本内容

> **❓ 常见问题**
> - **Q**: 如果 PDF 里有表格或图片怎么办？
> - **A**: 这个方法只能提取文字，表格和图片会被忽略。如需处理复杂 PDF，可以考虑使用专门的 OCR 工具

---

## 🎯 第三步：配置语义分块器

### 📖 这是什么？

这一步是配置"切割规则"——告诉分块器在什么地方进行分割。语义分块器会分析文本的语义相似度，在语义变化大的地方进行切割。

### 三种断点类型详解

语义分块器提供三种判断"何时切割"的方法：

| 类型 | 工作原理 | 适用场景 |
|------|---------|---------|
| `percentile` (百分位) | 计算所有句子间的差异，当差异超过 X 百分位时切割 | 最常用，适合大多数文档 |
| `standard_deviation` (标准差) | 当语义差异超过 X 个标准差时切割 | 适合语义分布均匀的文档 |
| `interquartile` (四分位) | 使用四分位距判断切割点 | 适合长度变化大的文档 |

### 通俗理解断点类型

```
假设我们有一串句子，语义分块器会计算相邻句子的"语义差异分数"：

句子 A → 句子 B: 差异 0.1 (很相似)
句子 B → 句子 C: 差异 0.2 (还是比较像)
句子 C → 句子 D: 差异 0.8 (话题变了！) ← 这里可能需要切割
句子 D → 句子 E: 差异 0.1 (又稳定了)

'percentile' 方法的意思是：
"把所有差异排个队，超过 90% 的差异才切割"
```

### 💻 完整代码

```python
# 创建语义分块器
# OpenAIEmbeddings(): 使用 OpenAI 的 Embedding 模型来计算语义相似度
# breakpoint_threshold_type='percentile': 使用百分位法判断切割点
# breakpoint_threshold_amount=90: 超过 90% 分位的差异才切割（比较保守）
text_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type='percentile',
    breakpoint_threshold_amount=90
)
```

> **💡 代码解释**
> - `SemanticChunker` 会用 Embedding 模型把每个句子转成向量，然后计算相邻句子的相似度
> - 阈值设得越高（如 95），分出的块越大；设得越低（如 70），分出的块越小越细

> **⚠️ 新手注意**
> - **推荐从 90 开始尝试**，这是一个比较平衡的值
> - 如果发现分出的块太大，可以降到 80-85
> - 如果块太碎，可以升到 92-95

---

## ✂️ 第四步：执行语义分块

### 📖 这是什么？

现在我们要真正开始"切割"了！分块器会分析整个文档，在语义变化处进行分割，生成一系列有意义的文本块。

### 💻 完整代码

```python
# 创建文档块
# create_documents 接收一个字符串列表，这里我们只传了一个文档
# 返回值 docs 是一个 Document 对象列表，每个对象包含一个语义完整的文本块
docs = text_splitter.create_documents([content])

# 查看分块结果
print(f"文档被分成了 {len(docs)} 个块")
print(f"第一个块的内容（前 200 字）：\n{docs[0].page_content[:200]}...")
```

> **💡 代码解释**
> - `create_documents()` 是核心方法，执行实际的分割操作
> - 返回的 `docs` 是 LangChain 的 `Document` 对象列表，每个对象有：
>   - `page_content`: 文本内容
>   - `metadata`: 元数据（如来源、页码等）

> **⚠️ 新手注意**
> - 分块数量取决于文档长度和复杂度，没有固定标准
> - 一篇 5000 字的文章可能分成 10-30 个块都是正常的

---

## 🗄️ 第五步：创建向量存储

### 📖 这是什么？

向量存储就像一个"智能图书馆"，把我们分好的文本块转换成向量（数字列表）并建立索引，以后可以用语义搜索快速找到相关内容。

### 通俗理解

```
普通搜索 vs 向量搜索：

普通搜索（关键词匹配）：
搜索："天气变热的原因"
匹配："天气"、"热" → 没出现这些词的段落就找不到

向量搜索（语义匹配）：
搜索："天气变热的原因"
→ 能找到"全球变暖的主要因素"这样的内容（意思相同但用词不同）
```

### 💻 完整代码

```python
# 导入 FAISS（Facebook AI 相似性搜索库）
import faiss
from langchain.vectorstores import FAISS

# 创建 Embedding 模型实例
# 这个模型会把文本转换成向量（一串数字）
embeddings = OpenAIEmbeddings()

# 从文档块创建 FAISS 向量存储
# 这一步会自动：
# 1. 把每个文本块转换成向量
# 2. 建立索引以便快速搜索
vectorstore = FAISS.from_documents(docs, embeddings)

# 创建检索器
# search_kwargs={"k": 2} 表示搜索时返回最相关的 2 个结果
chunks_query_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
```

> **💡 代码解释**
> - `FAISS.from_documents()`：自动完成 Embedding 计算和索引创建
> - `as_retriever()`：把向量存储转成"检索器"对象，提供统一的搜索接口
> - `k=2`：每次搜索返回 2 个最相关的块，可根据需要调整

> **⚠️ 新手注意**
> - FAISS 是 Facebook 开源的向量搜索库，适合中等规模数据
> - 如果处理超大规模数据（百万级向量），可以考虑专门的向量数据库如 Pinecone、Weaviate

---

## 🔍 第六步：测试检索效果

### 📖 这是什么？

我们来实际测试一下，看看系统能否找到与问题最相关的文本块。

### 💻 完整代码

```python
# 定义测试问题
test_query = "What is the main cause of climate change?"

# 使用检索器查找相关文本块
context = retrieve_context_per_question(test_query, chunks_query_retriever)

# 显示检索结果
show_context(context)
```

> **💡 代码解释**
> - `retrieve_context_per_question`：辅助函数，执行搜索并返回相关文本
> - `show_context`：辅助函数，格式化显示检索结果

### 预期输出示例

```
查询："What is the main cause of climate change?"

检索到的相关内容：
[1] "Climate change is primarily caused by human activities, particularly
     the burning of fossil fuels like coal, oil, and natural gas. This
     releases greenhouse gases, mainly carbon dioxide, into the atmosphere..."

[2] "The greenhouse effect is the process through which certain gases
     in Earth's atmosphere trap heat from the sun, leading to warming..."
```

> **❓ 常见问题**
> - **Q**: 如果检索结果不相关怎么办？
> - **A**: 可以尝试：
>   1. 调整分块阈值（让块更完整）
>   2. 增加 k 值（返回更多结果）
>   3. 优化问题的表述方式

---

## 🧪 完整代码整合

下面是可以一次性运行的完整代码：

```python
# ========== 1. 安装和导入 ==========
!pip install langchain-experimental langchain-openai python-dotenv

import os
import sys
from dotenv import load_dotenv
from helper_functions import *
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
import faiss
from langchain.vectorstores import FAISS

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# ========== 2. 加载文档 ==========
os.makedirs('data', exist_ok=True)
# 下载或手动放置 PDF 到 data/目录
path = "data/Understanding_Climate_Change.pdf"
content = read_pdf_to_string(path)

# ========== 3. 语义分块 ==========
text_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type='percentile',
    breakpoint_threshold_amount=90
)
docs = text_splitter.create_documents([content])
print(f"文档被分成了 {len(docs)} 个块")

# ========== 4. 创建向量存储 ==========
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# ========== 5. 测试检索 ==========
test_query = "What is the main cause of climate change?"
context = retrieve_context_per_question(test_query, retriever)
show_context(context)
```

---

## 📊 参数调优指南

### 分块阈值调整

| 阈值 | 效果 | 适用场景 |
|------|------|---------|
| 70-80 | 块较小，数量多 | 需要精细检索的文档 |
| 85-92 | 块大小适中 | 通用场景（推荐） |
| 93-98 | 块较大，数量少 | 需要保持大段上下文的文档 |

### 检索数量 k 调整

| k 值 | 效果 | 适用场景 |
|------|------|---------|
| 1-2 | 返回最相关的少量内容 | 简单问题、减少上下文长度 |
| 3-5 | 平衡相关性和信息量 | 通用场景（推荐） |
| 6-10 | 返回更多背景信息 | 复杂问题、需要更多上下文 |

---

## ⚠️ 常见错误及解决方法

### 错误 1：API 密钥错误
```
Error: Invalid API key provided
```
**解决**：检查 `.env` 文件中的 API 密钥是否正确，确保没有多余空格

### 错误 2：模块未找到
```
ModuleNotFoundError: No module named 'langchain_experimental'
```
**解决**：重新运行 pip install 命令，确保安装成功

### 错误 3：PDF 文件未找到
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/...'
```
**解决**：确认 PDF 文件已下载到正确的路径，可以用 `!ls data/` 检查

### 错误 4：网络连接超时
```
URLError: <urlopen error [Errno 60] Operation timed out>
```
**解决**：
- 检查网络连接
- 使用代理或镜像源
- 手动下载文件后放入指定目录

---

## 🎓 学习总结

### 你学到了什么？

✅ **语义分块的概念**：比传统分块更智能，在语义变化处分割
✅ **三种断点类型**：percentile、standard_deviation、interquartile
✅ **完整实现流程**：PDF 加载 → 语义分块 → 向量存储 → 检索测试
✅ **参数调优技巧**：如何根据需求调整分块阈值和检索数量

### 下一步可以做什么？

1. 📚 尝试用自己的 PDF 文档测试效果
2. 🔧 调整分块参数，观察对检索结果的影响
3. 🔄 对比语义分块和传统分块（如 RecursiveCharacterTextSplitter）的差异
4. 🚀 结合其他 RAG 技术（如重排序、融合检索）进一步提升效果

---

## 📚 相关资源

- [LangChain 语义分块官方文档](https://python.langchain.com/docs/how_to/semantic-chunker/)
- [Greg Kamradt 的原始讲解视频](https://youtu.be/8OJC21T2SL4?t=1933)
- [FAISS 向量搜索库文档](https://github.com/facebookresearch/faiss)

---

*本教程是 RAG 技术系列教程之一。完成本教程后，你可以继续学习其他高级 RAG 技术，如上下文压缩、融合检索和重排序等。*

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--semantic-chunking)
