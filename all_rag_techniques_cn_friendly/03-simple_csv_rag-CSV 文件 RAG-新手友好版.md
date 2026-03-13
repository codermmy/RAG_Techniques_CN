# 🌟 新手入门：CSV 文件 RAG（检索增强生成）系统

> **💡 给新手的说明**
> - **难度等级**：⭐⭐☆☆☆（入门级）
> - **预计时间**：25-40 分钟
> - **前置知识**：基础 Python 编程知识，了解 CSV 文件格式
> - **学习目标**：学会处理结构化数据（CSV），构建客户信息问答系统

---

## 📖 核心概念理解

### 什么是 CSV 文件 RAG？

**CSV RAG** 是专门处理表格数据的 RAG 系统。与处理 PDF 不同，它需要处理结构化的行列数据。

### 🍕 通俗理解：Excel 表格查询机器人

想象一下你有一个超大的 Excel 客户信息表：

1. **传统方式**：你需要 manually 查找、过滤、排序才能找到信息
2. **CSV RAG 方式**：直接用自然语言提问，比如"张三在哪家公司工作？"，系统自动找到答案

**RAG 处理 CSV 的工作流程**：
```
CSV 文件 → 加载数据 → 转换成文档 → 创建向量 → 存储 → 检索 → 回答
```

### 📊 CSV 文件结构示例

```
FirstName,LastName,Company,Email,Phone
John,Doe,Acme Corp,john@example.com,555-0100
Jane,Smith,Global Inc,jane@example.com,555-0101
```

### 🔑 核心组件解释

| 组件 | 作用 | 生活比喻 |
|------|------|----------|
| **CSVLoader** | 读取 CSV 文件 | Excel 打开表格 |
| **文档拆分** | 将数据分成小块 | 把表格按行分组 |
| **FAISS 向量存储** | 存储和搜索向量 | 智能索引系统 |
| **OpenAI Embeddings** | 文本转数字向量 | 内容数字化标签 |
| **检索链** | 检索 + 生成答案 | 查询 + 回答的完整流程 |

---

## 🛠️ 第一步：环境准备

### 📖 这是什么？

安装运行 CSV RAG 系统所需的 Python 库。

### 💻 完整代码

```python
# ============================================
# 安装所需的包
# ============================================
# 每个包的作用：
# - faiss-cpu: Facebook 的高效相似度搜索库
# - langchain: RAG 框架核心
# - langchain-community: 社区扩展组件
# - langchain-openai: OpenAI 集成
# - pandas: 数据处理和分析库
# - python-dotenv: 环境变量管理

!pip install faiss-cpu langchain langchain-community langchain-openai pandas python-dotenv
```

> **💡 代码解释**
> - 这一行命令会安装所有需要的包
> - 安装可能需要几分钟，请耐心等待
>
> **⚠️ 新手注意**
> - 如果遇到安装错误，可以逐个包安装
> - 国内用户可使用清华源加速：
>   ```
>   !pip install faiss-cpu -i https://pypi.tuna.tsinghua.edu.cn/simple
>   ```

---

## 🔑 第二步：导入库和配置

### 📖 这是什么？

导入需要用到的所有库，并设置 API 密钥。

### 💻 完整代码

```python
# ============================================
# 导入必要的库
# ============================================
from langchain_community.document_loaders.csv_loader import CSVLoader
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from dotenv import load_dotenv

# 从 .env 文件加载环境变量
load_dotenv()

# 设置 OpenAI API 密钥环境变量
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# 初始化语言模型
# gpt-3.5-turbo-0125 是 OpenAI 的一个高效模型
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
```

> **💡 代码解释**
> - `CSVLoader`：专门用来读取 CSV 文件的工具
> - `ChatOpenAI`：OpenAI 的聊天模型，用来生成答案
> - `OpenAIEmbeddings`：将文本转换为向量的工具
> - `load_dotenv()`：从 `.env` 文件加载配置
>
> **⚠️ 新手注意**
> - **API 密钥安全**：不要直接把密钥写在代码里！
> - 推荐做法是创建 `.env` 文件：
>   ```
>   # .env 文件内容
>   OPENAI_API_KEY=sk-your-actual-key-here
>   ```
> - 确保 `.env` 文件不要提交到 Git 仓库（加入 `.gitignore`）
>
> **❓ 常见问题**
> - **Q: 没有 API 密钥怎么办？**
> - A: 需要到 OpenAI 官网注册账号并创建 API 密钥

---

## 📄 第三步：下载和查看数据

### 📖 这是什么？

下载示例 CSV 文件，了解数据长什么样。

### 💻 完整代码

```python
# ============================================
# 创建 data 目录并下载示例数据
# ============================================
import os
os.makedirs('data', exist_ok=True)

# 下载示例 CSV 文件（包含虚拟客户数据）
!wget -O data/customers-100.csv https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/customers-100.csv

# 你也可以下载 PDF 作为额外参考（可选）
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
```

> **💡 代码解释**
> - `os.makedirs('data', exist_ok=True)` 创建 data 目录
> - `!wget` 从网络下载文件
> - 下载的是包含 100 个虚拟客户信息的 CSV 文件
>
> **⚠️ 新手注意**
> - 如果下载失败，可以手动下载后放到 `data` 目录
> - 你也可以使用自己的 CSV 文件

### 📊 查看 CSV 数据

```python
# 使用 pandas 读取并预览数据
import pandas as pd

file_path = 'data/customers-100.csv'  # CSV 文件路径
data = pd.read_csv(file_path)

# 显示前几行数据
data.head()
```

> **💡 预期输出**
> ```
>   Index  Customer Id  First Name  Last Name     Company  ...
>      1          1     Sheryl     Baxter        Acme Corp  ...
>      2          2     John       Doe         Global Inc   ...
>      ...
> ```
>
> **📊 术语解释**
> - `head()`：显示 DataFrame 的前 5 行，用于快速预览数据

---

## 📥 第四步：加载和拆分 CSV 数据

### 📖 这是什么？

使用 LangChain 的 CSVLoader 加载数据，并拆分成小块以便处理。

### 💻 完整代码

```python
# ============================================
# 加载 CSV 文件
# ============================================
loader = CSVLoader(file_path=file_path)
docs = loader.load_and_split()
```

> **💡 代码解释**
> - `CSVLoader` 会自动解析 CSV 文件
> - `load_and_split()` 加载数据并自动拆分
> - 每一行数据会被转换成一个"文档"对象
>
> **📊 数据格式说明**
> 加载后的数据格式类似于：
> ```
> Document 1:
> Index: 1
> Customer Id: 1
> First Name: Sheryl
> Last Name: Baxter
> Company: Acme Corp
> ...
>
> Document 2:
> Index: 2
> Customer Id: 2
> ...
> ```
>
> **⚠️ 新手注意**
> - 如果 CSV 文件很大，加载可能需要一些时间
> - 可以查看 `len(docs)` 了解拆分成了多少文档

---

## 🗄️ 第五步：创建向量存储

### 📖 这是什么？

创建 FAISS 向量存储，这是 RAG 系统的"记忆库"。

### 💻 完整代码

```python
# ============================================
# 初始化 FAISS 向量存储
# ============================================
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

# 创建 Embeddings 对象
embeddings = OpenAIEmbeddings()

# 创建 FAISS 索引
# IndexFlatL2 是最常用的索引类型，使用欧几里得距离计算相似度
index = faiss.IndexFlatL2(len(OpenAIEmbeddings().embed_query(" ")))

# 创建向量存储
vector_store = FAISS(
    embedding_function=OpenAIEmbeddings(),  # 使用的嵌入模型
    index=index,                             # FAISS 索引
    docstore=InMemoryDocstore(),            # 内存文档存储
    index_to_docstore_id={}                  # 索引到文档 ID 的映射
)
```

> **💡 代码解释**
>
> **FAISS 是什么？**
> - Facebook AI 开发的快速相似度搜索库
> - 就像图书馆的索引系统，可以秒级找到相似内容
>
> **IndexFlatL2 是什么？**
> - L2 代表欧几里得距离（直线距离）
> - 计算两个向量之间的距离，距离越近越相似
>
> **⚠️ 新手注意**
> - 这段代码看起来复杂，但核心就是创建一个可以存储和搜索向量的数据库
> - 对于初学者，可以用更简单的方式：
>   ```python
>   vector_store = FAISS.from_documents(docs, embeddings)
>   ```

---

## ➕ 第六步：添加数据到向量存储

### 📖 这是什么？

将拆分好的 CSV 数据添加到向量存储中。

### 💻 完整代码

```python
# ============================================
# 将文档添加到向量存储
# ============================================
vector_store.add_documents(documents=docs)
```

> **💡 代码解释**
> - 这一步会为每个文档创建向量表示
> - 向量会被存储到 FAISS 索引中
> - 完成后就可以进行检索了
>
> **⚠️ 新手注意**
> - 如果文档很多，这一步可能需要一些时间
> - 每次运行都会调用 OpenAI API，会产生少量费用

---

## 🔗 第七步：创建检索链

### 📖 这是什么？

检索链是 RAG 系统的核心，它把"检索"和"生成答案"两个步骤连接起来。

### 💻 完整代码

```python
# ============================================
# 创建检索链
# ============================================
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# 创建检索器
retriever = vector_store.as_retriever()

# 设置系统提示词
# 这告诉 AI 如何 behaved
system_prompt = (
    "你是一个问答任务的助手。"
    "使用以下检索到的上下文片段来回答"
    "问题。如果你不知道答案，就说你"
    "不知道。最多使用三句话，保持"
    "答案简洁。"
    "\n\n"
    "{context}"
)

# 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),  # 系统指令
    ("human", "{input}"),       # 用户输入

])

# 创建问答链
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# 创建完整的检索链
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
```

> **💡 代码解释**
>
> **检索链的工作流程**：
> 1. 用户提问 → `retriever` 检索相关文档
> 2. 检索到的文档 + 问题 → 传给 `llm`（语言模型）
> 3. `llm` 基于文档生成答案
> 4. 返回最终答案
>
> **提示词的作用**：
> - 告诉 AI 如何使用检索到的信息
> - 限制答案长度（"最多三句话"）
> - 设定行为准则（"不知道就说不知道"）
>
> **⚠️ 新手注意**
> - `create_stuff_documents_chain` 会把所有检索到的文档"塞进"提示词
> - 对于大量文档，可能需要更高级的方法

---

## 💬 第八步：向 RAG 机器人提问

### 📖 这是什么？

终于到了测试环节！向我们的 CSV RAG 系统提问。

### 💻 完整代码

```python
# ============================================
# 测试问答系统
# ============================================
# 向 RAG 机器人提问关于 CSV 数据的问题
answer = rag_chain.invoke({"input": "Sheryl Baxter 在哪家公司工作？"})
print(answer['answer'])
```

> **💡 预期输出**
> ```
> Sheryl Baxter 在 Acme Corp 工作。
> ```
>
> **⚠️ 新手注意**
> - 实际输出可能因数据而异
> - 如果答案不准确，可能是因为：
>   - 检索到的文档不包含正确答案
>   - 问题表述不够清晰
>   - 需要调整检索参数
>
> **🧪 更多测试问题**
> ```python
> # 尝试不同问题
> questions = [
>     "谁是 John Doe 的老板？",
>     "有哪些公司在纽约？",
>     "Sheryl 的邮箱是什么？",
>     "有多少客户？"
> ]
>
> for q in questions:
>     result = rag_chain.invoke({"input": q})
>     print(f"问题：{q}")
>     print(f"答案：{result['answer']}\n")
> ```

---

## 🎯 完整代码总结

下面是一个可以独立运行的简化版本：

```python
# 1. 导入必要的库
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain, create_stuff_documents_chain
import os

# 2. 设置 API 密钥
os.environ["OPENAI_API_KEY"] = "你的 API 密钥"

# 3. 加载 CSV 数据
loader = CSVLoader(file_path="data/customers-100.csv")
docs = loader.load_and_split()

# 4. 创建向量存储
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(docs, embeddings)

# 5. 创建检索链
retriever = vector_store.as_retriever()

system_prompt = """你是一个问答助手。使用检索到的上下文回答问题。
如果你不知道答案，就说不知道。保持答案简洁。

{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

llm = ChatOpenAI(model="gpt-3.5-turbo")
qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# 6. 提问测试
result = rag_chain.invoke({"input": "Sheryl Baxter 在哪家公司工作？"})
print(result['answer'])
```

---

## ❓ 常见问题 FAQ

### Q1: CSV 文件有什么格式要求？
**A**:
- 第一行应该是列名（标题行）
- 使用逗号分隔值
- 确保编码是 UTF-8（避免中文乱码）
- 没有特殊字符或格式

### Q2: 可以处理多大的 CSV 文件？
**A**:
- 小文件（<1000 行）：完全没问题
- 中等文件（1000-10000 行）：可以处理，但可能需要更多内存
- 大文件（>10000 行）：考虑分批处理或使用数据库

### Q3: 如何处理中文 CSV 文件？
**A**:
- 确保 CSV 文件使用 UTF-8 编码
- 在加载时指定编码：
  ```python
  loader = CSVLoader(file_path="data/客户信息.csv", encoding="utf-8")
  ```

### Q4: 答案不准确怎么办？
**A**:
- 增加检索数量：`retriever = vector_store.as_retriever(search_kwargs={"k": 5})`
- 优化提示词，给 AI 更清晰的指令
- 检查 CSV 数据质量

### Q5: 可以不用 OpenAI 吗？
**A**:
- 可以！可以使用其他 Embedding 模型
- 例如：HuggingFace 的免费模型
- 或者使用本地运行的模型

---

## 🚀 进阶技巧

### 自定义检索参数

```python
# 调整检索结果数量
retriever = vector_store.as_retriever(
    search_type="similarity",  # 相似度搜索
    search_kwargs={"k": 5}     # 返回 5 个最相关结果
)
```

### 添加更多提示词约束

```python
system_prompt = """你是一个专业的客服助手。
请基于以下客户信息回答问题：
{context}

要求：
1. 只回答与 CSV 数据相关的问题
2. 如果信息不存在，明确告知
3. 保持专业和礼貌
4. 答案控制在 3 句话以内
"""
```

---

## 📚 关键知识点回顾

| 概念 | 说明 |
|------|------|
| **CSV** | 逗号分隔值文件，常用于存储表格数据 |
| **CSVLoader** | LangChain 中专门加载 CSV 文件的工具 |
| **向量存储** | 将文本转换为向量并存储的数据库 |
| **检索链** | 连接检索和生成的完整流程 |
| **Embedding** | 将文本转换为数字向量的技术 |
| **FAISS** | Facebook 开发的高效相似度搜索库 |

---

## 🎓 与 PDF RAG 的区别

| 特性 | PDF RAG | CSV RAG |
|------|---------|---------|
| **数据结构** | 非结构化文本 | 结构化表格 |
| **加载方式** | PyPDFLoader | CSVLoader |
| **分块策略** | 按字符数切分 | 通常按行分组 |
| **适用场景** | 文档问答 | 数据查询 |

---

*本教程是 RAG 技术系列教程的 CSV 专题，建议先学习基础 RAG 教程再学习本教程。*

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--simple-csv-rag)
