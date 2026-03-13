# 🌟 新手入门：CSV 文件 RAG 系统（LlamaIndex 版）

> **💡 给新手的说明**
> - **难度等级**：⭐⭐☆☆☆（入门级）
> - **预计时间**：25-35 分钟
> - **前置知识**：基础 Python 编程知识，了解 CSV 文件格式
> - **学习目标**：学会使用 LlamaIndex 构建 CSV 数据的问答系统，实现结构化数据的智能查询

---

## 📖 核心概念理解

### 什么是 CSV RAG 系统？

**CSV RAG** 是将检索增强生成技术应用于结构化表格数据的系统。它让你能够用自然语言查询 CSV 文件中的数据。

### 🍕 通俗理解：智能表格助手

想象一下你有一个 Excel 表格，里面有 100 个客户的信息：

1. **传统查询方式**：你需要用 Excel 的筛选、排序功能，或者写 SQL 查询
2. **CSV RAG 方式**：你直接用自然语言提问，比如"Sheryl Baxter 在哪个公司工作？"，系统自动找到答案

**CSV RAG 的工作流程**：
```
你提问 → 系统理解问题 → 在 CSV 数据中查找 → 返回准确答案
```

### 🔑 核心组件解释

| 组件 | 作用 | 生活比喻 |
|------|------|----------|
| **PagedCSVReader** | 读取 CSV 文件并转换为文档 | 表格阅读助手 |
| **SimpleDirectoryReader** | 从目录加载文件 | 文件收集器 |
| **OpenAI Embedding** | 将文本转为向量 | 给数据打标签 |
| **FAISS 向量存储** | 存储和检索向量 | 数据索引柜 |
| **查询引擎** | 处理问题并返回答案 | 智能问答机器人 |

### 📊 CSV vs PDF RAG 对比

| 特性 | CSV RAG | PDF RAG |
|------|--------|--------|
| **数据结构** | 结构化（行列格式） | 非结构化（自由文本） |
| **处理方式** | 按行处理 | 需要分块 |
| **查询类型** | 精确查询为主 | 语义查询为主 |
| **典型应用** | 客户信息、产品目录 | 文档、报告、书籍 |

---

## 🛠️ 第一步：环境准备

### 📖 这是什么？

安装必要的 Python 库。CSV RAG 需要额外的 pandas 库来处理表格数据。

### 💻 完整代码

```python
# ============================================
# 安装所需的包
# ============================================
# 每个包的作用：
# - faiss-cpu: Facebook 的高效相似度搜索库
# - llama-index: LlamaIndex 框架核心
# - pandas: 数据处理和分析库（处理 CSV 必备）
# - python-dotenv: 管理 API 密钥

!pip install faiss-cpu llama-index pandas python-dotenv
```

> **💡 代码解释**
> - `pandas` 是 Python 中处理表格数据的核心库
> - 其他包与基础 RAG 教程相同
>
> **⚠️ 新手注意**
> - 如使用国内网络，可添加清华源
> - pandas 安装可能需要几分钟

---

## 🔑 第二步：配置 API 密钥和导入库

### 📖 这是什么？

设置 OpenAI API 密钥并导入所有需要的库。

### 💻 完整代码

```python
# ============================================
# 导入必要的库并配置 API 密钥
# ============================================
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import PagedCSVReader
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import VectorStoreIndex
import faiss
import os
import pandas as pd
from dotenv import load_dotenv

# 从 .env 文件加载环境变量
load_dotenv()

# 设置 OpenAI API 密钥
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# ============================================
# 配置 LlamaIndex 全局设置
# ============================================
EMBED_DIMENSION = 512  # Embedding 向量维度

# 设置使用的 LLM 模型
Settings.llm = OpenAI(model="gpt-3.5-turbo")

# 设置 Embedding 模型
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small", 
    dimensions=EMBED_DIMENSION
)
```

> **💡 代码解释**
> - `Settings.llm` 设置全局使用的语言模型
> - `gpt-3.5-turbo` 是性价比高的选择，适合问答任务
> - `Settings.embed_model` 设置全局 Embedding 模型

> **⚠️ 新手注意**
> - **API 密钥安全**：使用 `.env` 文件管理密钥
> - 如果使用 Google Colab，注意密钥不要泄露

> **📊 术语解释**
> - **LLM**：Large Language Model，大型语言模型
> - **Embedding**：将文本转换为数字向量的过程

---

## 📊 第三步：下载和预览 CSV 数据

### 📖 这是什么？

下载示例 CSV 文件并查看其结构。了解数据结构对后续处理很重要。

### 💻 完整代码

```python
# ============================================
# 创建目录并下载示例 CSV 文件
# ============================================
import os

# 创建 data 目录
os.makedirs('data', exist_ok=True)

# 下载示例 CSV 文件（包含 100 个虚拟客户数据）
!wget -O data/customers-100.csv https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/customers-100.csv

# 指定 CSV 文件路径
file_path = 'data/customers-100.csv'
```

> **💡 代码解释**
> - CSV 文件包含客户信息：姓名、公司、邮箱等
> - 这是一个标准的逗号分隔值文件

### 💻 使用 Pandas 预览数据

```python
# ============================================
# 使用 Pandas 读取并预览 CSV 数据
# ============================================
data = pd.read_csv(file_path)

# 显示前 5 行数据
print("CSV 文件前 5 行：")
print(data.head())

# 查看数据基本信息
print(f"\n数据形状：{data.shape} 行 x 列")
print(f"\n列名：{list(data.columns)}")
```

> **💡 代码解释**
> - `pd.read_csv()` 读取 CSV 文件为 DataFrame
> - `head()` 显示前 5 行数据
> - `shape` 显示数据的行数和列数

> **📊 预期输出示例**
> ```
> CSV 文件前 5 行：
>    First Name Last Name         Company  ...
> 0      Sheryl     Baxter    ACM Inc  ...
> 1       James    Johnston  Goodman LLC  ...
> ...
> 
> 数据形状：100 行 x 10 列
> 
> 列名：['First Name', 'Last Name', 'Company', 'City', ...]
> ```

---

## 🗄️ 第四步：创建向量存储

### 📖 这是什么？

创建 FAISS 向量存储，用于存储 CSV 数据的向量表示。

### 💻 完整代码

```python
# ============================================
# 创建 FAISS 向量存储
# ============================================
# 创建 FAISS 索引（使用 L2 距离）
fais_index = faiss.IndexFlatL2(EMBED_DIMENSION)

# 创建 LlamaIndex 的 FAISS 向量存储包装器
vector_store = FaissVectorStore(faiss_index=fais_index)
```

> **💡 代码解释**
> - 这与基础 RAG 教程中的代码相同
> - `IndexFlatL2` 使用欧几里得距离计算相似度

---

## 📄 第五步：加载 CSV 数据为文档

### 📖 这是什么？

使用 LlamaIndex 的专用 CSV 读取器将 CSV 文件转换为文档格式。

### 💻 完整代码

```python
# ============================================
# 使用 PagedCSVReader 加载 CSV 数据
# ============================================
# 创建 CSV 读取器
csv_reader = PagedCSVReader()

# 创建目录读取器，指定 CSV 文件的读取方式
reader = SimpleDirectoryReader(
    input_files=[file_path],  # 输入文件列表
    file_extractor={".csv": csv_reader}  # 指定.csv 文件的读取器
)

# 加载数据
docs = reader.load_data()

print(f"✓ 加载完成！共生成 {len(docs)} 个文档")
```

> **💡 代码解释**
> - `PagedCSVReader` 将 CSV 的每一行转换为一个 LlamaIndex Document
> - `file_extractor` 告诉 SimpleDirectoryReader 如何处理特定类型的文件
> - 每个文档包含一行的所有数据

### 💻 查看文档内容

```python
# ============================================
# 查看第一个文档的内容
# ============================================
print("第一个文档内容：")
print(docs[0].text)
```

> **💡 代码解释**
> - 每个文档的 text 属性包含该行的所有信息
> - 格式通常是"列名：值"的形式

> **📊 预期输出示例**
> ```
> 第一个文档内容：
> First Name: Sheryl
> Last Name: Baxter
> Company: ACM Inc
> City: ...
> Email: ...
> ...
> ```

---

## 🔄 第六步：运行数据摄入流水线

### 📖 这是什么？

数据摄入流水线将文档转换为向量并存储到 FAISS 中。

### 💻 完整代码

```python
# ============================================
# 创建并运行数据摄入流水线
# ============================================
pipeline = IngestionPipeline(
    vector_store=vector_store,  # 向量存储位置
    documents=docs,  # 要处理的文档
)

# 运行流水线
nodes = pipeline.run()

print(f"✓ 流水线运行完成！共生成 {len(nodes)} 个节点")
```

> **💡 代码解释**
> - `IngestionPipeline` 处理文档并存储到向量库
> - CSV 数据通常不需要额外的分块，因为每行已经是独立的信息单元
> - `nodes` 是处理后生成的节点列表

> **⚠️ 新手注意**
> - 对于大型 CSV 文件（数万行），处理可能需要一些时间
> - 节点数应该等于 CSV 行数

---

## 🔍 第七步：创建查询引擎

### 📖 这是什么？

查询引擎是用来问答 CSV 数据的接口。

### 💻 完整代码

```python
# ============================================
# 从节点创建向量存储索引
# ============================================
vector_store_index = VectorStoreIndex(nodes)

# ============================================
# 创建查询引擎
# ============================================
# similarity_top_k=2 表示每次检索获取最相似的 2 个结果
query_engine = vector_store_index.as_query_engine(similarity_top_k=2)
```

> **💡 代码解释**
> - `VectorStoreIndex` 创建可查询的索引
> - `as_query_engine()` 创建问答接口
> - `similarity_top_k` 控制检索的相关结果数量

> **⚠️ 新手注意**
> - 对于精确查询，`top_k=1` 可能就够了
> - 对于模糊查询，可以增加到 3-5

---

## 🧪 第八步：测试 CSV RAG 系统

### 📖 这是什么？

用自然语言问题测试 RAG 系统。

### 💻 完整代码

```python
# ============================================
# 测试查询：询问某人在哪个公司工作
# ============================================
response = query_engine.query("Which company does Sheryl Baxter work for?")

# 显示答案
print(f"问题：Which company does Sheryl Baxter work for?")
print(f"答案：{response.response}")
```

> **💡 代码解释**
> - `query()` 方法处理自然语言问题
> - `response.response` 包含最终答案

> **📊 预期输出示例**
> ```
> 问题：Which company does Sheryl Baxter work for?
> 答案：Sheryl Baxter works for ACM Inc.
> ```

### 💻 更多测试示例

```python
# ============================================
# 更多测试查询
# ============================================

# 测试 1：询问某个城市有哪些客户
response1 = query_engine.query("Which customers are from New York?")
print(f"问题：Which customers are from New York?")
print(f"答案：{response1.response}\n")

# 测试 2：询问某人的邮箱
response2 = query_engine.query("What is the email of James Johnston?")
print(f"问题：What is the email of James Johnston?")
print(f"答案：{response2.response}\n")

# 测试 3：统计类问题（可能不太准确）
response3 = query_engine.query("How many companies are in the data?")
print(f"问题：How many companies are in the data?")
print(f"答案：{response3.response}")
```

> **⚠️ 新手注意**
> - 简单查询（如查找某人信息）效果最好
> - 复杂统计问题可能不如直接计算准确
> - 对于统计需求，建议结合 pandas 使用

---

## 💡 进阶技巧：结合 Pandas 使用

### 📖 什么时候需要结合 Pandas？

RAG 适合语义查询，但统计计算用 Pandas 更准确：

### 💻 完整代码

```python
# ============================================
# 使用 Pandas 进行精确统计
# ============================================

# 统计公司数量（去重）
unique_companies = data['Company'].nunique()
print(f"唯一公司数量：{unique_companies}")

# 统计每个城市的客户数量
city_counts = data['City'].value_counts()
print(f"\n各城市客户数量：")
print(city_counts)

# 筛选特定条件的客户
ny_customers = data[data['City'].str.contains('New York', case=False)]
print(f"\n纽约客户列表：")
print(ny_customers[['First Name', 'Last Name', 'Company']])
```

> **💡 代码解释**
> - `nunique()` 计算唯一值数量
> - `value_counts()` 统计每个值的出现次数
> - 布尔索引可以筛选满足条件的行

---

## ⚠️ 常见问题与调试

### Q1: 查询结果不准确怎么办？

**可能的原因**：
1. Embedding 模型不理解某些专业术语
2. `top_k` 设置太小

**解决方案**：
- 尝试增加 `similarity_top_k` 到 3-5
- 在查询中使用更明确的词汇

### Q2: 如何处理大型 CSV 文件？

**解决方案**：
```python
# 分批处理大型 CSV
chunk_size = 1000  # 每次处理 1000 行
chunks = pd.read_csv(file_path, chunksize=chunk_size)

all_docs = []
for chunk in chunks:
    # 处理每个 chunk
    # ...
```

### Q3: 能否处理中文 CSV？

**解决方案**：
```python
# 指定编码读取中文 CSV
data = pd.read_csv(file_path, encoding='utf-8-sig')  # 处理 BOM

# 或者使用 gb18030 编码（Windows 常见）
data = pd.read_csv(file_path, encoding='gb18030')
```

### Q4: 如何处理包含特殊字符的数据？

**解决方案**：
```python
# 清理特殊字符
data = data.applymap(lambda x: str(x).strip() if isinstance(x, str) else x)
```

---

## 📚 总结

### 核心要点回顾

1. **CSV RAG 工作流程**：
   - 使用 `PagedCSVReader` 加载 CSV
   - 每行转换为一个文档
   - 创建向量索引
   - 用自然语言查询

2. **适用场景**：
   - 客户信息查询
   - 产品目录检索
   - 员工信息问答

3. **优势**：
   - 无需编写 SQL
   - 支持自然语言提问
   - 能理解模糊查询

### 进阶方向

1. **多表关联**：处理多个相关联的 CSV 文件
2. **混合查询**：结合 RAG 和 Pandas 的优势
3. **实时更新**：支持 CSV 数据的增量更新

---

## 🔗 相关资源

- [LlamaIndex CSV 读取器文档](https://docs.llamaindex.ai/en/stable/api_reference/readers/file/)
- [Pandas 官方教程](https://pandas.pydata.org/docs/user_guide/10min.html)
- [FAISS 文档](https://faiss.ai/)
