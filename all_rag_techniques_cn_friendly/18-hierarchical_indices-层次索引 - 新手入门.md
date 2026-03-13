# 🌟 新手入门：层次索引检索

> **💡 给新手的说明**
> - **难度等级**：⭐⭐⭐（中等，需要基础 Python 和 RAG 知识）
> - **预计学习时间**：45-60 分钟
> - **前置知识**：了解基本的向量检索、Embedding 概念
> - **本教程特色**：包含完整代码注释、常见问题解答、避坑指南
>
> **📚 什么是层次索引？** 想象你在图书馆找书。如果直接翻遍所有书（传统方法），效率很低。但如果你先看目录卡片（摘要层），找到相关书籍后再去书架取具体章节（详细层），就会快很多。这就是层次索引的核心思想！

---

## 📖 核心概念理解

### 通俗理解：两层搜索系统

**传统检索的问题**：
假设你有一份 100 页的报告，传统方法会把整份报告切成小块，然后逐块搜索。这就像在一本厚厚的书中逐页寻找答案——费时费力。

**层次索引的解决方案**：
```
第一层（摘要层）：快速浏览每页的摘要 → 找到最相关的几页
第二层（详细层）：在找到的相关页面中深入查找具体内容
```

**生活化比喻**：
- 🗺️ **看地图找路**：先看全国地图（摘要）确定省份，再看城市地图（详细）找到具体街道
- 📑 **查字典**：先看部首索引（摘要）找到大概位置，再逐字查找（详细）
- 🏪 **超市购物**：先看楼层指示牌（摘要）找到食品区，再在食品区找具体商品（详细）

### 核心术语解释

| 术语 | 通俗解释 |
|------|----------|
| **向量存储（Vector Store）** | 把文本转换成数字向量后存储的"智能仓库"，可以按语义相似度快速查找 |
| **Embedding（嵌入）** | 把文字转换成数字向量的技术，让计算机能理解文本的"含义距离" |
| **FAISS** | Facebook 开源的快速向量搜索库，就像一个超高效的"向量仓库管理员" |
| **分块（Chunking）** | 把长文档切成小段，方便处理和检索 |
| **异步处理（Async）** | 同时处理多个任务，不用等一个做完再做下一个 |

---

## 🛠️ 第一步：安装必要的包

### 📖 这是什么？
这些是运行本教程所需的基础工具包：
- `langchain`：RAG 系统的核心框架
- `langchain-openai`：连接 OpenAI 服务的桥梁
- `python-dotenv`：安全地管理 API 密钥

### 💻 完整代码

```python
# 安装所需的包
# ⚠️ 新手注意：运行此命令前请确保已安装 pip
# 如果安装失败，可以尝试添加 --user 参数
!pip install langchain langchain-openai python-dotenv
```

```python
# 克隆仓库以访问辅助函数和评估模块
# 🔍 这个仓库包含了一些现成的工具函数，我们可以直接使用
!git clone https://github.com/NirDiamant/RAG_TECHNIQUES.git
import sys
sys.path.append('RAG_TECHNIQUES')

# 如果需要使用最新数据运行，可以取消下面这行的注释
# !cp -r RAG_TECHNIQUES/data .
```

```python
# 导入所有需要的库
import asyncio      # 用于异步处理，让多个任务可以同时进行
import os           # 用于操作文件和目录
import sys          # 用于系统相关的操作
from dotenv import load_dotenv  # 用于加载环境变量
from langchain_openai import ChatOpenAI  # OpenAI 的聊天模型
from langchain.chains.summarize.chain import load_summarize_chain  # 用于生成摘要
from langchain.docstore.document import Document  # 文档对象

# 导入辅助函数（来自克隆的仓库）
from helper_functions import *
from evaluation.evalute_rag import *
from helper_functions import encode_pdf, encode_from_string

# 从 .env 文件加载环境变量
# 💡 这步很重要：你的 API 密钥应该放在 .env 文件中，而不是直接写在代码里
load_dotenv()

# 设置 OpenAI API 密钥环境变量
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
```

> **💡 代码解释**
> - `load_dotenv()`：从 `.env` 文件读取配置，这样你就不用把 API 密钥硬编码在代码里
> - `os.getenv('OPENAI_API_KEY')`：从环境变量中获取 API 密钥
>
> **⚠️ 新手注意**
> - 如果你还没有 OpenAI API 密钥，需要先在 https://platform.openai.com 注册获取
> - 创建 `.env` 文件，内容格式：`OPENAI_API_KEY=sk-your-key-here`
> - **不要**把 `.env` 文件上传到 Git，这会泄露你的密钥！

### ❓ 常见问题

**Q1: pip install 失败怎么办？**
```
尝试以下方法：
1. 升级 pip: python -m pip install --upgrade pip
2. 使用国内镜像：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple langchain
3. 检查 Python 版本：建议使用 Python 3.8+
```

**Q2: 克隆仓库失败？**
```
可能是网络问题，可以：
1. 使用代理
2. 手动下载 ZIP 文件解压
3. 使用镜像仓库
```

---

## 🛠️ 第二步：准备数据

### 📖 这是什么？
我们需要一个 PDF 文档来演示层次索引的效果。这里使用《理解气候变化》作为示例文档。

### 💻 完整代码

```python
# 创建 data 目录（如果不存在）
import os
os.makedirs('data', exist_ok=True)

# 下载本笔记本使用的 PDF 文档
# 📥 这会从 GitHub 下载示例 PDF 文件
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
```

```python
# 定义 PDF 文件路径
path = "data/Understanding_Climate_Change.pdf"
```

> **💡 代码解释**
> - `os.makedirs('data', exist_ok=True)`：创建 data 目录，`exist_ok=True` 表示如果目录已存在也不会报错
> - `!wget -O`：使用 wget 下载文件，`-O` 指定保存的文件名
>
> **⚠️ 新手注意**
> - 如果 wget 不可用，可以用 `curl -o` 替代
> - 下载失败的话，可以用任何 PDF 文档替代，只需修改 `path` 变量

---

## 🛠️ 第三步：实现层次索引编码函数

### 📖 这是什么？
这是本教程的**核心函数**！它会把 PDF 文档编码成两个向量存储：
1. **摘要层**：每页文档的摘要
2. **详细层**：原始文档的详细文本块

### 💻 完整代码

```python
async def encode_pdf_hierarchical(path, chunk_size=1000, chunk_overlap=200, is_string=False):
    """
    使用 OpenAI embeddings 异步将 PDF 书籍编码为层次向量存储。
    包含使用指数退避的速率限制处理。

    Args:
        path: PDF 文件的路径（或者如果 is_string=True 则是文本字符串）。
        chunk_size: 每个文本块的期望大小（默认 1000 字符）。
        chunk_overlap: 连续块之间的重叠量（默认 200 字符），重叠可以避免信息被切断。
        is_string: 如果 path 是文本字符串而不是文件路径，设为 True。

    Returns:
        包含两个 FAISS 向量存储的元组：
        1. 文档级别摘要向量存储
        2. 详细文本块向量存储
    """

    # ==================== 步骤 1: 加载 PDF 文档 ====================
    if not is_string:
        # 从 PDF 文件加载
        loader = PyPDFLoader(path)
        documents = await asyncio.to_thread(loader.load)
    else:
        # 从文本字符串加载
        text_splitter = RecursiveCharacterTextSplitter(
            # 设置一个非常小的块大小，仅用于演示
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        documents = text_splitter.create_documents([path])

    # ==================== 步骤 2: 为每个文档生成摘要 ====================
    # 使用 GPT-4o-mini 模型来生成摘要（这个模型又快又便宜）
    summary_llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=4000)
    summary_chain = load_summarize_chain(summary_llm, chain_type="map_reduce")

    async def summarize_doc(doc):
        """
        使用速率限制处理摘要单个文档。

        Args:
            doc: 要被摘要的文档。

        Returns:
            摘要后的 Document 对象，包含摘要内容和元数据。
        """
        # 使用指数退避重试摘要（防止 API 速率限制）
        summary_output = await retry_with_exponential_backoff(summary_chain.ainvoke([doc]))
        summary = summary_output['output_text']

        # 创建包含摘要的新文档对象
        return Document(
            page_content=summary,
            metadata={"source": path, "page": doc.metadata["page"], "summary": True}
        )

    # ==================== 步骤 3: 分批处理文档（避免速率限制） ====================
    batch_size = 5  # 每批处理 5 个文档，根据你的 API 速率限制调整此值
    # ⚠️ 新手注意：如果你的 API 限制更严格，可以减小这个值
    summaries = []

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        # 并行处理当前批次的所有文档
        batch_summaries = await asyncio.gather(*[summarize_doc(doc) for doc in batch])
        summaries.extend(batch_summaries)
        await asyncio.sleep(1)  # 批次之间短暂暂停，避免触发速率限制

    # ==================== 步骤 4: 将文档分割为详细文本块 ====================
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    detailed_chunks = await asyncio.to_thread(text_splitter.split_documents, documents)

    # ==================== 步骤 5: 更新详细文本块的元数据 ====================
    for i, chunk in enumerate(detailed_chunks):
        chunk.metadata.update({
            "chunk_id": i,              # 块 ID
            "summary": False,           # 标记这不是摘要
            "page": int(chunk.metadata.get("page", 0))  # 页码
        })

    # ==================== 步骤 6: 创建 Embeddings ====================
    embeddings = OpenAIEmbeddings()

    # ==================== 步骤 7: 创建向量存储（带速率限制处理） ====================
    async def create_vectorstore(docs):
        """
        使用速率限制处理从文档列表创建向量存储。

        Args:
            docs: 要被嵌入的文档列表。

        Returns:
            包含嵌入文档的 FAISS 向量存储。
        """
        return await retry_with_exponential_backoff(
            asyncio.to_thread(FAISS.from_documents, docs, embeddings)
        )

    # ==================== 步骤 8: 并发创建两个向量存储 ====================
    # asyncio.gather 让两个任务同时进行，节省时间
    summary_vectorstore, detailed_vectorstore = await asyncio.gather(
        create_vectorstore(summaries),      # 创建摘要向量存储
        create_vectorstore(detailed_chunks) # 创建详细文本块向量存储
    )

    return summary_vectorstore, detailed_vectorstore
```

> **💡 代码解释**
>
> **关于异步（async/await）**：
> - `async def` 定义异步函数，可以"暂停"让其他任务先执行
> - `await` 等待异步操作完成
> - `asyncio.gather()` 让多个任务同时运行
>
> **关于参数**：
> - `chunk_size=1000`：每个文本块约 1000 字符
> - `chunk_overlap=200`：相邻块重叠 200 字符，防止信息被切断
> - `batch_size=5`：每批处理 5 个文档，避免触发 API 限制
>
> **⚠️ 新手注意**
> 1. **速率限制**：OpenAI API 有调用频率限制，批量处理时需要控制节奏
> 2. **内存消耗**：大文件会消耗大量内存，建议分批次处理
> 3. **费用控制**：生成摘要会消耗 Token，注意监控使用量

### ❓ 常见问题

**Q1: 什么是异步处理？为什么需要它？**
```
想象你去餐厅吃饭：
- 同步处理：点菜 → 等着 → 上菜 → 吃 → 再点下一道
- 异步处理：点所有菜 → 聊天玩手机 → 菜陆续上 → 一起吃

异步可以让多个 API 调用同时进行，大大加快速度！
```

**Q2: chunk_overlap 的作用是什么？**
```
假设文档是："今天天气很好，我们一起去公园玩。"
- 没有重叠：["今天天气很好", "我们一起去公园玩"]
- 有重叠 2 字符：["今天天气很好，我", "，我们一起去公园玩"]

重叠可以确保语义完整的句子不被切断！
```

**Q3: 为什么要分批处理？**
```
OpenAI API 有速率限制，比如每分钟只能调用 60 次。
如果一次性发送 100 个请求，会被拒绝。
分批处理（比如每批 5 个）+ 批次间暂停，可以避免这个问题。
```

---

## 🛠️ 第四步：执行编码并保存向量存储

### 📖 这是什么？
这一步会调用我们刚才定义的函数，把 PDF 编码成向量存储，并保存到本地。下次使用时可以直接加载，不用重新计算。

### 💻 完整代码

```python
# 检查向量存储是否已经存在
if os.path.exists("../vector_stores/summary_store") and os.path.exists("../vector_stores/detailed_store"):
    # 如果已存在，直接加载（省时省钱！）
    embeddings = OpenAIEmbeddings()
    summary_store = FAISS.load_local("../vector_stores/summary_store", embeddings, allow_dangerous_deserialization=True)
    detailed_store = FAISS.load_local("../vector_stores/detailed_store", embeddings, allow_dangerous_deserialization=True)
else:
    # 如果不存在，执行编码
    print("开始编码文档...")
    summary_store, detailed_store = await encode_pdf_hierarchical(path)

    # 保存到本地
    print("保存向量存储...")
    summary_store.save_local("../vector_stores/summary_store")
    detailed_store.save_local("../vector_stores/detailed_store")
    print("完成！")
```

> **💡 代码解释**
> - `os.path.exists()`：检查文件或目录是否存在
> - `FAISS.load_local()`：从本地加载已保存的向量存储
> - `save_local()`：保存向量存储到本地
> - `allow_dangerous_deserialization=True`：允许从本地加载（自己生成的文件是安全的）
>
> **⚠️ 新手注意**
> - 第一次运行会调用 OpenAI API，需要付费账户
> - 保存的路径是相对于当前目录的，确保父目录存在
> - 如果路径错误，会创建失败

### 💰 费用估算

```
假设 PDF 有 10 页，每页约 500 字：
- 生成摘要：10 页 × 500 字 × $0.15/1M 输入 ≈ $0.00075
- 创建 Embedding: 约 50 个文本块 × $0.0001/1K 字 ≈ $0.005
总费用：约 $0.006（非常便宜！）
```

---

## 🛠️ 第五步：实现层次检索函数

### 📖 这是什么？
这是另一个**核心函数**！它实现了两层搜索：
1. 先在摘要层搜索，找到最相关的几个摘要
2. 对于每个相关摘要，再在详细层搜索对应页面的具体内容

### 💻 完整代码

```python
def retrieve_hierarchical(query, summary_vectorstore, detailed_vectorstore, k_summaries=3, k_chunks=5):
    """
    使用查询执行层次检索。

    Args:
        query: 搜索查询（比如 "什么是温室效应？"）。
        summary_vectorstore: 包含文档摘要的向量存储（第一层）。
        detailed_vectorstore: 包含详细文本块的向量存储（第二层）。
        k_summaries: 要检索的顶部摘要数量（默认 3）。
        k_chunks: 每个摘要要检索的详细文本块数量（默认 5）。

    Returns:
        相关详细文本块的列表。
    """

    # ==================== 第一层：检索顶部摘要 ====================
    # 在摘要向量存储中搜索与查询最相似的 k 个摘要
    top_summaries = summary_vectorstore.similarity_search(query, k=k_summaries)

    relevant_chunks = []

    # ==================== 第二层：为每个摘要检索详细文本块 ====================
    for summary in top_summaries:
        # 获取这个摘要对应的页码
        page_number = summary.metadata["page"]

        # 创建一个过滤器，只检索同一页的详细文本块
        # 💡 这就是层次索引的关键：先定位相关页面，再深入查找
        page_filter = lambda metadata: metadata["page"] == page_number

        # 在详细向量存储中搜索，但只返回同一页的结果
        page_chunks = detailed_vectorstore.similarity_search(
            query,
            k=k_chunks,
            filter=page_filter  # 只检索同一页的内容
        )

        # 把找到的文本块加入结果列表
        relevant_chunks.extend(page_chunks)

    return relevant_chunks
```

> **💡 代码解释**
>
> **层次检索的工作流程**：
> ```
> 用户查询："什么是温室效应？"
>        ↓
>   [第一层搜索] 在摘要中查找 → 找到第 2、5、7 页的摘要最相关
>        ↓
>   [第二层搜索] 分别在第 2、5、7 页查找详细内容
>        ↓
>   返回：来自这 3 页的共 15 个详细文本块
> ```
>
> **⚠️ 新手注意**
> - `k_summaries` 和 `k_chunks` 的选择：
>   - 增大 `k_summaries`：覆盖更多页面，但可能引入不相关内容
>   - 增大 `k_chunks`：每页获取更多信息，但结果会更多
> - 默认值（3 和 5）是经验值，可以根据需要调整

### ❓ 常见问题

**Q1: 为什么要分页过滤？**
```
如果不分页过滤，第二层搜索会返回所有页面的相关内容，
这就退化成普通的全文搜索了。分页过滤确保我们只从
最相关的几个部分获取信息，保持层次结构的优势。
```

**Q2: similarity_search 是什么原理？**
```
1. 把你的查询转换成向量（Embedding）
2. 计算查询向量与所有文档向量的相似度（比如余弦相似度）
3. 返回相似度最高的 k 个文档
```

---

## 🛠️ 第六步：测试检索效果

### 📖 这是什么？
让我们实际运行一下，看看层次检索的效果如何！

### 💻 完整代码

```python
# 定义一个测试查询
query = "What is the greenhouse effect?"

# 执行层次检索
results = retrieve_hierarchical(query, summary_store, detailed_store)

# 打印结果
print(f"查询：{query}")
print(f"找到 {len(results)} 个相关文本块\n")
print("=" * 50)

for i, chunk in enumerate(results, 1):
    print(f"\n【结果 {i}】")
    print(f"页码：{chunk.metadata['page']}")
    # 打印前 200 个字符（避免输出太长）
    print(f"内容：{chunk.page_content[:200]}...")
    print("-" * 50)
```

> **💡 预期输出示例**
> ```
> 查询：What is the greenhouse effect?
> 找到 15 个相关文本块
>
> ==================================================
>
> 【结果 1】
> 页码：2
> 内容：The greenhouse effect is a natural process that warms the Earth's surface...
> --------------------------------------------------
>
> 【结果 2】
> 页码：2
> 内容：Greenhouse gases trap heat in the atmosphere...
> --------------------------------------------------
> ```

### ⚠️ 新手注意

1. **结果数量**：默认返回 `k_summaries × k_chunks = 3 × 5 = 15` 个文本块
2. **结果顺序**：结果按摘要的相关性分组，同一组来自同一页
3. **内容截断**：示例中只打印前 200 字符，实际可以访问完整内容

---

## 📊 可视化理解

下面是层次索引的工作流程图：

```
┌─────────────────────────────────────────────────────────────┐
│                    层次索引检索流程                          │
└─────────────────────────────────────────────────────────────┘

                    用户查询
                      │
                      ▼
        ┌─────────────────────────┐
        │   第一层：摘要搜索       │
        │   (快速定位相关部分)     │
        └─────────────────────────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
    ┌────────┐  ┌────────┐  ┌────────┐
    │摘要 1   │  │摘要 2   │  │摘要 3   │
    │(第 2 页) │  │(第 5 页) │  │(第 7 页) │
    └────┬───┘  └────┬───┘  └────┬───┘
         │            │            │
         ▼            ▼            ▼
    ┌────────┐  ┌────────┐  ┌────────┐
    │第二层  │  │第二层  │  │第二层  │
    │详细搜索 │  │详细搜索 │  │详细搜索 │
    │(第 2 页) │  │(第 5 页) │  │(第 7 页) │
    └────┬───┘  └────┬───┘  └────┬───┘
         │            │            │
         ▼            ▼            ▼
    ┌────────┐  ┌────────┐  ┌────────┐
    │块 1-5  │  │块 1-5  │  │块 1-5  │
    └────────┘  └────────┘  └────────┘
                      │
                      ▼
              返回所有相关文本块
```

---

## 📊 性能对比

### 层次索引 vs 传统扁平检索

| 特性 | 传统扁平检索 | 层次索引检索 |
|------|-------------|-------------|
| **搜索范围** | 所有文本块 | 先摘要，再部分内容 |
| **搜索速度** | O(n)，n=总块数 | O(m) + O(k), m=摘要数，k≪n |
| **上下文保持** | 可能丢失 | 更好，有摘要提供上下文 |
| **适用场景** | 小文档 | 大文档、长文档集 |
| **实现复杂度** | 简单 | 中等 |

---

## 🎯 本方法的优势总结

1. **🚀 提高检索效率**
   - 通过先搜索摘要，快速定位相关部分
   - 避免处理所有详细文本块

2. **📚 更好的上下文保持**
   - 摘要提供高层上下文
   - 检索的信息不会脱离原文背景

3. **📈 可扩展性强**
   - 特别适合大型文档集
   - 文档越多，优势越明显

4. **🔧 灵活可调**
   - 可以调整检索的摘要数量
   - 可以调整每个摘要的详细块数量

---

## ⚠️ 避坑指南

### 常见错误及解决方法

**错误 1: API 速率限制**
```
错误信息：RateLimitError: Rate limit reached
解决方法：
1. 减小 batch_size（从 5 改到 3）
2. 增加批次间暂停时间（从 1 秒改到 2 秒）
3. 升级到更高的 API 套餐
```

**错误 2: 内存不足**
```
错误信息：MemoryError
解决方法：
1. 减小 chunk_size
2. 分批处理大文档
3. 使用磁盘缓存
```

**错误 3: 向量存储加载失败**
```
错误信息：Could not load vector store
解决方法：
1. 检查路径是否正确
2. 确保使用相同的 Embedding 模型
3. 删除旧文件重新生成
```

**错误 4: PDF 加载失败**
```
错误信息：PyPDF2.errors.PdfReadError
解决方法：
1. 检查 PDF 文件是否损坏
2. 尝试用其他工具打开 PDF
3. 转换为文本后使用 is_string=True
```

---

## ❓ 新手常见问题

### Q1: 层次索引适合所有场景吗？

**答**：不是的。层次索引特别适合：
- ✅ 长篇文档（报告、论文、书籍）
- ✅ 需要保持上下文的场景
- ✅ 文档集较大（100+ 页）

不太适合：
- ❌ 短文档（摘要可能比原文还长）
- ❌ 对速度要求极高的实时场景
- ❌ 预算有限（生成摘要需要额外费用）

### Q2: 可以自定义摘要的生成方式吗？

**答**：当然可以！你可以：
```python
# 修改摘要提示
summary_chain = load_summarize_chain(
    summary_llm,
    chain_type="map_reduce",
    return_intermediate_steps=True  # 返回中间步骤
)
```

### Q3: 如何评估检索质量？

**答**：可以用以下指标：
- **召回率**：找到所有相关文档的比例
- **精确率**：返回的文档中有多少是相关的
- **响应时间**：检索耗时
- **用户满意度**：最终用户对答案的评价

---

## 🎓 进阶思考

### 如何进一步优化？

1. **多级层次**：不只是两层，可以是三层或更多
   - 章节摘要 → 段落摘要 → 详细文本

2. **智能分层**：根据内容类型决定分层策略
   - 技术文档：更多摘要层
   - 叙事文档：更少摘要层

3. **动态调整**：根据查询复杂度自动调整 k 值
   - 简单查询：少检索
   - 复杂查询：多检索

---

## 📝 实战练习

### 练习 1：修改参数测试效果
```python
# 尝试不同的参数组合
results1 = retrieve_hierarchical(query, summary_store, detailed_store, k_summaries=1, k_chunks=3)
results2 = retrieve_hierarchical(query, summary_store, detailed_store, k_summaries=5, k_chunks=10)

# 比较结果数量和质量
print(f"方案 1 找到 {len(results1)} 个结果")
print(f"方案 2 找到 {len(results2)} 个结果")
```

### 练习 2：用自己的 PDF 测试
```python
# 替换成你自己的 PDF 路径
my_path = "data/my_document.pdf"
summary_store, detailed_store = await encode_pdf_hierarchical(my_path)

# 测试检索
my_query = "你的问题"
results = retrieve_hierarchical(my_query, summary_store, detailed_store)
```

---

## 📚 总结

恭喜你完成了层次索引的学习！现在你已经：

✅ **理解了**层次索引的核心概念和优势
✅ **掌握了**实现层次索引的关键代码
✅ **学会了**如何处理 API 速率限制
✅ **能够**在自己的项目中应用此技术

**下一步学习建议**：
1. 尝试用不同的文档测试效果
2. 调整参数观察变化
3. 结合其他 RAG 技术使用
4. 学习下一篇：飞镖板检索（Dartboard RAG）

---

> **💪 记住**：理解比记忆更重要！多动手实践，遇到问题多思考，你一定能掌握这项技术。
>
> 如果本教程对你有帮助，欢迎分享给更多朋友！🌟
