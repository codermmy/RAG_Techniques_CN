# 🌟 新手入门：假设性提示嵌入 (HyPE)

> **💡 给新手的说明**
> - **难度等级**：⭐⭐⭐⭐（中高级）
> - **预计学习时间**：45-60 分钟
> - **前置知识**：了解基础的 Python 编程，对 RAG 系统有基本认识
> - **本教程你将学会**：如何用 HyPE 技术解决查询 - 文档风格不匹配问题

---

## 📖 核心概念理解

### 什么是 HyPE？

**HyPE** = **Hy**pothetical **P**rompt **E**mbeddings（假设性提示嵌入）

### 通俗理解

**生活化比喻**：

想象你在整理一个问答知识库：

🔍 **传统 RAG 方法**：
- 你把文档切成一块一块存起来
- 用户问："气候变化原因？"
- 系统拿这个问题去匹配文档块
- **问题**：问题是疑问句，文档是陈述句，风格不匹配！

💡 **HyDE 方法**（对比参考）：
- 用户问问题时，临时生成一个假设性答案
- 用这个假设性答案去匹配文档
- **缺点**：每次查询都要等 LLM 生成答案，很慢！

🚀 **HyPE 方法**：
- **预先**为每个文档块生成多个"可能被问到的问题"
- 把这些"假设性问题"也转成向量存起来
- 用户问问题时，直接匹配预先存好的问题
- **优点**：问题匹配问题，风格一致；而且预先计算好了，查询超快！

### 核心思想对比

```
┌────────────────────────────────────────────────────────────────────────┐
│                      三种方法对比                                      │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  传统 RAG:                                                             │
│  文档块 ──→ 向量 ──→ 存储                                              │
│  查询 ──→ 向量 ──→ 匹配文档块                                         │
│  ⚠️ 查询和文档块风格不同（疑问句 vs 陈述句）                            │
│                                                                        │
│  HyDE:                                                                 │
│  查询 ──→ [LLM 生成假设答案] ──→ 向量 ──→ 匹配文档块                   │
│  ⚠️ 每次查询都要调用 LLM，慢！                                         │
│                                                                        │
│  HyPE:                                                                 │
│  文档块 ──→ [LLM 生成假设问题] ──→ 向量 ──→ 存储                        │
│  查询 ──→ 向量 ──→ 匹配假设问题 ──→ 返回文档块                         │
│  ✅ 问题匹配问题，风格一致；预先计算，查询快！                          │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 为什么 HyPE 有效？

| 问题 | 传统 RAG | HyDE | HyPE |
|------|---------|------|------|
| **风格不匹配** | ❌ 疑问句 vs 陈述句 | ✅ 假设答案 vs 陈述句 | ✅ 问题 vs 问题 |
| **查询延迟** | ✅ 快 | ❌ 慢（要生成） | ✅ 快（预先计算） |
| **检索精度** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 🛠️ 第一步：环境准备

### 📖 这是什么？

安装必要的 Python 包，并配置 API 密钥。

### 💻 完整代码

```python
# 安装所需的包
# !pip install faiss-cpu futures langchain-community python-dotenv tqdm
```

> **💡 代码解释**
> - `faiss-cpu`：Facebook 的向量搜索库
> - `futures`：Python 的并发处理库（多线程）
> - `langchain-community`：LangChain 社区扩展包
> - `python-dotenv`：环境变量管理
> - `tqdm`：进度条显示
>
> **⚠️ 新手注意**
> - 去掉 `!` 前面的 `#` 注释以在 Jupyter 中运行
> - 或在终端运行：`pip install faiss-cpu futures langchain-community python-dotenv tqdm`

### 克隆仓库获取辅助函数

```python
# 克隆仓库以访问辅助函数和评估模块
# !git clone https://github.com/NirDiamant/RAG_TECHNIQUES.git
import sys
sys.path.append('RAG_TECHNIQUES')
```

> **💡 代码解释**
> - 从 GitHub 克隆项目仓库
> - 把仓库路径添加到 Python 搜索路径
>
> **⚠️ 新手注意**
> - 如果已经克隆过，可以跳过这一步

### 导入库并配置环境变量

```python
import os
import sys
import faiss
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.docstore.in_memory import InMemoryDocstore

# 从 .env 文件加载环境变量
load_dotenv()

# 设置 OpenAI API 密钥环境变量
if not os.getenv('OPENAI_API_KEY'):
    os.environ["OPENAI_API_KEY"] = input("请输入您的 OpenAI API 密钥：")
else:
    os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# 导入辅助函数
from helper_functions import *
from evaluation.evalute_rag import *
```

> **💡 代码解释**
> - `faiss`：向量搜索库
> - `tqdm`：显示进度条
> - `ThreadPoolExecutor`：多线程并发处理
> - `InMemoryDocstore`：内存中的文档存储
> - `load_dotenv()`：从 `.env` 文件读取环境变量
>
> **⚠️ 新手注意**
> - 如果没有设置 API 密钥，程序会提示输入
> - 最好在 `.env` 文件中预先配置好

---

## 📏 第二步：定义常量

### 📖 这是什么？

配置 HyPE 的各种参数。

### 💻 完整代码

```python
# 下载所需的数据文件
import os
os.makedirs('data', exist_ok=True)

# 下载此笔记本中使用的 PDF 文档
# !wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
```

```python
# 文档路径
PATH = "data/Understanding_Climate_Change.pdf"

# 语言模型名称（用于生成假设性问题）
LANGUAGE_MODEL_NAME = "gpt-4o-mini"

# Embedding 模型名称（用于向量化）
EMBEDDING_MODEL_NAME = "text-embedding-3-small"

# 分块配置
CHUNK_SIZE = 1000      # 每个块的最小字符数
CHUNK_OVERLAP = 200    # 两个连续块之间的重叠字符数
```

> **💡 代码解释**
> - `PATH`：PDF 文档路径
> - `LANGUAGE_MODEL_NAME`：用于生成问题的 LLM
>   - `gpt-4o-mini`：性价比高，速度快
> - `EMBEDDING_MODEL_NAME`：用于生成向量的模型
>   - `text-embedding-3-small`：OpenAI 的小尺寸 embedding 模型
> - `CHUNK_SIZE`：块大小
>   - HyPE 可以容忍更大的块（因为问题会捕捉关键信息）
> - `CHUNK_OVERLAP`：重叠大小
>   - 避免重要信息被切到两块之间
>
> **⚠️ 新手注意**
> - 模型名称要参考 [OpenAI 官方文档](https://platform.openai.com/docs/models)
> - 不同模型的价格和速度不同

---

## 🤔 第三步：生成假设性提示嵌入

### 📖 这是什么？

这是 HyPE 的核心！为每个文档块生成多个"可能被问到的问题"，并把这些问题转成向量。

### 💻 完整代码

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def generate_hypothetical_prompt_embeddings(chunk_text: str):
    """
    使用 LLM 为单个块生成多个假设性问题。
    这些问题将在检索期间用作块的"代理"。

    参数：
    chunk_text (str): 块的文本内容

    返回：
    chunk_text (str): 块的文本内容（为了使多线程处理更容易）
    hypothetical prompt embeddings (List[float]): 从问题生成的嵌入向量列表
    """
    # 初始化 LLM（用于生成问题）
    llm = ChatOpenAI(temperature=0, model_name=LANGUAGE_MODEL_NAME)

    # 初始化 embedding 模型（用于向量化）
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

    # 创建生成问题的提示模板
    question_gen_prompt = PromptTemplate.from_template(
        "分析输入文本并生成关键问题，当回答这些问题时，"
        "捕捉文本的要点。每个问题应该是一行，"
        "没有编号或前缀。\n\n "
        "文本:\n{chunk_text}\n\n问题:\n"
    )

    # 创建问题生成链：提示词 → LLM → 字符串解析
    question_chain = question_gen_prompt | llm | StrOutputParser()

    # 从响应中解析问题
    # 注意:
    # - gpt4o 喜欢用 \n\n 分割问题，所以我们删除一个 \n
    # - 对于生产或如果使用 ollama 的较小模型，使用正则表达式解析是有益的
    # 例如 (无) 序列表
    # r"^\s*[\-\*\•]|\s*\d+\.\s*|\s*[a-zA-Z]\)\s*|\s*\(\d+\)\s*|\s*\([a-zA-Z]\)\s*|\s*\([ivxlcdm]+\)\s*"
    questions = question_chain.invoke({"chunk_text": chunk_text}).replace("\n\n", "\n").split("\n")

    # 为每个问题生成 embedding
    return chunk_text, embedding_model.embed_documents(questions)
```

> **💡 代码解释**
>
> **函数输入输出**：
> - 输入：`chunk_text`（文档块内容）
> - 输出：`(chunk_text, embeddings)`（原文本 + 问题向量列表）
>
> **处理流程**：
> 1. 初始化 LLM 和 embedding 模型
> 2. 创建提示词模板，告诉 LLM 生成关键问题
> 3. 调用 LLM 生成问题列表
> 4. 解析响应（按换行符分割）
> 5. 为每个问题生成向量
> 6. 返回原文本和向量列表
>
> **⚠️ 新手注意**
> - `temperature=0`：让输出更稳定
> - `StrOutputParser()`：把 LLM 的响应解析成字符串
> - `split("\n")`：按换行符分割成问题列表
> - 如果问题格式不整齐，可以用正则表达式解析（见代码注释）

### 示例：看看生成了什么问题

```python
# 测试一下
sample_chunk = "化石燃料燃烧产生的二氧化碳排放是气候变化的主要驱动因素。"
_, embeddings = generate_hypothetical_prompt_embeddings(sample_chunk)
print(f"生成了 {len(embeddings)} 个问题的向量")
```

> **💡 预期输出**
> ```
> 生成了 5 个问题的向量
> ```
>
> 这意味着 LLM 为这个块生成了 5 个假设性问题！

---

## 🏗️ 第四步：创建 FAISS 向量存储

### 📖 这是什么？

并行处理所有文档块，为每个块生成假设性问题向量，然后存入 FAISS 向量数据库。

### 💻 完整代码

```python
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List

def prepare_vector_store(chunks: List[str]):
    """
    从文本块列表创建并填充 FAISS 向量存储。

    此函数并行处理文本块列表，为每个块生成假设性提示嵌入。
    嵌入存储在 FAISS 索引中以进行高效的相似性搜索。

    参数：
    chunks (List[str]): 要嵌入和存储的文本块列表。

    返回：
    FAISS: 包含嵌入文本块的 FAISS 向量存储。
    """
    # 稍后初始化以查看向量长度
    vector_store = None

    # 创建线程池，并行处理所有块
    with ThreadPoolExecutor() as pool:
        # 提交所有任务到线程池
        futures = [pool.submit(generate_hypothetical_prompt_embeddings, c) for c in chunks]

        # 处理完成的任务
        for f in tqdm(as_completed(futures), total=len(chunks)):
            # 获取处理结果
            chunk, vectors = f.result()  # 检索处理过的块及其嵌入

            # 在第一个块上初始化 FAISS 向量存储
            if vector_store == None:
                vector_store = FAISS(
                    embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME),  # 定义 embedding 模型
                    index=faiss.IndexFlatL2(len(vectors[0])),  # 定义 L2 索引用于相似性搜索
                    docstore=InMemoryDocstore(),  # 使用内存文档存储
                    index_to_docstore_id={}  # 维护索引到文档映射
                )

            # 将块的每个生成嵌入向量与块内容配对
            # 每个块插入多次，每个提示向量一次
            chunks_with_embedding_vectors = [(chunk.page_content, vec) for vec in vectors]

            # 添加嵌入到存储
            vector_store.add_embeddings(chunks_with_embedding_vectors)

    return vector_store  # 返回填充的向量存储
```

> **💡 代码解释**
>
> **并行处理**：
> - `ThreadPoolExecutor`：创建线程池
> - `pool.submit()`：提交任务到线程池
> - `as_completed(futures)`：按完成顺序获取结果
> - `tqdm`：显示进度条
>
> **FAISS 初始化**：
> - `embedding_function`：用于生成向量的函数
> - `index=faiss.IndexFlatL2()`：L2 距离索引（欧几里得距离）
> - `docstore=InMemoryDocstore()`：内存文档存储
> - `index_to_docstore_id={}`：索引到文档 ID 的映射
>
> **关键技巧**：
> - 每个块会被插入**多次**（一次对应一个生成的问题）
> - 这样检索时，无论哪个问题匹配，都能找到原文本
>
> **⚠️ 新手注意**
> - `vector_store == None`：在第一个结果时才初始化
> - 需要知道向量维度（`len(vectors[0])`）才能初始化 FAISS
> - 大文档处理可能需要几分钟

### 为什么要并行处理？

```
串行处理（慢）：
块 1 ──→ 生成问题 ──→ 生成向量 ──→ 存储  (2 秒)
块 2 ──→ 生成问题 ──→ 生成向量 ──→ 存储  (2 秒)
块 3 ──→ 生成问题 ──→ 生成向量 ──→ 存储  (2 秒)
总计：6 秒

并行处理（快）：
块 1 ──→ 生成问题 ──→ 生成向量 ──→ 存储  ┐
块 2 ──→ 生成问题 ──→ 生成向量 ──→ 存储  ├─→ 同时处理 (2 秒)
块 3 ──→ 生成问题 ──→ 生成向量 ──→ 存储  ┘
总计：约 2-3 秒
```

---

## 📚 第五步：编码 PDF 到向量存储

### 📖 这是什么？

把 PDF 文件处理成块，然后调用前面的函数创建向量存储。

### 💻 完整代码

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from helper_functions import replace_t_with_space  # 假设这个函数存在

def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """
    使用 OpenAI embeddings 将 PDF 书籍编码到向量存储中。

    参数：
        path: PDF 文件的路径。
        chunk_size: 每个文本块的期望大小。
        chunk_overlap: 连续块之间的重叠量。

    返回：
        包含编码后书籍内容的 FAISS 向量存储。
    """
    # 步骤 1：加载 PDF 文档
    loader = PyPDFLoader(path)
    documents = loader.load()

    # 步骤 2：将文档分割成块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)

    # 步骤 3：清理文本（替换特殊字符）
    cleaned_texts = replace_t_with_space(texts)

    # 步骤 4：创建向量存储（包含假设性问题嵌入）
    vectorstore = prepare_vector_store(cleaned_texts)

    return vectorstore
```

> **💡 代码解释**
> - `PyPDFLoader`：读取 PDF 文件
> - `RecursiveCharacterTextSplitter`：递归字符文本分割器
> - `replace_t_with_space`：清理文本中的特殊字符（如 `\t`）
> - `prepare_vector_store`：前面定义的函数，生成假设性问题并存储
>
> **⚠️ 新手注意**
> - 如果 `replace_t_with_space` 不存在，可以用这个替代：
>   ```python
>   def replace_t_with_space(texts):
>       for text in texts:
>           text.page_content = text.page_content.replace('\t', ' ')
>       return texts
>   ```

### 创建 HyPE 向量存储

```python
# 使用 HyPE 时块大小可能相当大，因为我们不会因更多信息而损失精度。
# 对于生产环境，测试你的模型在生成每个块的足够问题方面有多详尽。
# 这主要取决于你的信息密度。
chunks_vector_store = encode_pdf(PATH, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
```

> **💡 代码解释**
> - 调用 `encode_pdf` 处理 PDF 并创建向量存储
> - 这会自动完成所有步骤：加载、分块、生成问题、创建向量、存储
>
> **⚠️ 新手注意**
> - 这个过程可能需要几分钟（取决于文档大小）
> - 进度条会显示处理进度

---

## 🔍 第六步：创建检索器

### 📖 这是什么？

从向量存储创建一个检索器，用于查询相关文档。

### 💻 完整代码

```python
# 创建检索器
# 根据查询相似性检索前 k=3 个最相关的块
chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 3})
```

> **💡 代码解释**
> - `as_retriever()`：把向量存储转换成检索器对象
> - `search_kwargs={"k": 3}`：每次检索返回 3 个最相关的文档
>
> **⚠️ 新手注意**
> - `k` 太小可能漏掉信息
> - `k` 太大可能包含噪音
> - 一般从 3-5 开始尝试

---

## 🧪 第七步：测试检索效果

### 📖 这是什么？

用示例查询测试检索器，看看效果如何。

### 💻 完整代码

```python
# 测试查询
test_query = "气候变化的主要原因是什么？"

# 检索上下文
context = retrieve_context_per_question(test_query, chunks_query_retriever)

# 去重（因为每个块有多个问题向量，可能重复检索到同一块）
context = list(set(context))

# 显示结果
show_context(context)
```

> **💡 代码解释**
> - `retrieve_context_per_question`：辅助函数，从检索器获取上下文
> - `list(set(context))`：去重，删除重复的块
> - `show_context`：辅助函数，美观地显示检索结果
>
> **⚠️ 新手注意**
> - 如果没有这些辅助函数，可以用：
>   ```python
>   results = chunks_query_retriever.invoke(test_query)
>   for i, doc in enumerate(results):
>       print(f"{i+1}) {doc.page_content[:200]}...")
>   ```

### 预期输出示例

```
检索到的相关文档:

1. 化石燃料燃烧产生的二氧化碳排放是气候变化的主要驱动因素。煤炭、石油和天然气的燃烧释放大量温室气体...

2. 森林砍伐导致碳汇减少，加剧了温室效应。树木通过光合作用吸收二氧化碳，减少森林意味着...

3. 工业活动释放了大量温室气体和空气污染物。制造业、采矿业和建筑业都贡献了显著的碳排放...
```

---

## 📊 第八步：评估结果

### 📖 这是什么？

用评估函数测试 RAG 系统的整体效果。

### 💻 完整代码

```python
evaluate_rag(chunks_query_retriever)
```

> **💡 代码解释**
> - `evaluate_rag`：辅助函数，评估 RAG 系统
> - 通常会：
>   - 用一组测试查询检索
>   - 计算准确率、召回率等指标
>   - 显示示例结果
>
> **⚠️ 新手注意**
> - 如果没有这个函数，可以手动测试：
>   ```python
>   test_queries = [
>       "气候变化的主要原因是什么？",
>       "温室气体有哪些？",
>       "森林砍伐如何影响气候？"
>   ]
>
>   for query in test_queries:
>       print(f"\n查询：{query}")
>       results = chunks_query_retriever.invoke(query)
>       for i, doc in enumerate(results, 1):
>           print(f"  {i}. {doc.page_content[:100]}...")
>   ```

---

## 📈 HyPE 的优势总结

### 根据论文评估结果

HyPE 在多个数据集上进行了评估，结果显示：

- **检索精度提高**：高达 **42 个百分点**
- **声明召回率提高**：高达 **45 个百分点**

（详见 [论文预印本](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5139335)）

### 核心优势

| 优势 | 说明 |
|------|------|
| ✅ **消除查询时开销** | 所有假设性生成在索引时离线完成，查询时不需要调用 LLM |
| ✅ **增强检索精度** | 查询与存储内容之间的对齐更好（问题匹配问题） |
| ✅ **可扩展且高效** | 无额外的每查询计算成本；检索速度与标准 RAG 相当 |
| ✅ **灵活且可扩展** | 可与重排序等高级 RAG 技术结合使用 |

---

## ❓ 常见问题 FAQ

### Q1：HyPE 和 HyDE 有什么区别？

**A**：

| 方面 | HyDE | HyPE |
|------|------|------|
| **生成时机** | 查询时生成 | 索引时预先生成 |
| **生成内容** | 假设性答案 | 假设性问题 |
| **查询延迟** | 高（要等 LLM 生成） | 低（预先计算好了） |
| **匹配方式** | 假设答案 → 文档 | 问题 → 问题 |

```
HyDE: 查询 ──→ [LLM 生成答案] ──→ 匹配文档
HyPE: 文档 ──→ [LLM 生成问题] ──→ 存储 ──→ 查询匹配问题
```

### Q2：每个块应该生成多少个问题？

**A**：
- 默认由 LLM 决定（通常 3-7 个）
- 可以在提示词中指定数量：
  ```python
  "生成 5 个关键问题..."
  ```
- 更多信息密集的块可以生成更多问题

### Q3：可以用本地模型吗？

**A**：当然可以！

```python
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

# 替换 OpenAI 模型
llm = Ollama(model="llama3.1:70b")
embedding_model = OllamaEmbeddings(model="nomic-embed-text")
```

### Q4：索引过程需要多长时间？

**A**：
- 取决于文档大小和模型速度
- 示例：100 页 PDF，用 GPT-4o-mini
  - 分块：约 200-300 个块
  - 每块 2-3 秒（生成问题 + 向量化）
  - 总计：约 10-15 分钟
- **但这是一次性的**，之后查询超快！

### Q5：内存占用大吗？

**A**：
- 每个块有多个向量（每个问题一个）
- 如果块太多，内存可能不够
- 解决方法：
  - 使用 FAISS 的磁盘索引
  - 增加 `chunk_size` 减少块数量
  - 限制每个块生成的问题数量

### Q6：如何与重排序（Rerank）结合？

**A**：

```python
# 1. 先用 HyPE 检索更多候选（如 k=10）
retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 10})
results = retriever.invoke(query)

# 2. 再用重排序模型精排
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker

compressor = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

# 3. 获取最终结果（如 top-3）
final_results = compression_retriever.invoke(query)
```

---

## 🎯 完整流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         HyPE 完整流程                                   │
└─────────────────────────────────────────────────────────────────────────┘

                        索引阶段（一次性）
═══════════════════════════════════════════════════════════════════════════

    PDF 文档
       │
       ▼
  ┌─────────┐
  │ 加载 PDF │
  └────┬────┘
       │
       ▼
  ┌─────────┐
  │ 分割成块 │ (chunk_size=1000, overlap=200)
  └────┬────┘
       │
       ▼
  ┌─────────────────────────────────────────┐
  │ 对每个块并行处理：                       │
  │  1. [LLM] 生成 3-7 个假设性问题          │
  │  2. [Embedding] 为每个问题生成向量      │
  │  3. [FAISS] 存储向量和原文本            │
  └─────────────────────────────────────────┘
       │
       ▼
  ┌─────────────┐
  │ 向量存储完成 │
  └─────────────┘

                        查询阶段（超快）
═══════════════════════════════════════════════════════════════════════════

    用户查询："气候变化的原因是什么？"
       │
       ▼
  ┌─────────────┐
  │ 查询向量化  │ (用同一个 Embedding 模型)
  └────┬────────┘
       │
       ▼
  ┌─────────────────────────────────────┐
  │ FAISS 向量检索                       │
  │ 查询向量 ↔ 预先存储的问题向量        │
  └────┬────────────────────────────────┘
       │
       ▼
  ┌─────────────┐
  │ 返回原文本块 │ (最匹配的 top-k)
  └─────────────┘
```

---

## 🎉 恭喜你学完了！

现在你已经掌握了：
1. ✅ HyPE 的核心概念和工作原理
2. ✅ HyPE 与 HyDE、传统 RAG 的区别
3. ✅ 完整的代码实现（从索引到查询）
4. ✅ 并行处理和性能优化技巧
5. ✅ 常见问题和解决方法

**下一步建议**：
- 用自己的文档测试 HyPE
- 调整 `chunk_size` 和 `k` 参数看效果
- 尝试结合重排序（Rerank）进一步提升精度
- 阅读 [原论文](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5139335) 了解更多细节

---

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--hype-hypothetical-prompt-embeddings)
