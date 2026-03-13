# 🌟 新手入门：上下文窗口增强检索

> **💡 给新手的说明**
> - **难度等级**：⭐⭐⭐ 中级
> - **预计学习时间**：45-60 分钟
> - **前置知识**：了解向量搜索基础、FAISS 概念
> - **本教程特色**：保留所有技术细节，增加通俗解释和代码注释

---

## 📖 核心概念理解

### 什么是上下文窗口增强？

想象你在搜索引擎中搜索"苹果公司的创始人"：

**传统向量搜索**：
```
检索结果："...于 1976 年创立..."
问题：没有提到"谁"创立，也没有提到"苹果公司"
```

**上下文窗口增强检索**：
```
检索结果："史蒂夫·乔布斯和史蒂夫·沃兹尼亚克于 1976 年创立了苹果公司..."
效果：包含了前后文的完整信息
```

### 通俗理解

| 场景 | 传统检索 | 上下文增强 |
|-----|---------|-----------|
| 看电影 | 只看最精彩的 10 秒片段 | 看完整场景（含前后情节） |
| 读论文 | 只返回包含关键词的句子 | 返回完整段落 |
| 听演讲 | 只播放最相关的一句话 | 播放包含这句话的完整段落 |

### 核心思想

```
传统检索：[块 25] ← 孤立的一块

上下文增强：[块 24] + [块 25] + [块 26]
            前一块   相关块   后一块
            （提供更完整的上下文）
```

### 为什么要用上下文窗口？

传统向量搜索的问题：

```
┌─────────────────────────────────────┐
│  问题 1：语义完整但信息不完整       │
│  "他提出了这个理论" ← 谁是"他"？   │
├─────────────────────────────────────┤
│  问题 2：代词引用丢失               │
│  "该方法显著提升了性能" ← 什么方法？│
├─────────────────────────────────────┤
│  问题 3：跨块信息割裂               │
│  块 1："深度学习在 2012 年"         │
│  块 2："取得了突破性进展"           │
│  → 单独看都不完整                   │
└─────────────────────────────────────┘
```

上下文增强的解决方案：
- ✅ 为每个检索到的块添加前后邻居
- ✅ 智能处理重叠，避免重复
- ✅ 返回连贯的文本片段

---

## 🛠️ 第一步：环境准备与包安装

### 📖 这是什么？

准备运行上下文窗口增强所需的 Python 包和环境。

### 💻 完整代码

```python
# 安装所需的包
# langchain: 文档处理和向量存储
# python-dotenv: 管理环境变量
!pip install langchain python-dotenv
```

```python
# 克隆仓库以访问辅助函数和评估模块
# 这提供了项目中定义的通用工具函数
!git clone https://github.com/NirDiamant/RAG_TECHNIQUES.git

# 将仓库路径添加到 Python 路径
import sys
sys.path.append('RAG_TECHNIQUES')

# 如果需要最新数据，取消下面这行的注释
# !cp -r RAG_TECHNIQUES/data .
```

```python
import os
import sys
from dotenv import load_dotenv
from langchain.docstore.document import Document

# 导入辅助函数（来自克隆的仓库）
from helper_functions import *
from evaluation.evalute_rag import *

# 从 .env 文件加载环境变量
load_dotenv()

# 设置 OpenAI API 密钥环境变量
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
```

> **💡 代码解释**
> - `git clone`：克隆项目仓库，获取辅助函数
> - `sys.path.append()`：将仓库路径加入 Python 搜索路径
> - `helper_functions`：包含 `read_pdf_to_string` 等工具函数
> - `evaluation`：包含 RAG 评估函数
>
> **⚠️ 新手注意**
> - 你需要 OpenAI API 密钥来生成 embeddings
> - 辅助函数也可以自己实现，不一定要克隆仓库
> - 在生产环境中，建议将代码复制到本地项目

---

## 🛠️ 第二步：读取 PDF 文档

### 📖 这是什么？

加载 PDF 文档并将其转换为字符串，为后续分块做准备。

### 💻 完整代码

```python
# 下载所需的数据文件
import os
os.makedirs('data', exist_ok=True)  # 创建 data 目录

# 下载示例 PDF 文档
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
```

```python
# PDF 文件路径
path = "data/Understanding_Climate_Change.pdf"

# 使用辅助函数读取 PDF
# 这个函数封装了 PDF 解析的复杂性
content = read_pdf_to_string(path)

print(f"成功读取 PDF，总字符数：{len(content)}")
print(f"前 200 字符预览：\n{content[:200]}")
```

> **💡 代码解释**
> - `read_pdf_to_string`：辅助函数，处理 PDF 解析
> - 返回的是纯文本字符串，可直接用于分块
>
> **⚠️ 新手注意**
> - 如果 PDF 有密码保护，需要额外处理
> - 扫描版 PDF（图片格式）需要 OCR 才能提取文字
> - 复杂排版的 PDF 可能丢失格式信息

---

## 🛠️ 第三步：分割文本并保留索引

### 📖 这是什么？

将文本分割成块，**关键**是每个块都要记录它在原文中的位置索引。这样后面才能找到"邻居"块。

### 重要概念：为什么需要索引？

```
原文：[段落 1][段落 2][段落 3][段落 4][段落 5]...
            ↓ 分割后
块列表：[块 0][块 1][块 2][块 3][块 4]...
索引：     0    1    2    3    4

当检索到 块 2 时，我们知道：
- 前一个邻居：块 1（索引=2-1）
- 后一个邻居：块 3（索引=2+1）
```

### 💻 完整代码

```python
from typing import List

def split_text_to_chunks_with_indices(text: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    使用滑动窗口将给定文本分割成指定大小的块，每个块记录其索引。

    参数：
        text (str): 要分割成块的输入文本。
        chunk_size (int): 每个块的大小（字符数）。
        chunk_overlap (int): 块与块之间的重叠字符数。

    返回：
        List[Document]: Document 对象列表，每个包含 page_content 和 metadata。
                       metadata 中包含 'index'（块的序号）和 'text'（完整原文）。

    示例：
        >>> text = "ABCDEFGHIJ"
        >>> chunks = split_text_to_chunks_with_indices(text, chunk_size=4, chunk_overlap=2)
        >>> for c in chunks:
        ...     print(f"索引{c.metadata['index']}: {c.page_content}")
        索引 0: ABCD
        索引 1: CDEF
        索引 2: EF GH
        ...
    """
    chunks = []  # 存储结果块
    start = 0    # 起始位置

    # 滑动窗口分割
    while start < len(text):
        # 计算结束位置
        end = start + chunk_size

        # 提取块内容
        chunk = text[start:end]

        # 创建 Document 对象
        # page_content: 块的文本内容
        # metadata: 元数据，包含索引和原文
        chunks.append(Document(
            page_content=chunk,
            metadata={
                "index": len(chunks),  # 当前块的序号
                "text": text           # 完整原文（用于需要时）
            }
        ))

        # 移动起始位置（考虑重叠）
        start += chunk_size - chunk_overlap

    return chunks
```

> **💡 代码解释**
>
> **滑动窗口工作原理**：
> ```
> chunk_size = 400, chunk_overlap = 200
>
> 块 0: [0──────────────────400]
> 块 1:       [200──────────────────600]
> 块 2:             [400──────────────────800]
>                        ↓
>                   重叠 200 字符
> ```
>
> **为什么需要重叠？**
> - 避免在分割点丢失上下文
> - 确保相关概念不会被切成两半
> - 提高检索召回率
>
> **⚠️ 新手注意**
> - `chunk_overlap` 必须小于 `chunk_size`
> - 重叠越大，冗余越多，但上下文越完整
> - 推荐：重叠为 chunk_size 的 25%-50%

### 分割文档

```python
# 设置分块参数
chunks_size = 400      # 每块 400 字符
chunk_overlap = 200    # 重叠 200 字符（50%）

# 执行分割
docs = split_text_to_chunks_with_indices(content, chunks_size, chunk_overlap)

print(f"文档被分割成 {len(docs)} 个块")
print(f"第一个块的索引：{docs[0].metadata['index']}")
print(f"第一个块内容预览：\n{docs[0].page_content[:100]}")
```

---

## 🛠️ 第四步：创建向量存储

### 📖 这是什么？

使用 OpenAI 的 embeddings 将文本块转换为向量，然后用 FAISS 构建向量索引，支持快速相似性搜索。

### 概念解释

| 概念 | 说明 |
|-----|------|
| Embedding | 将文本转换为数值向量，语义相似的文本向量距离也近 |
| FAISS | Facebook 开发的高效向量搜索库 |
| VectorStore | 向量存储，管理文档向量和检索 |

### 💻 完整代码

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# 创建 Embeddings 对象
# 使用 OpenAI 的 text-embedding 模型
embeddings = OpenAIEmbeddings()

# 从文档块创建向量存储
# FAISS 会为每个块计算向量并构建索引
vectorstore = FAISS.from_documents(docs, embeddings)

# 创建检索器
# search_kwargs={"k": 1}: 默认返回最相关的 1 个结果
chunks_query_retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

print(f"向量存储创建完成，包含 {vectorstore.index.ntotal} 个向量")
```

> **💡 代码解释**
> - `OpenAIEmbeddings()`：使用 OpenAI API 生成向量表示
> - `FAISS.from_documents()`：批量处理文档并构建索引
> - `as_retriever()`：将向量存储转换为检索器接口
> - `k=1`：每次检索返回 1 个最相关的块（我们会手动添加邻居）
>
> **⚠️ 新手注意**
> - OpenAI Embeddings 需要网络和 API 密钥
> - FAISS 索引保存在内存中，重启会丢失
> - 大文档集建议使用持久化存储

---

## 🛠️ 第五步：按索引检索块的函数

### 📖 这是什么？

FAISS 支持按相似性搜索，但不直接支持"按索引检索"。我们需要自己实现这个功能，以便获取邻居块。

### 💻 完整代码

```python
def get_chunk_by_index(vectorstore, target_index: int):
    """
    根据块在其元数据中的索引从 vectorstore 检索块。

    参数：
        vectorstore (VectorStore): 包含块的向量存储。
        target_index (int): 要检索的块的索引。

    返回：
        Optional[Document]: 检索到的块作为 Document 对象，如果未找到则返回 None。

    注意：
        这是一个简化版本。在实际应用中，根据向量存储的实现，
        可能需要更有效的方法来按索引检索块（如建立倒排索引）。
    """
    # 获取所有文档（简化实现，效率不高）
    # 注意：这种方法在大量文档时效率很低
    all_docs = vectorstore.similarity_search("", k=vectorstore.index.ntotal)

    # 遍历查找匹配索引的块
    for doc in all_docs:
        if doc.metadata.get('index') == target_index:
            return doc

    # 没找到返回 None
    return None
```

> **💡 代码解释**
> - `similarity_search("", k=N)`：用空查询获取所有文档
> - `doc.metadata.get('index')`：获取我们之前存储的索引
> - 这是简化实现，生产环境应该用更高效的方法
>
> **⚠️ 新手注意 - 性能警告**
> - 这个函数每次都要获取所有文档，时间复杂度 O(n)
> - 块数量少时（<1000）没问题
> - 大量块时，建议预先建立索引映射：
>   ```python
>   # 预建索引
>   index_map = {doc.metadata['index']: doc for doc in all_docs}
>
>   # O(1) 查找
>   def get_chunk_by_index(target_index):
>       return index_map.get(target_index)
>   ```

### 测试检索函数

```python
# 测试：检索索引为 0 的块
chunk = get_chunk_by_index(vectorstore, 0)

if chunk:
    print(f"成功检索到块 0：")
    print(chunk.page_content[:200])  # 预览前 200 字符
else:
    print("未找到块")
```

---

## 🛠️ 第六步：核心函数 - 带上下文窗口的检索

### 📖 这是什么？

这是本教程的核心函数。它执行以下步骤：
1. 基于查询检索相关块
2. 为每个相关块获取前后邻居
3. 智能拼接（考虑重叠）
4. 返回扩展后的上下文

### 通俗理解

```
查询："气候变化的主要原因"

步骤 1：向量搜索找到最相关的块 25
        ↓
步骤 2：获取邻居块 24 和 26
        ↓
步骤 3：拼接 = 块 24 + 块 25 + 块 26（去除重叠）
        ↓
步骤 4：返回完整的上下文片段
```

### 💻 完整代码

```python
def retrieve_with_context_overlap(vectorstore, retriever, query: str,
                                   num_neighbors: int = 1,
                                   chunk_size: int = 200,
                                   chunk_overlap: int = 20) -> List[str]:
    """
    基于查询检索块，然后获取相邻块并连接它们，
    考虑重叠和正确的索引。

    参数：
        vectorstore (VectorStore): 包含块的向量存储。
        retriever: 用于获取相关文档的检索器对象。
        query (str): 搜索相关块的查询。
        num_neighbors (int): 在每个相关块之前和之后检索的块数。
                            例如 num_neighbors=2 会获取前后各 2 个块。
        chunk_size (int): 最初分割时每个块的大小。
        chunk_overlap (int): 最初分割时块之间的重叠。

    返回：
        List[str]: 连接块序列的列表，每个序列以相关块为中心。
                  如果有多个相关块，返回列表包含多个扩展片段。

    示例：
        假设检索到块 10，num_neighbors=1：
        返回 ["块 9 内容 + 块 10 内容 + 块 11 内容（去重后）"]
    """
    # 步骤 1：使用向量搜索检索相关块
    relevant_chunks = retriever.get_relevant_documents(query)

    result_sequences = []  # 存储结果

    # 步骤 2：为每个相关块构建上下文窗口
    for chunk in relevant_chunks:
        # 获取当前块的索引
        current_index = chunk.metadata.get('index')

        # 如果没有索引，跳过
        if current_index is None:
            continue

        # 步骤 3：确定要检索的块范围
        start_index = max(0, current_index - num_neighbors)  # 不能小于 0
        end_index = current_index + num_neighbors + 1  # +1 因为 range 是左闭右开

        # 步骤 4：检索范围内的所有块
        neighbor_chunks = []
        for i in range(start_index, end_index):
            neighbor_chunk = get_chunk_by_index(vectorstore, i)
            if neighbor_chunk:  # 确保块存在
                neighbor_chunks.append(neighbor_chunk)

        # 步骤 5：按索引排序块（确保顺序正确）
        neighbor_chunks.sort(key=lambda x: x.metadata.get('index', 0))

        # 步骤 6：连接块，考虑重叠
        # 从第一个块开始
        concatenated_text = neighbor_chunks[0].page_content

        # 依次添加后续块
        for i in range(1, len(neighbor_chunks)):
            current_chunk = neighbor_chunks[i].page_content

            # 计算重叠开始位置
            # 由于分块时有重叠，拼接时需要去除
            overlap_start = max(0, len(concatenated_text) - chunk_overlap)

            # 拼接：保留已拼接文本（去掉重叠部分）+ 当前块
            concatenated_text = concatenated_text[:overlap_start] + current_chunk

        # 将完整片段添加到结果
        result_sequences.append(concatenated_text)

    return result_sequences
```

> **💡 代码解释 - 重叠处理**
>
> ```
> 假设 chunk_size=400, chunk_overlap=200
>
> 块 24: [200──────────────────600]
> 块 25:       [400──────────────────800]
> 块 26:             [600──────────────────1000]
>
> 拼接过程：
> 1. 从块 24 开始：content[200:600]
> 2. 添加块 25：
>    - 重叠部分：400-600（200 字符）
>    - 拼接：content[200:400] + content[400:800]
> 3. 添加块 26：
>    - 重叠部分：600-800（200 字符）
>    - 拼接：已拼接 [:600] + content[600:1000]
>
> 最终：content[200:1000]（连续的 800 字符）
> ```
>
> **⚠️ 新手注意**
> - `num_neighbors` 控制窗口大小：
>   - 太小：上下文不足
>   - 太大：引入无关内容，增加 token 消耗
> - 推荐从 1-2 开始，根据效果调整

---

## 🛠️ 第七步：比较标准检索与上下文增强

### 💻 完整代码

```python
# 测试查询
query = "解释森林砍伐和化石燃料在气候变化中的作用。"

# 方法 1：标准检索（基线）
# k=1 表示只返回最相关的 1 个块
baseline_chunk = chunks_query_retriever.get_relevant_documents(query, k=1)

# 方法 2：上下文增强检索
# num_neighbors=1: 前后各加 1 个邻居
# chunk_size=400, chunk_overlap=200: 与分割参数一致
enriched_chunks = retrieve_with_context_overlap(
    vectorstore,
    chunks_query_retriever,
    query,
    num_neighbors=1,
    chunk_size=400,
    chunk_overlap=200
)

# 打印对比结果
print("=" * 60)
print("【基线检索结果】")
print("=" * 60)
print(baseline_chunk[0].page_content)

print("\n" + "=" * 60)
print("【上下文增强检索结果】")
print("=" * 60)
print(enriched_chunks[0])
```

> **💡 预期输出对比**
>
> **基线检索**（可能）：
> ```
> "...对气候产生深远影响。燃烧煤炭、石油和天然气
> 会释放二氧化碳等温室气体，这些气体会在大气中
> 形成温室效应..."
> ```
>
> **上下文增强**（可能）：
> ```
> "森林砍伐是气候变化的主要驱动因素之一...
> （前文：解释森林如何吸收 CO2）
>
> ...对气候产生深远影响。燃烧煤炭、石油和天然气
> 会释放二氧化碳等温室气体...
> （核心内容：化石燃料的影响）
>
> ...减少碳排放是应对气候变化的关键措施。"
> （后文：可能的解决方案）
> ```

---

## 🛠️ 第八步：完整示例 - 展示上下文窗口优势

### 💻 完整代码

```python
# 构造一个演示文档（AI 历史）
document_content = """
人工智能（AI）有着丰富的历史，可以追溯到 20 世纪中叶。"人工智能"一词于 1956 年在达特茅斯会议上被创造出来，标志着该领域的正式开始。

在 1950 年代和 1960 年代，AI 研究专注于符号方法和问题解决。Logic Theorist 由 Allen Newell 和 Herbert A. Simon 于 1955 年创建，通常被认为是第一个 AI 程序。

1960 年代见证了专家系统的发展，该系统使用预定义规则来解决复杂问题。DENDRAL 创建于 1965 年，是最早的专家系统之一，设计用于分析化学化合物。

然而，1970 年代带来了第一个"AI 冬季"，这是一个资金减少和 AI 研究兴趣降低的时期，主要是由于承诺过多而成果不足。

1980 年代见证了专家系统在公司中的普及而复兴。日本政府的第五代计算机项目也刺激了全球 AI 研究投资的增加。

神经网络在 1980 年代和 1990 年代崭露头角。反向传播算法虽然更早被发现，但在此期间被广泛用于训练多层网络。

1990 年代后期和 2000 年代标志着机器学习方法的兴起。支持向量机（SVM）和随机森林在各种分类和回归任务中变得流行。

深度学习是机器学习的一个子集，使用具有多层的神经网络，在 2010 年代初期开始显示出有希望的结果。突破出现在 2012 年，当时深度神经网络在 ImageNet 竞赛中显著优于其他机器学习方法。

从那时起，深度学习彻底改变了许多 AI 应用，包括图像和语音识别、自然语言处理和游戏。2016 年，Google 的 AlphaGo 击败了世界冠军围棋选手，这是 AI 的一个里程碑式成就。

当前的 AI 时代特征是将深度学习与其他 AI 技术集成，开发更高效和更强大的硬件，以及围绕 AI 部署的伦理考虑。

Transformers 于 2017 年推出，已成为自然语言处理中的主导架构，使 GPT（生成式预训练 Transformer）等模型能够生成类似人类的文本。

随着 AI 的不断发展，新的挑战 и 机遇也随之出现。可解释 AI、稳健和公平的机器学习以及人工通用智能（AGI）是该领域当前和未来研究的关键领域。
"""

# 分割演示文档
chunks_size = 250      # 较小的块，便于演示
chunk_overlap = 20     # 较小的重叠

document_chunks = split_text_to_chunks_with_indices(document_content, chunks_size, chunk_overlap)

# 创建向量存储
document_vectorstore = FAISS.from_documents(document_chunks, embeddings)

# 创建检索器
document_retriever = document_vectorstore.as_retriever(search_kwargs={"k": 1})

# 测试查询
query = "深度学习何时在 AI 中变得突出？"

# 方法 1：标准检索
context = document_retriever.get_relevant_documents(query)
context_pages_content = [doc.page_content for doc in context]

# 方法 2：上下文增强检索
sequences = retrieve_with_context_overlap(
    document_vectorstore,
    document_retriever,
    query,
    num_neighbors=1  # 前后各加 1 个块
)

print("=" * 60)
print("【查询】", query)
print("=" * 60)

print("\n【常规检索】")
print(context_pages_content[0])

print("\n" + "=" * 60)
print("【上下文增强检索】")
print(sequences[0])
```

### 预期输出分析

**常规检索**（可能只返回）：
```
"...深度学习是机器学习的一个子集，使用具有多层的
神经网络，在 2010 年代初期开始显示出有希望的结果..."
```

**上下文增强检索**（返回更完整）：
```
"...1990 年代后期和 2000 年代标志着机器学习方法的兴起。
支持向量机（SVM）和随机森林在各种分类和回归任务中变得流行。

深度学习是机器学习的一个子集，使用具有多层的神经网络，
在 2010 年代初期开始显示出有希望的结果。突破出现在 2012 年，
当时深度神经网络在 ImageNet 竞赛中显著优于其他机器学习方法。

从那时起，深度学习彻底改变了许多 AI 应用，包括图像和
语音识别、自然语言处理和游戏。2016 年，Google 的 AlphaGo
击败了世界冠军围棋选手..."
```

> **💡 优势分析**
>
> 上下文增强提供了：
> 1. **前因**：深度学习兴起前的背景（SVM、随机森林）
> 2. **核心答案**：2010 年代初期，2012 年突破
> 3. **后果**：深度学习的影响和里程碑事件

---

## 📊 可视化对比

### 传统检索 vs 上下文增强

```
传统向量搜索：
┌─────────────────────────────────────────────────────┐
│  查询："深度学习何时兴起？"                          │
│                                                     │
│  向量数据库                                         │
│  ┌─────┬─────┬─────┬─────┬─────┐                   │
│  │块 10│块 11│块 12│块 13│块 14│  ...             │
│  └─────┴─────┴─────┴─────┴─────┘                   │
│                    ↑                                │
│              只返回这一块                           │
│              （信息不完整）                         │
└─────────────────────────────────────────────────────┘

上下文窗口增强：
┌─────────────────────────────────────────────────────┐
│  查询："深度学习何时兴起？"                          │
│                                                     │
│  向量数据库                                         │
│  ┌─────┬─────┬─────┬─────┬─────┐                   │
│  │块 10│块 11│块 12│块 13│块 14│  ...             │
│  └─────┴─────┴─────┴─────┴─────┘                   │
│              ┌─────────────────┐                    │
│              ←    返回这个区域   →                  │
│              （包含上下文）                         │
└─────────────────────────────────────────────────────┘
```

---

## ❓ 常见问题 FAQ

### Q1: `num_neighbors` 应该设多少？
**A**: 取决于你的需求：
- 简单事实查询：1 就够了
- 复杂推理问题：2-3 可能更好
- 实验建议：从 1 开始，逐步增加看效果

### Q2: 上下文窗口会增加多少 token？
**A**: 粗略估算：
- 每个邻居块 ≈ chunk_size 个字符
- num_neighbors=1 时，增加约 2 倍内容（前后各 1 块）
- 例如：400 字符的块 → 约 1200 字符的输出

### Q3: 为什么不直接增大 chunk_size？
**A**: 有Trade-off：
- 大块：检索精度下降（向量表示不够聚焦）
- 小块 + 上下文：检索精准 + 上下文完整
- 两者结合是最好的！

### Q4: 可以用在向量检索之前吗？
**A**: 可以，但有更好的方法：
- 检索前扩展：增加查询文本，可能影响检索精度
- 检索后扩展（本方法）：精准定位后补充上下文
- 推荐：检索后扩展

### Q5: 与 CCH 技术有何不同？
**A**: 两者互补：
- **CCH**：在嵌入前添加头部（文档标题等）
- **上下文窗口**：检索后添加邻居块
- 最佳实践：两者结合使用！

---

## 🔑 关键要点总结

1. **核心思想**：检索相关块后，添加前后邻居提供更完整上下文
2. **关键参数**：`num_neighbors` 控制窗口大小
3. **重叠处理**：智能去除重复内容，保持文本连贯
4. **优势明显**：解决孤立块信息不完整的问题
5. **实现简单**：只需几十行代码即可实现
6. **适用场景**：特别适合需要上下文的问答任务

---

## 📚 进阶学习建议

### 实践练习
1. **基础**：用示例代码跑通整个流程
2. **进阶**：调整 `num_neighbors`，观察输出变化
3. **高级**：实现自适应邻居数量（根据查询复杂度）

### 扩展改进
```python
# 挑战 1：智能邻居选择
# - 不是固定 num_neighbors
# - 根据相关性分数动态决定

# 挑战 2：双向扩展
# - 只在相关方向扩展（如果后文更相关，只向后）

# 挑战 3：层级上下文
# - 先加 1 层邻居
# - 如果不够，再加第 2 层
```

### 与其他技术组合
- **CCH + 上下文窗口**：头部信息 + 邻居块
- **RSE + 上下文窗口**：相关片段 + 扩展上下文
- **全部组合**：最强配置！

> **💪 动手练习**：找一个实际问答场景，对比有无上下文窗口的回答质量，记录你的发现！

---

*本教程保持与原文档一致的技术深度，同时增加了通俗解释和实用指导。*
