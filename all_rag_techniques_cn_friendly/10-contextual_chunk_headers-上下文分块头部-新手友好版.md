# 🌟 新手入门：上下文分块头部 (CCH)

> **💡 给新手的说明**
> - **难度等级**：⭐⭐⭐ 中级
> - **预计学习时间**：45-60 分钟
> - **前置知识**：了解 RAG 基本概念、会使用 Python
> - **本教程特色**：保留所有技术细节，增加通俗解释和代码注释

---

## 📖 核心概念理解

### 什么是上下文分块头部？

想象一下，你正在图书馆找一本书。如果每页纸都单独存放，你找到其中一页时，可能不知道它属于哪本书、哪一章。这就像 RAG 系统中的"分块"（chunk）——把文档切成小片段存储。

**上下文分块头部（Contextual Chunk Headers, CCH）** 就像是给每页纸贴上一个标签，写明：
- 📚 这本书叫什么名字（文档标题）
- 📑 属于哪一章（章节标题）
- 📝 这本书大概讲什么（文档摘要）

这样做的好处是：当你检索到某个片段时，能立刻知道它的"来历"，不会断章取义。

### 通俗理解

| 生活场景 | RAG 中的对应 |
|---------|------------|
| 读书时看到"他站了起来"，需要知道"他"是谁 | 分块中的代词需要上下文才能理解 |
| 论文中的"如图 3 所示"，需要看到图 3 | 分块中的引用需要原文档结构 |
| 会议记录中的"如上所述"，需要知道前面说了什么 | 分块失去前文会丢失信息 |

### 为什么要用 CCH？

在 RAG（检索增强生成）系统中，常见的问题是：**单个分块通常不包含足够的上下文**。这会导致两个问题：

1. **检索不到**：分块使用了隐式引用或代词，检索系统无法匹配
2. **理解错误**：LLM 拿到孤立分块，产生"幻觉"或错误理解

---

## 🛠️ 第一步：环境准备与包安装

### 📖 这是什么？

在开始实现 CCH 之前，我们需要安装必要的 Python 包。这就像做饭前要准备好锅碗瓢盆一样。

### 💻 完整代码

```python
# 安装所需的包
# langchain: 用于文本分割和文档处理
# openai: 调用 OpenAI API 生成文档标题
# python-dotenv: 管理环境变量（API 密钥等敏感信息）
# tiktoken: OpenAI 的 tokenizer，用于计算文本的 token 数量
!pip install langchain openai python-dotenv tiktoken
```

```python
# 导入所有需要的库
import cohere  # Cohere API，用于重排序（rerank）功能
import tiktoken  # OpenAI 的 tokenizer
from typing import List  # 类型提示，让代码更规范
from openai import OpenAI  # OpenAI 客户端
import os  # 操作系统接口，用于读取环境变量
from dotenv import load_dotenv  # 从.env 文件加载环境变量
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 文本分割器

# 从 .env 文件加载环境变量
# 这一步很重要！你的 API 密钥应该保存在 .env 文件中，而不是直接写在代码里
load_dotenv()

# 设置环境变量
os.environ["CO_API_KEY"] = os.getenv('CO_API_KEY')      # Cohere API 密钥
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')  # OpenAI API 密钥
```

> **💡 代码解释**
> - `load_dotenv()` 会读取当前目录下的 `.env` 文件，把里面的配置加载到环境变量
> - `os.getenv('CO_API_KEY')` 从环境变量中获取 API 密钥，这样更安全
>
> **⚠️ 新手注意**
> - 你需要先注册 Cohere 和 OpenAI 账号获取 API 密钥
> - 在项目根目录创建 `.env` 文件，内容格式：
>   ```
>   CO_API_KEY=你的 cohere 密钥
>   OPENAI_API_KEY=你的 openai 密钥
>   ```
> - **永远不要**把 API 密钥直接提交到 Git！

---

## 🛠️ 第二步：加载文档并分割成块

### 📖 这是什么？

RAG 系统的第一步是把大文档切成小块（分块）。想象你在整理一本厚书，把它拆成一页页的卡片，方便后续检索。

### 💻 完整代码

```python
# 下载所需的数据文件
import os
os.makedirs('data', exist_ok=True)  # 创建 data 目录，如果已存在则不报错

# 下载本笔记本使用的 PDF 文档
# 这些是示例文档，你可以替换成自己的文档
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
!wget -O data/nike_2023_annual_report.txt https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/nike_2023_annual_report.txt
```

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_into_chunks(text: str, chunk_size: int = 800) -> list[str]:
    """
    使用 RecursiveCharacterTextSplitter 将给定文本分割成指定大小的块。

    参数：
        text (str): 要分割成块的输入文本。
        chunk_size (int, optional): 每个块的最大大小。默认值为 800。

    返回：
        list[str]: 文本块列表。

    示例：
        >>> text = "This is a sample text to be split into chunks."
        >>> chunks = split_into_chunks(text, chunk_size=10)
        >>> print(chunks)
        ['This is a', 'sample', 'text to', 'be split', 'into', 'chunks.']
    """
    # 创建文本分割器
    # chunk_size=800: 每个块最多 800 个字符
    # chunk_overlap=0: 块与块之间不重叠（CCH 要求无重叠）
    # length_function=len: 使用字符数计算长度
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        length_function=len
    )

    # create_documents 会自动处理文本分割
    documents = text_splitter.create_documents([text])

    # 提取纯文本内容（去掉 Document 对象的包装）
    return [document.page_content for document in documents]

# 输入文档的文件路径
FILE_PATH = "data/nike_2023_annual_report.txt"

# 读取文档并将其分割成块
with open(FILE_PATH, "r") as file:
    document_text = file.read()

# 执行分割，每个块 800 字符
chunks = split_into_chunks(document_text, chunk_size=800)

# 打印前几个块看看效果
print(f"文档被分割成了 {len(chunks)} 个块")
print("第一个块的内容：")
print(chunks[0][:200])  # 只打印前 200 字符
```

> **💡 代码解释**
> - `RecursiveCharacterTextSplitter` 是 LangChain 的智能分割器，会尽量在段落、句子边界处切割
> - `chunk_overlap=0` 表示块与块之间不重叠，这是 CCH 的要求
> - 返回的是纯字符串列表，每个字符串是一个分块
>
> **⚠️ 新手注意**
> - `chunk_size` 设置要合理：太小会丢失上下文，太大检索精度下降
> - 实际项目中，800-2000 字符是常用范围
> - 如果文档很大，分割可能需要几秒到几分钟

---

## 🛠️ 第三步：生成描述性文档标题

### 📖 这是什么？

CCH 的核心思想是给每个分块添加"头部信息"。最简单也最重要的头部就是**文档标题**。如果文档本身没有好标题，我们可以用 LLM 自动生成一个。

### 通俗理解

这就像给每个文件命名：
- ❌ 不好的命名：`document1.txt`（没有信息量）
- ✅ 好的命名：`Nike_2023 年度报告_财务数据.pdf`（一看就知道内容）

### 💻 完整代码

```python
# 定义提示词模板
# 这个模板告诉 LLM 如何生成文档标题
DOCUMENT_TITLE_PROMPT = """
INSTRUCTIONS
以下文档的标题是什么？

你的响应必须是文档的标题，没有其他内容。不要响应其他任何内容。

{document_title_guidance}

{truncation_message}

DOCUMENT
{document_text}
""".strip()

# 如果文档被截断，添加这个提示
TRUNCATION_MESSAGE = """
另请注意，下面提供的文档文本只是文档的前~{num_words} 个单词。
这对于此任务应该足够了。你的响应仍应涉及整个文档，而不仅仅是下面提供的文本。
""".strip()

# 常量定义
MAX_CONTENT_TOKENS = 4000  # 最多处理 4000 个 token
MODEL_NAME = "gpt-4o-mini"  # 使用的模型
TOKEN_ENCODER = tiktoken.encoding_for_model('gpt-3.5-turbo')  # token 编码器

def make_llm_call(chat_messages: list[dict]) -> str:
    """
    调用 OpenAI 语言模型 API。

    参数：
        chat_messages (list[dict]): 聊天完成的消息字典列表。

    返回：
        str: 语言模型生成的响应。
    """
    # 创建 OpenAI 客户端
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 调用 API
    response = client.chat.completions.create(
        model=MODEL_NAME,           # 使用 gpt-4o-mini 模型
        messages=chat_messages,     # 对话消息
        max_tokens=MAX_CONTENT_TOKENS,  # 最大输出长度
        temperature=0.2,            # 温度参数，越低输出越稳定
    )

    # 提取并清理响应文本
    return response.choices[0].message.content.strip()

def truncate_content(content: str, max_tokens: int) -> tuple[str, int]:
    """
    将内容截断到指定的最大令牌数。

    参数：
        content (str): 要截断的输入文本。
        max_tokens (int): 保留的最大令牌数。

    返回：
        tuple[str, int]: 包含截断内容和令牌数的元组。
    """
    # 将文本编码成 tokens
    tokens = TOKEN_ENCODER.encode(content, disallowed_special=())

    # 只保留前 max_tokens 个
    truncated_tokens = tokens[:max_tokens]

    # 解码回文本，并返回 token 数量
    return TOKEN_ENCODER.decode(truncated_tokens), min(len(tokens), max_tokens)

def get_document_title(document_text: str, document_title_guidance: str = "") -> str:
    """
    使用语言模型提取文档标题。

    参数：
        document_text (str): 文档文本。
        document_title_guidance (str, optional): 标题提取的额外指导。默认值为 ""。

    返回：
        str: 提取的文档标题。
    """
    # 如果内容太长则截断
    document_text, num_tokens = truncate_content(document_text, MAX_CONTENT_TOKENS)

    # 如果截断了，添加提示告诉 LLM
    truncation_message = TRUNCATION_MESSAGE.format(num_words=3000) if num_tokens >= MAX_CONTENT_TOKENS else ""

    # 准备标题提取的提示
    prompt = DOCUMENT_TITLE_PROMPT.format(
        document_title_guidance=document_title_guidance,  # 可选的额外指导
        document_text=document_text,                       # 文档内容
        truncation_message=truncation_message              # 截断提示
    )

    # 构造消息列表（OpenAI API 格式）
    chat_messages = [{"role": "user", "content": prompt}]

    # 调用 LLM 并返回结果
    return make_llm_call(chat_messages)

# 示例用法
if __name__ == "__main__":
    # 假设 document_text 在其他地方定义
    document_title = get_document_title(document_text)
    print(f"文档标题：{document_title}")
```

> **💡 代码解释**
> - `DOCUMENT_TITLE_PROMPT` 是一个模板，用 `{}` 占位符填充不同内容
> - `truncate_content` 确保不会超过 LLM 的输入限制（4000 tokens）
> - `temperature=0.2` 让输出更稳定、更可预测（适合提取任务）
>
> **⚠️ 新手注意**
> - 如果你的文档已有好标题，可以跳过这一步直接用
> - `temperature` 参数范围 0-2，越低越确定，越高越随机
> - 调用 LLM API 会消耗 token，注意查看用量

---

## 🛠️ 第四步：添加分块头部并验证效果

### 📖 这是什么？

现在我们有了文档标题，接下来把它添加到每个分块中。然后我们用 Cohere 的重排序器（Reranker）来验证：添加头部后，检索效果是否真的变好了？

### 通俗理解

就像两封求职信：
- ❌ 没有抬头的："我认为我适合这个职位..."（HR 不知道你在申请什么）
- ✅ 有抬头的："尊敬的 Google 招聘经理：我认为我适合软件工程师职位..."（清晰明确）

### 💻 完整代码

```python
def rerank_documents(query: str, chunks: List[str]) -> List[float]:
    """
    使用 Cohere Rerank API 重新排序搜索结果。

    参数：
        query (str): 搜索查询。
        chunks (List[str]): 要重新排序的文档块列表。

    返回：
        List[float]: 每个块的相似性分数列表，按原始顺序。
    """
    MODEL = "rerank-english-v3.0"  # Cohere 的重排序模型
    client = cohere.Client(api_key=os.environ["CO_API_KEY"])

    # 调用重排序 API
    reranked_results = client.rerank(model=MODEL, query=query, documents=chunks)
    results = reranked_results.results

    # 提取排序后的索引和相关性分数
    reranked_indices = [result.index for result in results]
    reranked_similarity_scores = [result.relevance_score for result in results]

    # 转换回原始文档的顺序
    # 因为 rerank 返回的是按相关性排序的，但我们需要保持原始顺序来比较
    similarity_scores = [0] * len(chunks)
    for i, index in enumerate(reranked_indices):
        similarity_scores[index] = reranked_similarity_scores[i]

    return similarity_scores

def compare_chunk_similarities(chunk_index: int, chunks: List[str], document_title: str, query: str) -> None:
    """
    比较有无上下文头部的块的相似性分数。

    参数：
        chunk_index (int): 要检查的块索引。
        chunks (List[str]): 所有文档块的列表。
        document_title (str): 文档标题。
        query (str): 用于比较的搜索查询。

    打印：
        块头部、块文本、查询，以及有无头部的相似性分数。
    """
    # 获取指定块
    chunk_text = chunks[chunk_index]

    # 准备两个版本：无头部 vs 有头部
    chunk_wo_header = chunk_text  # 无头部版本
    chunk_w_header = f"文档标题：{document_title}\n\n{chunk_text}"  # 有头部版本

    # 获取两者的相似性分数
    similarity_scores = rerank_documents(query, [chunk_wo_header, chunk_w_header])

    # 打印对比结果
    print(f"\n块头部:\n文档标题：{document_title}")
    print(f"\n块文本:\n{chunk_text}")
    print(f"\n查询：{query}")
    print(f"\n无上下文分块头部的相似性：{similarity_scores[0]:.4f}")
    print(f"有上下文分块头部的相似性：{similarity_scores[1]:.4f}")

# 执行对比测试
CHUNK_INDEX_TO_INSPECT = 86  # 要检查的块索引
QUERY = "Nike 气候变化影响"    # 搜索查询

compare_chunk_similarities(CHUNK_INDEX_TO_INSPECT, chunks, document_title, QUERY)
```

> **💡 代码解释**
> - `rerank_documents` 用 Cohere 的重排序器评估每个块与查询的相关性
> - `compare_chunk_similarities` 对比同一块有无头部的得分差异
> - 相似性分数范围通常在 0-1 之间，越高越相关
>
> **⚠️ 新手注意**
> - 第 86 号块是一个很好的演示案例，它谈论气候变化但没有明确提到 "Nike"
> - 无头部时相关性约 0.1，有头部后约 0.92，提升显著！
> - 你可以改 `CHUNK_INDEX_TO_INSPECT` 测试其他块

### 实际效果示例

运行上面的代码后，你会看到类似这样的输出：

```
块头部:
文档标题：Nike 2023 年度报告

块文本:
[关于气候变化影响的内容，但没有提到 Nike]

查询：Nike 气候变化影响

无上下文分块头部的相似性：0.1023
有上下文分块头部的相似性：0.9245
```

> **💡 关键洞察**
>
> 这个块明显是关于气候变化对某个组织的影响，但它没有明确提到"Nike"。
> - 无头部时：系统不知道这是 Nike 的信息，相关性仅 0.1
> - 有头部后：系统知道这是 Nike 的报告，相关性飙升到 0.92

---

## 🛠️ 第五步：在分块头部中添加更多上下文

### 📖 除了文档标题，还能加什么？

文档标题是最简单也最重要的头部，但你还可以添加：

| 头部类型 | 说明 | 适用场景 |
|---------|------|---------|
| 简洁的文档摘要 | 用一两句话概括文档内容 | 帮助理解整体主题 |
| 章节/子章节标题 | 如 "第三章 财务数据 > 第二节 收入分析" | 处理涉及章节结构的查询 |
| 作者/日期信息 | 文档的作者、发布时间 | 需要时效性或权威性判断时 |

### 💻 代码示例

```python
# 添加更丰富头部的示例
def create_rich_chunk_header(document_title: str, chapter_title: str = "", doc_summary: str = "") -> str:
    """
    创建丰富的分块头部。

    参数：
        document_title: 文档标题
        chapter_title: 章节标题（可选）
        doc_summary: 文档摘要（可选）

    返回：
        完整的头部字符串
    """
    header_parts = []

    # 添加文档标题（必须有）
    header_parts.append(f"📚 文档：{document_title}")

    # 添加章节标题（如果有）
    if chapter_title:
        header_parts.append(f"📑 章节：{chapter_title}")

    # 添加文档摘要（如果有）
    if doc_summary:
        header_parts.append(f"📝 摘要：{doc_summary}")

    return "\n".join(header_parts)

# 使用示例
rich_header = create_rich_chunk_header(
    document_title="Nike 2023 年度报告",
    chapter_title="第二部分 可持续发展",
    doc_summary="本报告涵盖 Nike 的财务业绩、环境影响和社会责任倡议"
)

print(rich_header)
```

输出：
```
📚 文档：Nike 2023 年度报告
📑 章节：第二部分 可持续发展
📝 摘要：本报告涵盖 Nike 的财务业绩、环境影响和社会责任倡议
```

---

## 📊 评估结果

### KITE 基准测试

我们在一个名为 KITE（知识密集型任务评估）的端到端 RAG 基准上评估了 CCH。

#### 数据集介绍

| 数据集 | 内容 | 规模 |
|-------|------|------|
| AI Papers | 关于 AI 和 RAG 的学术论文 | ~100 篇 PDF |
| BVP Cloud 10-Ks | 云公司的年度财务报告 | ~70 家公司 PDF |
| Sourcegraph Handbook | 公司内部手册 | ~800 个 markdown 文件 |
| Supreme Court Opinions | 最高法院判决意见 | 2022 年任期全部 |

#### 测试结果

| 数据集 | 无 CCH | 有 CCH | 提升幅度 |
|-------|-------|-------|---------|
| AI Papers | 4.5 | 4.7 | +4.4% |
| BVP Cloud | 2.6 | 6.3 | +142% |
| Sourcegraph | 5.7 | 5.8 | +1.8% |
| Supreme Court | 6.1 | 7.4 | +21% |
| **平均分** | **4.72** | **6.04** | **+27.9%** |

> **💡 结果解读**
> - CCH 在所有四个数据集上都带来了性能提升
> - BVP Cloud（财务报告）提升最大，因为这类文档特别依赖上下文理解
> - 总体平均分提升 27.9%，效果显著

#### FinanceBench 测试

在 FinanceBench 基准上：
- 基准得分：19%
- CCH + RSE 组合：83%

> **⚠️ 注意**：这个测试同时使用了 CCH 和相关片段提取（RSE），所以无法单独量化 CCH 的贡献，但组合效果非常显著。

---

## ❓ 常见问题 FAQ

### Q1: CCH 会增加多少 token 消耗？
**A**: 取决于头部内容的丰富程度。仅添加文档标题通常增加 10-50 tokens，添加章节和摘要可能增加 100-300 tokens。考虑到检索质量的提升，这个成本通常是值得的。

### Q2: 我的文档已经有标题了，还需要生成吗？
**A**: 如果文档标题已经很清晰（如 "Nike_2023 年度报告.pdf"），可以直接使用，无需调用 LLM 生成。

### Q3: CCH 适用于所有类型的文档吗？
**A**: CCH 对结构化文档（报告、论文、手册）效果最好。对于短文档（如单页文章），收益可能不明显。

### Q4: 可以在检索时不加头部，只在生成时加吗？
**A**: 可以，但最佳实践是**检索和生成时都使用头部**。因为检索时的向量表示也会受头部影响，能检索到更相关的块。

---

## 🔑 关键要点总结

1. **核心思想**：给每个分块添加头部信息（文档标题、章节等），提供更完整的上下文
2. **实现简单**：只需在分块文本前拼接头部字符串
3. **效果显著**：在多个基准测试中平均提升 27.9%
4. **成本低**：每个块只增加少量 token
5. **适用广泛**：特别适合结构化文档和需要上下文理解的场景

---

## 📚 进阶学习建议

1. **实践**：用你自己的文档跑一遍这个流程
2. **扩展**：尝试添加更多类型的头部（摘要、关键词等）
3. **组合**：与相关片段提取（RSE）技术结合使用
4. **调优**：根据你的具体场景调整头部内容的详细程度

> **💪 动手练习**：找一个你熟悉的文档，手动添加头部后，感受一下理解的差异！

---

*本教程保持与原文档一致的技术深度，同时增加了通俗解释和实用指导。如需进一步了解，建议阅读原始论文和文档。*
