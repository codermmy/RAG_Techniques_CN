# 上下文分块头部 (CCH)

## 概述

上下文分块头部 (CCH) 是一种创建包含更高级别上下文（如文档级别或章节级别上下文）的分块头部的方法，并在嵌入之前将这些分块头部前置到分块中。这为嵌入提供了更准确和完整的文本内容和含义表示。在我们的测试中，此功能显著提高了检索质量。除了提高正确信息的检索率外，CCH 还降低了无关结果出现在搜索结果中的比率。这降低了 LLM 在下游聊天和生成应用中误解文本片段的比率。

## 动机

开发者在 RAG 中面临的许多问题归结为：单个分块通常不包含足够的上下文供检索系统或 LLM 正确使用。这导致无法回答问题，更令人担忧的是产生幻觉。

此问题的示例
- 分块通常通过隐式引用和代词来指代其主题。这导致它们在应该被检索时未被检索，或者未被 LLM 正确理解。
- 单个分块通常仅在整个章节或文档的上下文中有意义，单独阅读时可能会产生误导。

## 关键组件

#### 上下文分块头部
这里的思想是通过前置分块头部来为分块添加更高级别的上下文。此分块头部可以简单到只有文档标题，也可以使用文档标题、简洁的文档摘要以及章节和子章节标题的完整层次结构的组合。

## 方法详情

#### 上下文生成
在下面的演示中，我们使用 LLM 为文档生成描述性标题。这是通过一个简单的提示完成的，您传入文档文本的截断版本，并要求 LLM 为文档生成描述性标题。如果您已经有足够描述性的文档标题，则可以直接使用它们。我们发现文档标题是包含在分块头部中最简单且最重要的更高级别上下文类型。

您可以在分块头部中包含的其他类型的上下文：
- 简洁的文档摘要
- 章节/子章节标题
    - 这有助于检索系统处理对文档中较大章节或主题的查询。

#### 使用分块头部嵌入分块
您为每个分块嵌入的文本只是分块头部和分块文本的连接。如果您在检索期间使用重排序器，则需要确保在那里也使用相同的连接。

#### 将分块头部添加到搜索结果
在向 LLM 展示搜索结果时包含分块头部也是有益的，因为它为 LLM 提供了更多上下文，使其不太可能误解分块的含义。

![Your Technique Name](../images/contextual_chunk_headers.svg)

## 设置

运行此 notebook 需要 Cohere API 密钥和 OpenAI API 密钥。

# 包安装与导入

下面的单元格安装运行此 notebook 所需的所有必要包。


```python
# 安装所需的包
!pip install langchain openai python-dotenv tiktoken
```

```python
import cohere
import tiktoken
from typing import List
from openai import OpenAI
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 从 .env 文件加载环境变量
load_dotenv()
os.environ["CO_API_KEY"] = os.getenv('CO_API_KEY') # Cohere API key
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY') # OpenAI API key
```

## 加载文档并将其分割成块
我们将在此演示中使用基本的 LangChain RecursiveCharacterTextSplitter，但您可以将 CCH 与更复杂的分块方法结合使用以获得更好的性能。

```python
# 下载所需的数据文件
import os
os.makedirs('data', exist_ok=True)

# 下载此笔记本中使用的 PDF 文档
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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        length_function=len
    )
    documents = text_splitter.create_documents([text])
    return [document.page_content for document in documents]

# 输入文档的文件路径
FILE_PATH = "data/nike_2023_annual_report.txt"

# 读取文档并将其分割成块
with open(FILE_PATH, "r") as file:
    document_text = file.read()

chunks = split_into_chunks(document_text, chunk_size=800)
```

## 生成描述性文档标题以用于分块头部

```python
# 常量
DOCUMENT_TITLE_PROMPT = """
INSTRUCTIONS
以下文档的标题是什么？

你的响应必须是文档的标题，没有其他内容。不要响应其他任何内容。

{document_title_guidance}

{truncation_message}

DOCUMENT
{document_text}
""".strip()

TRUNCATION_MESSAGE = """
另请注意，下面提供的文档文本只是文档的前 ~{num_words} 个单词。这对于此任务应该足够了。你的响应仍应涉及整个文档，而不仅仅是下面提供的文本。
""".strip()

MAX_CONTENT_TOKENS = 4000
MODEL_NAME = "gpt-4o-mini"
TOKEN_ENCODER = tiktoken.encoding_for_model('gpt-3.5-turbo')

def make_llm_call(chat_messages: list[dict]) -> str:
    """
    调用 OpenAI 语言模型 API。

    参数：
        chat_messages (list[dict]): 聊天完成的消息字典列表。

    返回：
        str: 语言模型生成的响应。
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=chat_messages,
        max_tokens=MAX_CONTENT_TOKENS,
        temperature=0.2,
    )
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
    tokens = TOKEN_ENCODER.encode(content, disallowed_special=())
    truncated_tokens = tokens[:max_tokens]
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
    truncation_message = TRUNCATION_MESSAGE.format(num_words=3000) if num_tokens >= MAX_CONTENT_TOKENS else ""

    # 准备标题提取的提示
    prompt = DOCUMENT_TITLE_PROMPT.format(
        document_title_guidance=document_title_guidance,
        document_text=document_text,
        truncation_message=truncation_message
    )
    chat_messages = [{"role": "user", "content": prompt}]

    return make_llm_call(chat_messages)

# 示例用法
if __name__ == "__main__":
    # 假设 document_text 在其他地方定义
    document_title = get_document_title(document_text)
    print(f"文档标题：{document_title}")
```

## 添加分块头部并衡量影响
让我们看一个具体的例子来演示添加分块头部的影响。我们将使用 Cohere 重排序器来衡量有无分块头部时查询的相关性。

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
    MODEL = "rerank-english-v3.0"
    client = cohere.Client(api_key=os.environ["CO_API_KEY"])

    reranked_results = client.rerank(model=MODEL, query=query, documents=chunks)
    results = reranked_results.results
    reranked_indices = [result.index for result in results]
    reranked_similarity_scores = [result.relevance_score for result in results]

    # 转换回原始文档的顺序
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
    chunk_text = chunks[chunk_index]
    chunk_wo_header = chunk_text
    chunk_w_header = f"文档标题：{document_title}\n\n{chunk_text}"

    similarity_scores = rerank_documents(query, [chunk_wo_header, chunk_w_header])

    print(f"\n块头部:\n文档标题：{document_title}")
    print(f"\n块文本:\n{chunk_text}")
    print(f"\n查询：{query}")
    print(f"\n无上下文分块头部的相似性：{similarity_scores[0]:.4f}")
    print(f"有上下文分块头部的相似性：{similarity_scores[1]:.4f}")

# Notebook 单元格用于执行
# 假设 chunks 和 document_title 在前面的单元格中定义
CHUNK_INDEX_TO_INSPECT = 86
QUERY = "Nike 气候变化影响"

compare_chunk_similarities(CHUNK_INDEX_TO_INSPECT, chunks, document_title, QUERY)
```

这个块显然是关于气候变化对某个组织的影响，但它没有明确提到 "Nike"。因此，与查询 "Nike 气候变化影响" 的相关性仅约为 0.1。通过简单地将文档标题添加到块中，相似度上升到了 0.92。

# 评估结果

#### KITE

我们在一个名为 KITE（知识密集型任务评估）的端到端 RAG 基准上评估了 CCH。

KITE 目前包含 4 个数据集，共 50 个问题。
- **AI Papers** - 约 100 篇关于 AI 和 RAG 的学术论文，从 arXiv 以 PDF 格式下载。
- **BVP Cloud 10-Ks** - Bessemer Cloud Index 中所有公司（约 70 家）的 10-K，以 PDF 格式。
- **Sourcegraph Company Handbook** - 约 800 个 markdown 文件，保留原始目录结构，从 Sourcegraph 公开的公司手册 GitHub [页面](https://github.com/sourcegraph/handbook/tree/main/content) 下载。
- **Supreme Court Opinions** - 2022 年任期的所有最高法院意见（2023 年 1 月至 6 月发布），从官方最高法院 [网站](https://www.supremecourt.gov/opinions/slipopinion/22) 以 PDF 格式下载。

每个样本都包含标准答案。大多数样本还包含评分标准。评分以 0-10 分制对每个问题进行，由强大的 LLM 进行评分。

我们比较有 CCH 和无 CCH 的性能。对于 CCH 配置，我们使用文档标题和文档摘要。两种配置之间的所有其他参数保持不变。我们使用 Cohere 3 重排序器，并使用 GPT-4o 进行响应生成。

|                         | 无 CCH   | CCH          |
|-------------------------|----------|--------------|
| AI Papers               | 4.5      | 4.7          |
| BVP Cloud               | 2.6      | 6.3          |
| Sourcegraph             | 5.7      | 5.8          |
| Supreme Court Opinions  | 6.1      | 7.4          |
| **平均**                | 4.72     | 6.04         |

我们可以看到 CCH 在所有四个数据集上都带来了性能提升。一些数据集看到大幅提升，而其他数据集则看到小幅提升。总体平均分从 4.72 提高到 6.04，增长了 27.9%。

#### FinanceBench

我们还在 FinanceBench 上评估了 CCH，它贡献了 83% 的得分，而基准得分为 19%。对于该基准，我们联合测试了 CCH 和相关片段提取 (RSE)，因此我们无法确切说 CCH 对该结果贡献了多少。但 CCH 和 RSE 的组合显然在 FinanceBench 上带来了显著的准确性提升。

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--contextual-chunk-headers)
