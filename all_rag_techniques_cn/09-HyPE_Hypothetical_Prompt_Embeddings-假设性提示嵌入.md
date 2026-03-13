# 假设性提示嵌入 (HyPE)

## 概述

本代码实现了一个由假设性提示嵌入 (HyPE) 增强的检索增强生成 (RAG) 系统。与难以处理查询 - 文档风格不匹配的传统 RAG 流程不同，HyPE 在索引阶段预先计算假设性问题。这将检索转变为问题 - 问题匹配问题，消除了对昂贵的运行时查询扩展技术的需求。

## Notebook 关键组件

1. PDF 处理和文本提取
2. 文本分块以保持连贯的信息单元
3. **假设性提示嵌入生成** - 使用 LLM 为每个块创建多个代理问题
4. 使用 [FAISS](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) 和 OpenAI embeddings 创建向量存储
5. 检索器设置用于查询处理后的文档
6. RAG 系统评估

## 方法详情

### 文档预处理

1. 使用 `PyPDFLoader` 加载 PDF。
2. 使用 `RecursiveCharacterTextSplitter` 将文本分割成具有指定块大小和重叠的块。

### 假设性问题生成

HyPE 不嵌入原始文本块，而是为每个块**生成多个假设性提示**。这些**预先计算的问题**模拟用户查询，改善与现实世界搜索的对齐。这消除了 HyDE 等技术中所需的运行时合成答案生成的需求。

### 向量存储创建

1. 使用 OpenAI embeddings 嵌入每个假设性问题。
2. 构建 FAISS 向量存储，将**每个问题嵌入与其原始块关联**。
3. 这种方法**为每个块存储多个表示**，增加检索灵活性。

### 检索器设置

1. 检索器针对**问题 - 问题匹配**而非直接文档检索进行优化。
2. FAISS 索引在假设性提示嵌入上实现**高效的最近邻**搜索。
3. 检索到的块为下游 LLM 生成提供**更丰富、更精确的上下文**。

## 关键特性

1. **预先计算的假设性提示** - 无运行时开销即可改善查询对齐。
2. **多向量表示** - 每个块被索引多次，以获得更广泛的语义覆盖。
3. **高效检索** - FAISS 确保在增强嵌入上的快速相似性搜索。
4. **模块化设计** - 流程易于适应不同的数据集和检索设置。此外，它与大多数优化（如重排序等）兼容。

## 评估

HyPE 的有效性在多个数据集上进行了评估，结果显示：

- 检索精度提高高达 42 个百分点
- 声明召回率提高高达 45 个百分点
    （详见 [预印本](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5139335) 中的完整评估结果）

## 此方法的优势

1. **消除查询时开销** - 所有假设性生成在索引时离线完成。
2. **增强检索精度** - 查询与存储内容之间的对齐更好。
3. **可扩展且高效** - 无额外的每查询计算成本；检索速度与标准 RAG 相当。
4. **灵活且可扩展** - 可与重排序等高级 RAG 技术结合使用。

## 结论

HyPE 为传统 RAG 系统提供了一种可扩展且高效的替代方案，克服了查询 - 文档风格不匹配问题，同时避免了运行时查询扩展的计算成本。通过将假设性提示生成移至索引阶段，它显著增强了检索精度和效率，使其成为实际应用的实用解决方案。

更多详情，请参阅完整论文：[预印本](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5139335)


<div style="text-align: center;">

<img src="../images/hype.svg" alt="HyPE" style="width:70%; height:auto;">
</div>

# 包安装与导入

下面的单元格安装运行此 notebook 所需的所有必要包。


```python
# 安装所需的包
!pip install faiss-cpu futures langchain-community python-dotenv tqdm
```

```python
# 克隆仓库以访问辅助函数和评估模块
!git clone https://github.com/NirDiamant/RAG_TECHNIQUES.git
import sys
sys.path.append('RAG_TECHNIQUES')
# 如果需要运行最新数据
# !cp -r RAG_TECHNIQUES/data .
```

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

# 设置 OpenAI API 密钥环境变量（如果不使用 OpenAI 请注释掉）
if not os.getenv('OPENAI_API_KEY'):
    os.environ["OPENAI_API_KEY"] = input("请输入您的 OpenAI API 密钥：")
else:
    os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# 原始路径追加已替换为 Colab 兼容版本
from helper_functions import *
from evaluation.evalute_rag import *
```

### 定义常量

- `PATH`: 数据路径，将被嵌入到 RAG 流程中

本教程使用 OpenAI 端点（[可用模型](https://platform.openai.com/docs/pricing)）。
- `LANGUAGE_MODEL_NAME`: 要使用的语言模型名称。
- `EMBEDDING_MODEL_NAME`: 要使用的 embedding 模型名称。

本教程使用 `RecursiveCharacterTextSplitter` 分块方法，其中分块长度函数使用 Python 的 `len` 函数。这里可调整的分块变量有：
- `CHUNK_SIZE`: 一个块的最小长度
- `CHUNK_OVERLAP`: 两个连续块之间的重叠

```python
# 下载所需的数据文件
import os
os.makedirs('data', exist_ok=True)

# 下载此笔记本中使用的 PDF 文档
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf

```

```python
PATH = "data/Understanding_Climate_Change.pdf"
LANGUAGE_MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
```

### 定义假设性提示嵌入生成

下面的代码块为每个文本块生成假设性问题并将其嵌入以供检索。

- LLM 从输入块中提取关键问题。
- 这些问题使用 OpenAI 的模型进行嵌入。
- 该函数返回原始块及其稍后用于检索的提示嵌入。

为确保输出整洁，会删除多余的换行符，必要时可使用正则表达式解析来改善列表格式。

```python
def generate_hypothetical_prompt_embeddings(chunk_text: str):
    """
    使用 LLM 为单个块生成多个假设性问题。
    这些问题将在检索期间用作块的"代理"。

    参数：
    chunk_text (str): 块的文本内容

    返回：
    chunk_text (str): 块的文本内容。这是为了使多线程处理更容易
    hypothetical prompt embeddings (List[float]): 从问题生成的嵌入向量列表
    """
    llm = ChatOpenAI(temperature=0, model_name=LANGUAGE_MODEL_NAME)
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

    question_gen_prompt = PromptTemplate.from_template(
        "分析输入文本并生成关键问题，当回答这些问题时，\
        捕捉文本的要点。每个问题应该是一行，\
        没有编号或前缀。\n\n \
        文本:\n{chunk_text}\n\n问题:\n"
    )
    question_chain = question_gen_prompt | llm | StrOutputParser()

    # 从响应中解析问题
    # 注意:
    # - gpt4o 喜欢用 \n\n 分割问题，所以我们删除一个 \n
    # - 对于生产或如果使用 ollama 的较小模型，使用正则表达式解析是有益的
    # 例如 (无) 序列表
    # r"^\s*[\-\*\•]|\s*\d+\.\s*|\s*[a-zA-Z]\)\s*|\s*\(\d+\)\s*|\s*\([a-zA-Z]\)\s*|\s*\([ivxlcdm]+\)\s*"
    questions = question_chain.invoke({"chunk_text": chunk_text}).replace("\n\n", "\n").split("\n")

    return chunk_text, embedding_model.embed_documents(questions)
```

### 定义 FAISS 向量存储的创建和填充

下面的代码块通过并行嵌入文本块来构建 FAISS 向量存储。

发生了什么？
- 并行处理 - 使用线程更快地生成嵌入。
- FAISS 初始化 - 设置 L2 索引以实现高效相似性搜索。
- 块嵌入 - 每个块存储多次，每个生成的问题嵌入一次。
- 内存存储 - 使用 InMemoryDocstore 进行快速查找。

这确保了高效检索，改善查询与预先计算的问题嵌入的对齐。

```python
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

    with ThreadPoolExecutor() as pool:
        # 使用线程加速提示嵌入生成
        futures = [pool.submit(generate_hypothetical_prompt_embeddings, c) for c in chunks]

        # 处理完成时的嵌入
        for f in tqdm(as_completed(futures), total=len(chunks)):

            chunk, vectors = f.result()  # 检索处理过的块及其嵌入

            # 在第一个块上初始化 FAISS 向量存储
            if vector_store == None:
                vector_store = FAISS(
                    embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME),  # 定义 embedding 模型
                    index=faiss.IndexFlatL2(len(vectors[0]))  # 定义 L2 索引用于相似性搜索
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

### 将 PDF 编码到 FAISS 向量存储中

下面的代码块处理 PDF 文件并将其内容存储为嵌入以供检索。

发生了什么？
- PDF 加载 - 从文档中提取文本。
- 分块 - 将文本分割成重叠的段以更好地保留上下文。
- 预处理 - 清理文本以提高嵌入质量。
- 向量存储创建 - 生成嵌入并将其存储在 FAISS 中以供检索。

```python
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

    # 加载 PDF 文档
    loader = PyPDFLoader(path)
    documents = loader.load()

    # 将文档分割成块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    vectorstore = prepare_vector_store(cleaned_texts)

    return vectorstore
```

### 创建 HyPE 向量存储

现在我们处理 PDF 并存储其嵌入。
此步骤使用编码的文档初始化 FAISS 向量存储。

```python
# 使用 HyPE 时块大小可能相当大，因为我们不会因更多信息而损失精度。
# 对于生产环境，测试你的模型在生成每个块的足够问题方面有多详尽。
# 这主要取决于你的信息密度。
chunks_vector_store = encode_pdf(PATH, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
```

### 创建检索器

现在我们设置检索器以从向量存储中获取相关块。

根据查询相似性检索前 `k=3` 个最相关的块。

```python
chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 3})
```

### 测试检索器

现在我们使用示例查询测试检索。

- 查询向量存储以找到最相关的块。
- 对结果进行去重以删除可能重复的块。
- 显示检索到的上下文以供检查。

此步骤验证检索器为给定问题返回有意义且多样化的信息。

```python
test_query = "气候变化的主要原因是什么？"
context = retrieve_context_per_question(test_query, chunks_query_retriever)
context = list(set(context))
show_context(context)
```

### 评估结果

```python
evaluate_rag(chunks_query_retriever)
```

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--hype-hypothetical-prompt-embeddings)
