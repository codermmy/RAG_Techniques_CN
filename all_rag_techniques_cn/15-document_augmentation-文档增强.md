# 通过问题生成进行文档增强以改善检索

## 概述

此实现演示了一种文本增强技术，该技术利用额外的问题生成来改善向量数据库内的文档检索。通过生成和整合与每个文本片段相关的各种问题，系统增强了标准检索过程，从而提高了找到可作为生成式问答上下文的相关文档的可能性。

## 动机

通过用相关问题丰富文本片段，我们旨在显著提高识别包含用户查询答案的文档最相关部分的准确性。

## 前提条件

此方法使用 OpenAI 的语言模型和 embeddings。您需要一个 OpenAI API 密钥来使用此实现。确保您已安装所需的 Python 包：

```
pip install langchain openai faiss-cpu PyPDF2 pydantic
```

## 关键组件

1. **PDF 处理和文本分块**：处理 PDF 文档并将其分割成可管理的文本片段。
2. **问题增强**：使用 OpenAI 的语言模型在文档和片段级别生成相关问题。
3. **向量存储创建**：使用 OpenAI 的 embedding 模型计算文档的 embeddings 并创建 FAISS 向量存储。
4. **检索和答案生成**：使用 FAISS 查找最相关的文档并基于提供的上下文生成答案。

## 方法细节

### 文档预处理

1. 使用 LangChain 的 PyPDFLoader 将 PDF 转换为字符串。
2. 将文本分割成重叠的文本文档（text_document）用于构建上下文目的，然后将每个文档分割成重叠的文本片段（text_fragment）用于检索和语义搜索目的。

### 文档增强

1. 使用 OpenAI 的语言模型在文档或文本片段级别生成问题。
2. 使用 QUESTIONS_PER_DOCUMENT 常量配置要生成的问题数量。

### 向量存储创建

1. 使用 OpenAIEmbeddings 类计算文档 embeddings。
2. 从这些 embeddings 创建 FAISS 向量存储。

### 检索和生成

1. 基于给定查询从 FAISS 存储中检索最相关的文档。
2. 使用检索到的文档作为上下文，使用 OpenAI 的语言模型生成答案。

## 此方法的优势

1. **增强检索过程**：增加为给定查询找到最相关 FAISS 文档的概率。
2. **灵活的上下文调整**：允许轻松调整文本文档和片段的上下文窗口大小。
3. **高质量语言理解**：利用 OpenAI 强大的语言模型进行问题生成和答案生成。

## 实现细节

- `OpenAIEmbeddingsWrapper` 类为 embedding 生成提供一致的接口。
- `generate_questions` 函数使用 OpenAI 的聊天模型从文本创建相关问题。
- `process_documents` 函数处理文档分割、问题生成和向量存储创建的核心逻辑。
- 主执行演示了加载 PDF、处理其内容并执行示例查询。

## 结论

此技术提供了一种改善向量文档搜索系统中信息检索质量的方法。通过生成类似于用户查询的额外问题并利用 OpenAI 的先进语言模型，它潜在地在后续任务（如问答）中带来更好的理解和更准确的响应。

## 关于 API 使用的说明

请注意，此实现使用 OpenAI 的 API，这可能会根据使用情况产生费用。请确保监控您的 API 使用情况，并在您的 OpenAI 帐户设置中设置适当的限制。

# 包安装和导入

下面的单元格安装运行此笔记本所需的所有必要包。

```python
# 安装所需的包
!pip install faiss-cpu langchain langchain-openai python-dotenv
```

```python
# 克隆仓库以访问辅助函数和评估模块
!git clone https://github.com/NirDiamant/RAG_TECHNIQUES.git
import sys
sys.path.append('RAG_TECHNIQUES')
# 如果需要使用最新数据运行
# !cp -r RAG_TECHNIQUES/data .
```

```python
import sys
import os
import re
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from enum import Enum
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


# 原始路径追加已替换为 Colab 兼容性

from helper_functions import *


class QuestionGeneration(Enum):
    """
    枚举类，用于指定文档处理的问题生成级别。

    属性:
        DOCUMENT_LEVEL (int): 表示在整个文档级别生成问题。
        FRAGMENT_LEVEL (int): 表示在单个文本片段级别生成问题。
    """
    DOCUMENT_LEVEL = 1
    FRAGMENT_LEVEL = 2

# 根据模型不同，Mitral 7B 最大可达 8000，Llama 3.1 8B 可达 128k
DOCUMENT_MAX_TOKENS = 4000
DOCUMENT_OVERLAP_TOKENS = 100

# 在较短文本上计算 Embedding 和文本相似度
FRAGMENT_MAX_TOKENS = 128
FRAGMENT_OVERLAP_TOKENS = 16

# 在文档或片段级别生成问题
QUESTION_GENERATION = QuestionGeneration.DOCUMENT_LEVEL
# 为每个文档或片段生成多少个问题
QUESTIONS_PER_DOCUMENT = 40
```

### 定义此管道使用的类和函数

```python
class QuestionList(BaseModel):
    question_list: List[str] = Field(..., title="为文档或片段生成的问题列表")


class OpenAIEmbeddingsWrapper(OpenAIEmbeddings):
    """
    OpenAI embeddings 的包装器类，提供与原始 OllamaEmbeddings 类似的接口。
    """

    def __call__(self, query: str) -> List[float]:
        """
        允许实例作为可调用对象，为查询生成 embedding。

        Args:
            query (str): 要嵌入的查询字符串。

        Returns:
            List[float]: 查询的 embedding，作为浮点数列表。
        """
        return self.embed_query(query)

def clean_and_filter_questions(questions: List[str]) -> List[str]:
    """
    清理和过滤问题列表。

    Args:
        questions (List[str]): 要清理和过滤的问题列表。

    Returns:
        List[str]: 清理和过滤后的问题列表，以问号结尾。
    """
    cleaned_questions = []
    for question in questions:
        cleaned_question = re.sub(r'^\d+\.\s*', '', question.strip())
        if cleaned_question.endswith('?'):
            cleaned_questions.append(cleaned_question)
    return cleaned_questions

def generate_questions(text: str) -> List[str]:
    """
    使用 OpenAI 根据提供的文本生成问题列表。

    Args:
        text (str): 用于生成问题的上下文数据。

    Returns:
        List[str]: 唯一的、过滤后的问题列表。
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = PromptTemplate(
        input_variables=["context", "num_questions"],
        template="使用上下文数据：{context}\n\n生成至少{num_questions}个可以关于此上下文提出的可能问题列表。确保问题可以直接在上下文中回答，不包括任何答案或标题。用换行符分隔问题。"
    )
    chain = prompt | llm.with_structured_output(QuestionList)
    input_data = {"context": text, "num_questions": QUESTIONS_PER_DOCUMENT}
    result = chain.invoke(input_data)

    # 从 QuestionList 对象中提取问题列表
    questions = result.question_list

    filtered_questions = clean_and_filter_questions(questions)
    return list(set(filtered_questions))

def generate_answer(content: str, question: str) -> str:
    """
    使用 OpenAI 根据提供的上下文为给定问题生成答案。

    Args:
        content (str): 用于生成答案的上下文数据。
        question (str): 要生成答案的问题。

    Returns:
        str: 基于提供的上下文对问题的精确答案。
    """
    llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="使用上下文数据：{context}\n\n对以下问题提供简洁精确的答案：{question}"
    )
    chain =  prompt | llm
    input_data = {"context": content, "question": question}
    return chain.invoke(input_data)

def split_document(document: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    将文档分割成较小的文本块。

    Args:
        document (str): 要分割的文档文本。
        chunk_size (int): 每个块的大小，以令牌数量计。
        chunk_overlap (int): 连续块之间的重叠令牌数。

    Returns:
        List[str]: 文本块列表，每个块是文档内容的字符串。
    """
    tokens = re.findall(r'\b\w+\b', document)
    chunks = []
    for i in range(0, len(tokens), chunk_size - chunk_overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        chunks.append(chunk_tokens)
        if i + chunk_size >= len(tokens):
            break
    return [" ".join(chunk) for chunk in chunks]

def print_document(comment: str, document: Any) -> None:
    """
    打印注释后跟文档内容。

    Args:
        comment (str): 在文档详情之前打印的注释或描述。
        document (Any): 要打印内容的文档。

    Returns:
        None
    """
    print(f'{comment} (类型：{document.metadata["type"]}, 索引：{document.metadata["index"]}): {document.page_content}')
```

### 示例用法

```python
# 初始化 OpenAIEmbeddings
embeddings = OpenAIEmbeddingsWrapper()

# 示例文档
example_text = "This is an example document. It contains information about various topics."

# 生成问题
questions = generate_questions(example_text)
print("生成的问题:")
for q in questions:
    print(f"- {q}")

# 生成答案
sample_question = questions[0] if questions else "What is this document about?"
answer = generate_answer(example_text, sample_question)
print(f"\n问题：{sample_question}")
print(f"答案：{answer}")

# 分割文档
chunks = split_document(example_text, chunk_size=10, chunk_overlap=2)
print("\n文档块:")
for i, chunk in enumerate(chunks):
    print(f"块 {i + 1}: {chunk}")

# 使用 OpenAIEmbeddings 的示例
doc_embedding = embeddings.embed_documents([example_text])
query_embedding = embeddings.embed_query("What is the main topic?")
print("\n文档 Embedding (前 5 个元素):", doc_embedding[0][:5])
print("查询 Embedding (前 5 个元素):", query_embedding[:5])
```

### 主管道

```python
def process_documents(content: str, embedding_model: OpenAIEmbeddings):
    """
    处理文档内容，将其分割为片段，生成问题，
    创建 FAISS 向量存储，并返回检索器。

    Args:
        content (str): 要处理的文档内容。
        embedding_model (OpenAIEmbeddings): 用于向量化的 embedding 模型。

    Returns:
        VectorStoreRetriever: 用于检索最相关 FAISS 文档的检索器。
    """
    # 将整个文本内容分割为文本文档
    text_documents = split_document(content, DOCUMENT_MAX_TOKENS, DOCUMENT_OVERLAP_TOKENS)
    print(f'文本内容分割为：{len(text_documents)} 个文档')

    documents = []
    counter = 0
    for i, text_document in enumerate(text_documents):
        text_fragments = split_document(text_document, FRAGMENT_MAX_TOKENS, FRAGMENT_OVERLAP_TOKENS)
        print(f'文本文档 {i} - 分割为：{len(text_fragments)} 个片段')

        for j, text_fragment in enumerate(text_fragments):
            documents.append(Document(
                page_content=text_fragment,
                metadata={"type": "ORIGINAL", "index": counter, "text": text_document}
            ))
            counter += 1

            if QUESTION_GENERATION == QuestionGeneration.FRAGMENT_LEVEL:
                questions = generate_questions(text_fragment)
                documents.extend([
                    Document(page_content=question, metadata={"type": "AUGMENTED", "index": counter + idx, "text": text_document})
                    for idx, question in enumerate(questions)
                ])
                counter += len(questions)
                print(f'文本文档 {i} 文本片段 {j} - 生成：{len(questions)} 个问题')

        if QUESTION_GENERATION == QuestionGeneration.DOCUMENT_LEVEL:
            questions = generate_questions(text_document)
            documents.extend([
                Document(page_content=question, metadata={"type": "AUGMENTED", "index": counter + idx, "text": text_document})
                for idx, question in enumerate(questions)
            ])
            counter += len(questions)
            print(f'文本文档 {i} - 生成：{len(questions)} 个问题')

    for document in documents:
        print_document("数据集", document)

    print(f'创建存储，计算 {len(documents)} 个 FAISS 文档的 embeddings')
    vectorstore = FAISS.from_documents(documents, embedding_model)

    print("创建返回最相关 FAISS 文档的检索器")
    return vectorstore.as_retriever(search_kwargs={"k": 1})
```

### 示例

```python
# 下载所需的数据文件
import os
os.makedirs('data', exist_ok=True)

# 下载本笔记本使用的 PDF 文档
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
```

```python
# 将示例 PDF 文档加载到字符串变量
path = "data/Understanding_Climate_Change.pdf"
content = read_pdf_to_string(path)

# 实例化将被 FAISS 使用的 OpenAI Embeddings 类
embedding_model = OpenAIEmbeddings()

# 处理文档并创建检索器
document_query_retriever = process_documents(content, embedding_model)

# 检索器使用示例
query = "What is climate change?"
retrieved_docs = document_query_retriever.get_relevant_documents(query)
print(f"\nQuery: {query}")
print(f"Retrieved document: {retrieved_docs[0].page_content}")
```

### 在存储中查找最相关的 FAISS 文档。在大多数情况下，这将是一个增强的问题而不是原始文本文档。

```python
query = "How do freshwater ecosystems change due to alterations in climatic factors?"
print (f'Question:{os.linesep}{query}{os.linesep}')
retrieved_documents = document_query_retriever.invoke(query)

for doc in retrieved_documents:
    print_document("Relevant fragment retrieved", doc)
```

### 找到父文本文档并将其用作生成模型的上下文来生成问题的答案。

```python
context = doc.metadata['text']
print (f'{os.linesep}Context:{os.linesep}{context}')
answer = generate_answer(context, query)
print(f'{os.linesep}Answer:{os.linesep}{answer}')
```

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--document-augmentation)
