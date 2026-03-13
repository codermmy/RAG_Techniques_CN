# 导入必要的库和模块
import sys  # 用于系统特定的参数和函数
import os  # 用于操作系统相关的操作，如读取环境变量
import re  # 正则表达式库，用于文本处理
from langchain_core.documents import Document  # LangChain 的文档类，用于封装文本内容
from langchain_community.vectorstores import FAISS  # Facebook 的高效相似度搜索库
from enum import Enum  # 枚举类，用于定义常量
from langchain_openai import OpenAIEmbeddings  # OpenAI 的文本嵌入模型
from langchain_openai import ChatOpenAI  # OpenAI 的聊天模型
from typing import Any, Dict, List, Tuple  # 类型注解工具
from pydantic import BaseModel, Field  # 用于定义结构化输出的数据模型
import argparse  # Python 内置的命令行参数解析库

# 从 .env 文件加载环境变量
from dotenv import load_dotenv

load_dotenv()  # 加载.env 文件中的环境变量

# 设置 OpenAI API 密钥环境变量
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# 将父目录添加到 Python 路径，这样可以导入上级目录中的模块
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from helper_functions import *  # 导入自定义的辅助函数


class QuestionGeneration(Enum):
    """
    枚举类，用于指定文档处理的问题生成级别。

    什么是问题生成增强（Question Generation Augmentation）？
    - 为文档自动生成相关问题，然后将这些问题也加入索引
    - 当用户提问时，更容易匹配到相关的文档

    两种生成级别：
    1. DOCUMENT_LEVEL（文档级别）：为整个文档块生成问题
       - 优点：问题覆盖更广，能捕捉文档整体主题
       - 缺点：可能丢失细节信息
    2. FRAGMENT_LEVEL（片段级别）：为每个小片段生成问题
       - 优点：问题更精细，能捕捉具体细节
       - 缺点：生成的问题数量多，索引更大
    """
    DOCUMENT_LEVEL = 1  # 文档级别：为整个文档生成问题
    FRAGMENT_LEVEL = 2  # 片段级别：为每个小片段生成问题


# 配置常量，控制文档处理和問題生成的参数
DOCUMENT_MAX_TOKENS = 4000  # 文档最大 token 数，超过这个长度会分割
DOCUMENT_OVERLAP_TOKENS = 100  # 文档之间的重叠 token 数，确保上下文连续性
FRAGMENT_MAX_TOKENS = 128  # 片段最大 token 数，用于片段级别的问题生成
FRAGMENT_OVERLAP_TOKENS = 16  # 片段之间的重叠 token 数
QUESTION_GENERATION = QuestionGeneration.DOCUMENT_LEVEL  # 默认使用文档级别的问题生成
QUESTIONS_PER_DOCUMENT = 40  # 为每个文档/片段生成的问题数量


class QuestionList(BaseModel):
    """
    问题列表的数据模型

    用于存储为文档生成的问题列表
    继承自 BaseModel，确保输出格式的一致性
    """
    question_list: List[str] = Field(..., title="为文档或片段生成的问题列表")
    # question_list 是一个字符串列表，每个元素是一个生成的问题


class OpenAIEmbeddingsWrapper(OpenAIEmbeddings):
    """
    OpenAI 嵌入的包装类，提供与原始 OllamaEmbeddings 类似的接口。

    为什么要包装 OpenAIEmbeddings？
    - OpenAIEmbeddings 默认使用 embed_documents 和 embed_query 方法
    - 为了与其他嵌入模型接口保持一致，添加了__call__方法
    - 这样可以直接使用 embedding_model(query) 的方式调用
    """
    def __call__(self, query: str) -> List[float]:
        """
        使用嵌入模型将查询转换为向量

        参数：
            query: 要嵌入的查询字符串

        返回：
            浮点数列表，表示查询的向量表示
        """
        return self.embed_query(query)


def clean_and_filter_questions(questions: List[str]) -> List[str]:
    """
    清理和过滤问题。

    这个函数完成以下工作：
    1. 移除问题开头的数字编号（如"1. "、"2. "）
    2. 只保留以问号结尾的有效问题
    3. 去除多余的空格

    为什么要清理？
    - AI 生成的问题通常带有编号，需要移除以便后续处理
    - 有些生成的内容可能不是有效的问题，需要过滤掉

    参数：
        questions: AI 生成的原始问题列表

    返回：
        清理和过滤后的问题列表
    """
    cleaned_questions = []
    for question in questions:
        # 使用正则表达式移除开头的数字编号（如"1. "、"2. "）
        cleaned_question = re.sub(r'^\d+\.\s*', '', question.strip())
        # 只保留以问号结尾的有效问题
        if cleaned_question.endswith('?'):
            cleaned_questions.append(cleaned_question)
    return cleaned_questions


def generate_questions(text: str) -> List[str]:
    """
    基于给定的文本生成问题。

    这个函数使用 LLM 为文本生成相关问题，这些问题：
    - 可以基于文本内容直接回答
    - 覆盖文本的各个方面和细节
    - 用于增强索引，提高检索匹配率

    参数：
        text: 要为其生成问题的文本内容

    返回：
        生成的问题列表（已清理和去重）
    """
    # 初始化 LLM 模型
    # model="gpt-4o-mini" 使用较轻量级的 GPT-4o-mini 模型，适合批量任务
    # temperature=0 确保输出稳定一致
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 定义问题生成的提示模板
    prompt = PromptTemplate(
        input_variables=["context", "num_questions"],
        template="使用上下文数据：{context}\n\n生成至少 {num_questions} 个可以关于此上下文提出的问题。"
    )

    # 创建 LLM 链，指定输出格式为 QuestionList
    chain = prompt | llm.with_structured_output(QuestionList)

    # 准备输入数据
    input_data = {"context": text, "num_questions": QUESTIONS_PER_DOCUMENT}

    # 调用链生成问题
    result = chain.invoke(input_data)
    questions = result.question_list

    # 清理、过滤并去重后返回
    return list(set(clean_and_filter_questions(questions)))


def generate_answer(content: str, question: str) -> str:
    """
    基于给定的内容为问题生成答案。

    这个函数用于在检索到相关文档后，让 AI 基于文档内容生成准确的回答

    参数：
        content: 作为参考上下文的文本内容
        question: 要回答的问题

    返回：
        AI 生成的答案字符串
    """
    # 初始化 LLM 模型
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 定义问答提示模板
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="使用上下文数据：{context}\n\n为以下问题提供简洁准确的回答：{question}"
    )

    # 创建 LLM 链
    chain = prompt | llm

    # 准备输入数据
    input_data = {"context": content, "question": question}

    # 调用链生成答案
    return chain.invoke(input_data)


def split_document(document: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    将文档分割成块。

    这个函数使用基于词元的分割方法：
    1. 使用正则表达式提取所有单词（词元）
    2. 按指定的 chunk_size 和 chunk_overlap 分割成块
    3. 将每个块的词元重新连接成字符串

    为什么要这样分割？
    - 基于词元分割比基于字符分割更准确
    - 可以确保每个块有大致相同的词元数量
    - 重叠确保上下文连续性，避免关键信息被分割

    参数：
        document: 要分割的文档字符串
        chunk_size: 每个块的最大词元数
        chunk_overlap: 相邻块之间的重叠词元数

    返回：
        分割后的文本块列表
    """
    # 使用正则表达式提取所有单词（词元）
    # \b\w+\b 匹配单词边界之间的单词字符
    tokens = re.findall(r'\b\w+\b', document)

    chunks = []
    # 按步长（chunk_size - chunk_overlap）滑动窗口分割
    for i in range(0, len(tokens), chunk_size - chunk_overlap):
        # 获取当前窗口的词元
        chunk_tokens = tokens[i:i + chunk_size]
        chunks.append(chunk_tokens)
        # 如果已经到达文档末尾，退出循环
        if i + chunk_size >= len(tokens):
            break

    # 将每个块的词元列表连接成字符串
    return [" ".join(chunk) for chunk in chunks]


def print_document(comment: str, document: Any) -> None:
    """
    打印文档信息。

    这个辅助函数用于调试和展示处理过程中的文档信息

    参数：
        comment: 注释文本，说明打印文档的场景（如"数据集"、"检索到的相关片段"）
        document: 要打印的文档对象，包含 page_content 和 metadata
    """
    # 打印文档信息，包括类型、索引和内容
    print(f'{comment} (类型：{document.metadata["type"]}, 索引：{document.metadata["index"]}): {document.page_content}')


class DocumentProcessor:
    """
    文档处理器类：处理文档并创建增强的检索器

    这个类实现了文档增强（Document Augmentation）技术：
    1. 将文档分割成块
    2. 为每个块（或整个文档）生成相关问题
    3. 将原始文档和生成的问题一起存入索引
    4. 创建检索器，可以同时检索文档和问题

    为什么要生成问题？
    - 用户查询通常是问题形式
    - 将问题加入索引可以提高查询匹配率
    - 即使文档内容与查询措辞不同，但生成的问题可能匹配

    使用流程：
    1. 初始化时传入文档内容和嵌入模型
    2. 调用 run 方法处理文档并返回检索器
    """
    def __init__(self, content: str, embedding_model: OpenAIEmbeddings):
        """
        初始化文档处理器

        参数：
            content: 要处理的文档内容（字符串）
            embedding_model: 嵌入模型，用于将文本转换为向量
        """
        self.content = content  # 存储文档内容
        self.embedding_model = embedding_model  # 存储嵌入模型

    def run(self):
        """
        运行文档处理流程。

        这个方法是整个文档增强的核心，完成以下工作：
        1. 将文档分割成较大的块（DOCUMENT_MAX_TOKENS=4000）
        2. 将每个大块进一步分割成小片段（FRAGMENT_MAX_TOKENS=128）
        3. 根据配置为文档或片段生成问题
        4. 创建包含原始文档和生成问题的 FAISS 索引
        5. 返回检索器

        返回：
            可以用于检索的检索器对象
        """
        # 第一步：将文档分割成较大的块（4000 tokens）
        text_documents = split_document(self.content, DOCUMENT_MAX_TOKENS, DOCUMENT_OVERLAP_TOKENS)
        print(f'文本内容已分割为：{len(text_documents)} 个文档')

        documents = []  # 存储所有文档（原始文档 + 生成的问题）
        counter = 0  # 文档计数器，用于生成唯一索引

        # 第二步：遍历每个文档块
        for i, text_document in enumerate(text_documents):
            # 将每个文档块进一步分割成小片段（128 tokens）
            text_fragments = split_document(text_document, FRAGMENT_MAX_TOKENS, FRAGMENT_OVERLAP_TOKENS)
            print(f'文本文档 {i} - 已分割为：{len(text_fragments)} 个片段')

            # 第三步：处理每个片段
            for j, text_fragment in enumerate(text_fragments):
                # 创建原始文档对象
                documents.append(Document(
                    page_content=text_fragment,
                    metadata={"type": "ORIGINAL", "index": counter, "text": text_document}
                    # type: ORIGINAL 表示这是原始文档内容
                    # index: 唯一索引号
                    # text: 保存所属的完整文档块内容，用于后续答案生成
                ))
                counter += 1

                # 如果是片段级别的问题生成，为每个片段生成问题
                if QUESTION_GENERATION == QuestionGeneration.FRAGMENT_LEVEL:
                    questions = generate_questions(text_fragment)
                    # 为每个问题创建文档对象
                    documents.extend([
                        Document(page_content=question,
                                 metadata={"type": "AUGMENTED", "index": counter + idx, "text": text_document})
                        # type: AUGMENTED 表示这是增强（生成）的问题
                        # text: 同样保存所属的完整文档块内容
                        for idx, question in enumerate(questions)
                    ])
                    counter += len(questions)
                    print(f'文本文档 {i} 文本片段 {j} - 已生成：{len(questions)} 个问题')

            # 如果是文档级别的问题生成，为整个文档块生成问题
            if QUESTION_GENERATION == QuestionGeneration.DOCUMENT_LEVEL:
                questions = generate_questions(text_document)
                # 为每个问题创建文档对象
                documents.extend([
                    Document(page_content=question,
                             metadata={"type": "AUGMENTED", "index": counter + idx, "text": text_document})
                    for idx, question in enumerate(questions)
                ])
                counter += len(questions)
                print(f'文本文档 {i} - 已生成：{len(questions)} 个问题')

        # 第四步：打印所有创建的文档信息（用于调试）
        for document in documents:
            print_document("数据集", document)

        # 第五步：创建 FAISS 向量存储
        # from_documents 会自动计算每个文档的向量并建立索引
        print(f'创建存储，计算 {len(documents)} 个 FAISS 文档的嵌入')
        vectorstore = FAISS.from_documents(documents, self.embedding_model)

        # 第六步：创建并返回检索器
        # as_retriever 将向量存储转换为检索器接口
        # search_kwargs={"k": 1} 表示每次检索返回 1 个最相关的文档
        print("创建检索器，返回最相关的 FAISS 文档")
        return vectorstore.as_retriever(search_kwargs={"k": 1})


def parse_args():
    """
    解析命令行参数。

    允许用户通过命令行传递参数来运行脚本
    例如：python document_augmentation.py --path ./book.pdf

    返回：
        包含解析后参数的对象
    """
    parser = argparse.ArgumentParser(description="处理文档并创建检索器。")
    parser.add_argument('--path', type=str, default='../data/Understanding_Climate_Change.pdf',
                        help="要处理的 PDF 文档路径")
    return parser.parse_args()


if __name__ == "__main__":
    """
    程序主入口

    当直接运行这个脚本时（而不是作为模块导入），会执行这里的代码
    演示了完整的文档增强和检索流程
    """
    # 解析命令行参数
    args = parse_args()

    # 加载示例 PDF 文档到字符串变量
    # read_pdf_to_string 是 helper_functions 中定义的辅助函数
    content = read_pdf_to_string(args.path)

    # 实例化 OpenAI 嵌入模型，将由 FAISS 使用
    # 这个模型用于将文本转换为向量表示
    embedding_model = OpenAIEmbeddings()

    # 处理文档并创建检索器
    processor = DocumentProcessor(content, embedding_model)
    document_query_retriever = processor.run()

    # 检索器使用示例
    # 演示如何使用创建的检索器进行查询
    query = "什么是气候变化？"
    retrieved_docs = document_query_retriever.get_relevant_documents(query)
    print(f"\n查询：{query}")
    print(f"检索到的文档：{retrieved_docs[0].page_content}")

    # 进一步查询示例
    # 展示另一个查询示例
    query = "淡水生态系统如何因气候因素的变化而变化？"
    retrieved_documents = document_query_retriever.get_relevant_documents(query)
    for doc in retrieved_documents:
        print_document("检索到的相关片段", doc)

    # 从检索到的文档中提取上下文
    context = doc.metadata['text']

    # 使用 AI 生成答案
    # generate_answer 函数基于上下文为问题生成准确的回答
    answer = generate_answer(context, query)
    print(f'{os.linesep}答案:{os.linesep}{answer}')
