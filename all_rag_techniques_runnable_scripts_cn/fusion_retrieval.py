# 导入必要的库和模块
import os  # 用于操作系统相关的操作，如读取环境变量
import sys  # 用于系统特定的参数和函数
from dotenv import load_dotenv  # 用于从.env 文件加载环境变量到系统环境中
from langchain_core.documents import Document  # LangChain 的文档类，用于封装文本内容
from typing import List  # 类型注解工具，用于指定列表类型
from rank_bm25 import BM25Okapi  # BM25 算法库，用于基于关键词的文本检索
import numpy as np  # NumPy 库，用于数值计算和数组操作

# 将父目录添加到 Python 路径
# 这样可以导入上级目录中的模块，如 helper_functions 等
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from helper_functions import *  # 导入自定义的辅助函数
from evaluation.evalute_rag import *  # 导入 RAG 评估相关的函数

# 加载环境变量
# .env 文件通常包含敏感信息如 API 密钥，不应该直接写在代码中
load_dotenv()
# 设置 OpenAI API 密钥环境变量，这样后续的 OpenAI 模型调用会自动使用这个密钥进行认证
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_KEY')


# 将 PDF 编码到向量存储并返回分割的文档的函数
def encode_pdf_and_get_split_documents(path, chunk_size=1000, chunk_overlap=200):
    """
    使用 OpenAI 嵌入将 PDF 书籍编码到向量存储中。

    这个函数完成以下工作：
    1. 读取 PDF 文件
    2. 将文本分割成适当大小的块
    3. 使用嵌入模型将每个文本块转换为向量
    4. 将向量存储到 FAISS 数据库中以便后续检索

    参数：
        path: PDF 文件的路径，如 "../data/book.pdf"
        chunk_size: 每个文本块的大小（字符数），默认 1000
            - 太大了可能包含过多不相关信息
            - 太小了可能丢失上下文
        chunk_overlap: 连续块之间的重叠量（字符数），默认 200
            - 重叠确保相邻块之间的上下文连续性
            - 避免关键信息被分割到两个块中

    返回：
        vectorstore: 包含编码后书籍内容的 FAISS 向量存储
        cleaned_texts: 清洗后的文本块列表
    """
    # 使用 PyPDFLoader 加载 PDF 文件
    # PyPDFLoader 是 LangChain 提供的 PDF 加载器，可以提取 PDF 中的文本
    loader = PyPDFLoader(path)
    documents = loader.load()  # 加载 PDF，返回文档对象列表

    # 创建递归字符文本分割器
    # 这种分割器会尝试在段落、句子等自然边界处分割文本
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    # 将文档分割成小块
    texts = text_splitter.split_documents(documents)

    # 清理文本：替换制表符等特殊字符为空格
    cleaned_texts = replace_t_with_space(texts)

    # 初始化 OpenAI 嵌入模型
    # 这个模型可以将文本转换为向量表示
    embeddings = OpenAIEmbeddings()

    # 使用 FAISS 创建向量存储
    # FAISS 是 Facebook 开发的高效相似度搜索库
    # from_documents 方法会自动计算每个文档的向量并建立索引
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore, cleaned_texts


# 为关键词检索创建 BM25 索引的函数
def create_bm25_index(documents: List[Document]) -> BM25Okapi:
    """
    从给定文档创建 BM25 索引。

    BM25 是什么？
    BM25（Best Matching 25）是一种经典的关键词检索算法，用于评估文档与查询的相关度。
    它基于词频统计：一个词在文档中出现越多，且在所有文档中越稀有，就越能代表这个文档。

    为什么要用 BM25？
    - 向量检索擅长捕捉语义相似性（如"汽车"和"车辆"）
    - BM25 擅长精确匹配关键词（如专有名词、技术术语）
    - 两者结合（融合检索）可以获得更好的检索效果

    参数：
        documents (List[Document]): 要索引的文档列表

    返回：
        BM25Okapi: 可用于 BM25 评分的索引对象
    """
    # 将每个文档的内容分割成单词（分词）
    # BM25 需要基于词语来计算统计信息
    tokenized_docs = [doc.page_content.split() for doc in documents]

    # 创建并返回 BM25 索引
    # BM25Okapi 会预先计算每个词的文档频率等信息，加速后续查询
    return BM25Okapi(tokenized_docs)


# 融合检索函数，结合基于关键词的（BM25）和基于向量的搜索
def fusion_retrieval(vectorstore, bm25, query: str, k: int = 5, alpha: float = 0.5) -> List[Document]:
    """
    执行融合检索，结合基于关键词的（BM25）和基于向量的搜索。

    什么是融合检索？
    融合检索（Fusion Retrieval）是一种将多种检索方法的结果组合起来的技术。
    这里我们结合了：
    1. 向量检索：基于语义相似度，能理解"汽车"和"车辆"是相似的概念
    2. BM25 检索：基于关键词匹配，擅长精确匹配专有名词和技术术语

    为什么要融合？
    - 单一检索方法有局限性
    - 向量检索可能忽略精确的关键词匹配
    - BM25 无法理解语义相似性
    - 融合两者可以互补优势，提高检索质量

    参数：
    vectorstore (VectorStore): 包含文档的向量存储，用于向量相似度搜索
    bm25 (BM25Okapi): 预先计算的 BM25 索引，用于关键词检索
    query (str): 查询字符串，用户的问题或搜索词
    k (int): 要检索的文档数量，默认返回 5 个最相关的文档
    alpha (float): 向量搜索分数的权重，范围 0-1
        - alpha=0.5 表示向量和 BM25 各占 50% 权重
        - alpha=0.8 表示向量检索占 80%，BM25 占 20%
        - alpha=0 表示完全使用 BM25

    返回：
    List[Document]: 基于组合分数的 top k 个文档
    """
    # 步骤 1：获取所有文档
    # 我们需要对所有文档计算分数，所以先获取全部文档
    # similarity_search("") 空查询会返回所有文档
    all_docs = vectorstore.similarity_search("", k=vectorstore.index.ntotal)

    # 步骤 2：计算 BM25 分数
    # 对查询进行分词，然后计算每个文档的 BM25 分数
    # get_scores 返回一个数组，每个元素是对应文档的 BM25 分数
    bm25_scores = bm25.get_scores(query.split())

    # 步骤 3：计算向量相似度分数
    # similarity_search_with_score 返回 (文档，距离) 元组
    # 距离越小表示越相似，我们需要转换为分数（越大越好）
    vector_results = vectorstore.similarity_search_with_score(query, k=len(all_docs))

    # 提取向量距离分数
    vector_scores = np.array([score for _, score in vector_results])

    # 步骤 4：归一化分数到 0-1 范围
    # 因为向量距离是越小越好，所以用 1- 归一化值 转换为分数
    # BM25 分数已经是越大越好，直接归一化即可
    vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores))
    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))

    # 步骤 5：计算加权组合分数
    # alpha * 向量分数 + (1-alpha) * BM25 分数
    combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores

    # 步骤 6：按分数排序，获取 top k 个文档的索引
    # argsort 返回排序后的索引，[::-1] 反转数组实现降序排列
    sorted_indices = np.argsort(combined_scores)[::-1]

    # 步骤 7：返回分数最高的 k 个文档
    return [all_docs[i] for i in sorted_indices[:k]]


class FusionRetrievalRAG:
    """
    融合检索 RAG 类：封装完整的融合检索流程

    这个类将文档处理、索引创建和查询检索整合在一起，
    提供了一个简单易用的接口来进行融合检索。

    使用流程：
    1. 初始化时传入 PDF 文件路径，自动完成文档处理和索引创建
    2. 调用 run 方法传入查询，即可获取检索结果
    """
    def __init__(self, path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        初始化 FusionRetrievalRAG 类，设置向量存储和 BM25 索引。

        参数：
        path (str): PDF 文件的路径，如 "../data/book.pdf"
        chunk_size (int): 每个文本块的大小（字符数），默认 1000
        chunk_overlap (int): 连续块之间的重叠（字符数），默认 200
        """
        # 调用前面定义的函数，处理 PDF 并创建向量存储
        self.vectorstore, self.cleaned_texts = encode_pdf_and_get_split_documents(path, chunk_size, chunk_overlap)
        # 基于处理后的文本创建 BM25 索引
        self.bm25 = create_bm25_index(self.cleaned_texts)

    def run(self, query: str, k: int = 5, alpha: float = 0.5):
        """
        执行给定查询的融合检索。

        参数：
        query (str): 搜索查询，用户的问题
        k (int): 要检索的文档数量，默认 5 个
        alpha (float): 向量搜索与 BM25 的权重，默认各占 50%

        返回：
        List[Document]: top k 个检索到的文档，并在控制台展示结果
        """
        # 调用融合检索函数获取 top k 个文档
        top_docs = fusion_retrieval(self.vectorstore, self.bm25, query, k, alpha)

        # 提取文档内容
        docs_content = [doc.page_content for doc in top_docs]

        # 使用辅助函数展示检索结果
        show_context(docs_content)


def parse_args():
    """
    解析命令行参数。

    这个函数允许用户通过命令行传递参数来运行脚本
    例如：python fusion_retrieval.py --path ./book.pdf --query "什么是气候变化"

    返回：
    args: 解析后的参数对象
    """
    import argparse  # Python 内置的命令行参数解析库
    parser = argparse.ArgumentParser(description="融合检索 RAG 脚本")

    # 定义各个命令行参数及其默认值和说明
    parser.add_argument('--path', type=str, default="../data/Understanding_Climate_Change.pdf",
                        help='PDF 文件的路径。')
    parser.add_argument('--chunk_size', type=int, default=1000, help='每个块的大小。')
    parser.add_argument('--chunk_overlap', type=int, default=200, help='连续块之间的重叠。')
    parser.add_argument('--query', type=str, default='气候变化对环境有什么影响？',
                        help='用于检索文档的查询。')
    parser.add_argument('--k', type=int, default=5, help='要检索的文档数量。')
    parser.add_argument('--alpha', type=float, default=0.5, help='向量搜索与 BM25 的权重。')

    return parser.parse_args()


if __name__ == "__main__":
    """
    程序主入口

    当直接运行这个脚本时（而不是作为模块导入），会执行这里的代码
    """
    # 解析命令行参数
    args = parse_args()

    # 创建融合检索 RAG 实例
    # 初始化时会自动处理 PDF 文件并创建索引
    retriever = FusionRetrievalRAG(path=args.path, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    # 执行查询并展示结果
    retriever.run(query=args.query, k=args.k, alpha=args.alpha)
