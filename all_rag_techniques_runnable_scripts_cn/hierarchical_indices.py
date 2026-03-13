# ============================================================================
# 导入必要的库和模块
# ============================================================================
import asyncio
import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains.summarize.chain import load_summarize_chain
from langchain.docstore.document import Document
from helper_functions import encode_pdf, encode_from_string

# 将父目录添加到路径，因为我们使用 notebooks
# 这样可以导入上级目录中的 helper_functions 等模块
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from helper_functions import *
from evaluation.evalute_rag import *

# 从 .env 文件加载环境变量
# .env 文件包含敏感配置信息，如 API 密钥
load_dotenv()

# 设置 OpenAI API 密钥环境变量
# 后续使用 OpenAI 服务时会自动读取这个环境变量
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


# ============================================================================
# 分层 PDF 编码函数（异步版本）
# ============================================================================
# 这个函数的核心思想：
# 1. 为 PDF 的每一页生成一个"摘要"（简短概述）
# 2. 同时也将每一页分割成详细的"文本块"
# 3. 分别构建两个向量库：摘要库和详细库
# 4. 检索时先用摘要定位相关页面，再从这些页面找详细块
#
# 这种分层结构的优势：
# - 摘要库小，检索快，能快速定位相关区域
# - 详细库保留完整信息，保证答案质量
async def encode_pdf_hierarchical(path, chunk_size=1000, chunk_overlap=200, is_string=False):
    """
    异步将 PDF 书籍使用 OpenAI 嵌入编码到分层向量存储中

    分层向量存储包含两层：
    1. 摘要层：每页 PDF 的摘要，用于快速定位
    2. 详细层：每页的具体文本块，用于精确检索

    包含带有指数退避的速率限制处理（防止 API 调用过快被限流）

    参数：
        path: PDF 文件路径（或当 is_string=True 时是文本内容）
        chunk_size: 每个文本块的大小
        chunk_overlap: 块与块之间的重叠
        is_string: 如果为 True，path 是文本内容而不是文件路径

    返回：
        (summary_vectorstore, detailed_vectorstore): 两个向量存储的元组
    """
    # 根据输入类型加载文档
    if not is_string:
        # 从 PDF 文件加载
        # PyPDFLoader 读取 PDF 并返回文档对象
        loader = PyPDFLoader(path)
        # asyncio.to_thread 在后台线程中运行同步操作，避免阻塞异步事件循环
        documents = await asyncio.to_thread(loader.load)
    else:
        # 直接从文本字符串创建文档
        # 使用文本分割器将长文本分割成块
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, is_separator_regex=False
        )
        # create_documents 将文本分割成带元数据的文档对象
        documents = text_splitter.create_documents([path])

    # 初始化用于生成摘要的 LLM
    # gpt-4o-mini 是轻量级模型，适合快速生成摘要
    summary_llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=4000)
    # 创建摘要链
    # map_reduce 是摘要类型：先对每个块生成摘要，再合并成一个最终摘要
    summary_chain = load_summarize_chain(summary_llm, chain_type="map_reduce")

    # 定义异步摘要函数
    async def summarize_doc(doc):
        """
        为单个文档生成摘要

        参数：
            doc: 要摘要的文档对象

        返回：
            包含摘要的新文档对象
        """
        # 异步调用摘要链
        # retry_with_exponential_backoff 在 API 限流时自动重试
        summary_output = await retry_with_exponential_backoff(summary_chain.ainvoke([doc]))
        # 提取摘要文本
        summary = summary_output['output_text']
        # 创建包含摘要的新文档对象
        # metadata 包含源文件路径、页码和摘要标记
        return Document(page_content=summary, metadata={"source": path, "page": doc.metadata["page"], "summary": True})

    # 批量生成摘要
    summaries = []
    # 批次大小：每次处理 5 个文档，平衡速度和 API 限制
    batch_size = 5
    # 分批处理文档
    for i in range(0, len(documents), batch_size):
        # 取出一批文档
        batch = documents[i:i + batch_size]
        # 并行生成这批文档的摘要
        # asyncio.gather 并发执行多个异步任务
        batch_summaries = await asyncio.gather(*[summarize_doc(doc) for doc in batch])
        # 将摘要添加到列表
        summaries.extend(batch_summaries)
        # 休眠 1 秒，避免 API 调用过快被限流
        await asyncio.sleep(1)

    # 将原始文档分割成详细文本块
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    # split_documents 将文档分割成小块
    detailed_chunks = await asyncio.to_thread(text_splitter.split_documents, documents)

    # 为每个块添加元数据
    for i, chunk in enumerate(detailed_chunks):
        # metadata 是字典，存储文档的附加信息
        chunk.metadata.update({
            "chunk_id": i,  # 块的唯一标识
            "summary": False,  # 标记这不是摘要
            "page": int(chunk.metadata.get("page", 0))  # 页码，转换为整数
        })

    # 初始化嵌入模型
    # OpenAI 的嵌入模型将文本转换为向量（数学表示）
    embeddings = OpenAIEmbeddings()

    # 定义异步创建向量库的函数
    async def create_vectorstore(docs):
        """
        异步创建 FAISS 向量存储

        参数：
            docs: 要嵌入的文档列表

        返回：
            FAISS 向量存储
        """
        # asyncio.to_thread 在线程池中运行同步的 from_documents
        # retry_with_exponential_backoff 处理 API 限流
        return await retry_with_exponential_backoff(asyncio.to_thread(FAISS.from_documents, docs, embeddings))

    # 并行创建两个向量存储
    # asyncio.gather 并发执行多个异步任务
    summary_vectorstore, detailed_vectorstore = await asyncio.gather(
        create_vectorstore(summaries),  # 创建摘要向量库
        create_vectorstore(detailed_chunks)  # 创建详细向量库
    )

    # 返回两个向量存储
    return summary_vectorstore, detailed_vectorstore


# ============================================================================
# 分层检索函数
# ============================================================================
def retrieve_hierarchical(query, summary_vectorstore, detailed_vectorstore, k_summaries=3, k_chunks=5):
    """
    使用查询执行分层检索

    检索策略：
    1. 先在摘要库中搜索，找到最相关的 k 个摘要
    2. 对于每个摘要，找到它所在的页面
    3. 在详细库中，只从这些页面检索相关块

    这种策略的优势：
    - 先用摘要快速定位相关区域
    - 再在相关区域内精确检索，提高准确性

    参数：
        query: 用户查询
        summary_vectorstore: 摘要向量存储
        detailed_vectorstore: 详细向量存储
        k_summaries: 检索的摘要数量
        k_chunks: 每个页面检索的块数量

    返回：
        relevant_chunks: 相关文档块列表
    """
    # 第一步：在摘要库中检索最相关的摘要
    # similarity_search 返回与查询向量最接近的文档
    top_summaries = summary_vectorstore.similarity_search(query, k=k_summaries)

    # 存储最终的相关块
    relevant_chunks = []

    # 第二步：对于每个摘要，找到对应页面的详细块
    for summary in top_summaries:
        # 从摘要元数据中获取页码
        page_number = summary.metadata["page"]

        # 定义页面过滤函数
        # 这个函数只接受与摘要同页的块
        page_filter = lambda metadata: metadata["page"] == page_number

        # 在详细库中检索
        # filter 参数限制只从指定页面检索
        page_chunks = detailed_vectorstore.similarity_search(query, k=k_chunks, filter=page_filter)

        # 将这些块添加到结果列表
        relevant_chunks.extend(page_chunks)

    return relevant_chunks


# ============================================================================
# 分层 RAG 主类
# ============================================================================
class HierarchicalRAG:
    """
    分层 RAG（检索增强生成）系统

    这个类实现了完整的分层检索流程：
    1. 构建摘要和详细两个向量库
    2. 支持从缓存加载已构建的向量库
    3. 使用分层策略进行检索

    分层检索的优势：
    - 先粗粒度（摘要）定位，再细粒度（详细块）检索
    - 兼顾速度和准确性
    """

    def __init__(self, pdf_path, chunk_size=1000, chunk_overlap=200):
        """
        初始化分层 RAG 系统

        参数：
            pdf_path: PDF 文件路径
            chunk_size: 文本块大小
            chunk_overlap: 块重叠大小
        """
        # 保存 PDF 路径
        self.pdf_path = pdf_path
        # 保存分块参数
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 初始化为 None，后续会构建
        self.summary_store = None
        self.detailed_store = None

    async def run(self, query):
        """
        运行分层 RAG 系统

        参数：
            query: 用户查询

        流程：
            1. 检查是否有缓存的向量库，有则加载，没有则构建
            2. 使用分层检索找到相关块
            3. 打印结果
        """
        # 检查向量存储是否已缓存到本地
        # 缓存可以避免重复构建，节省时间和 API 调用
        if os.path.exists("../vector_stores/summary_store") and os.path.exists("../vector_stores/detailed_store"):
            print("加载已缓存的向量存储...")
            # 重新初始化嵌入模型（需要与构建时相同的模型）
            embeddings = OpenAIEmbeddings()
            # 从磁盘加载摘要向量库
            # allow_dangerous_deserialization=True 允许加载pickle文件
            self.summary_store = FAISS.load_local("../vector_stores/summary_store", embeddings, allow_dangerous_deserialization=True)
            # 从磁盘加载详细向量库
            self.detailed_store = FAISS.load_local("../vector_stores/detailed_store", embeddings, allow_dangerous_deserialization=True)
        else:
            print("构建新的向量存储...")
            # 调用异步函数构建分层向量存储
            # 这个过程包括：读取 PDF、生成摘要、分割块、嵌入向量
            self.summary_store, self.detailed_store = await encode_pdf_hierarchical(self.pdf_path, self.chunk_size, self.chunk_overlap)
            # 将向量存储保存到本地，下次可以直接加载
            self.summary_store.save_local("../vector_stores/summary_store")
            self.detailed_store.save_local("../vector_stores/detailed_store")

        # 执行分层检索
        # retrieve_hierarchical 先在摘要库搜索，再在详细库搜索
        results = retrieve_hierarchical(query, self.summary_store, self.detailed_store)

        # 打印每个结果
        for chunk in results:
            # 显示页码
            print(f"页面：{chunk.metadata['page']}")
            # 显示内容片段（...表示有截断）
            print(f"内容：{chunk.page_content}...")
            print("---")


# ============================================================================
# 参数解析函数
# ============================================================================
def parse_args():
    """
    解析命令行参数

    允许用户配置：
    - PDF 路径
    - 分块大小和重叠
    - 测试查询

    返回：
        解析后的参数对象
    """
    import argparse
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="在给定 PDF 上运行分层 RAG。")

    # PDF 路径参数
    parser.add_argument("--pdf_path", type=str, default="../data/Understanding_Climate_Change.pdf", help="PDF 文档的路径。")

    # 分块大小参数
    parser.add_argument("--chunk_size", type=int, default=1000, help="每个文本块的大小。")

    # 分块重叠参数
    parser.add_argument("--chunk_overlap", type=int, default=200, help="连续块之间的重叠。")

    # 查询参数
    parser.add_argument("--query", type=str, default='什么是温室效应',
                        help="在文档中搜索的查询。")
    return parser.parse_args()


# ============================================================================
# 程序主入口
# ============================================================================
if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    # 创建分层 RAG 实例
    rag = HierarchicalRAG(args.pdf_path, args.chunk_size, args.chunk_overlap)

    # 运行异步 RAG 系统
    # asyncio.run() 是异步程序的入口
    asyncio.run(rag.run(args.query))
