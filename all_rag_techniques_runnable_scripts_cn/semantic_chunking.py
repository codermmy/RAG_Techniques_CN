# ============================================================================
# 导入必要的库和模块
# ============================================================================
import time  # 用于时间相关操作，这里用来记录代码运行时间
import os    # 用于操作系统相关的操作，如文件路径处理
import sys   # 用于操作系统相关的操作，如添加模块搜索路径
import argparse  # 用于解析命令行参数
from dotenv import load_dotenv  # 用于从.env 文件加载环境变量
# 导入自定义的辅助函数模块
from helper_functions import *
# 从 langchain_experimental 导入语义分块器
# SemanticChunker 是一种智能分块器，它根据语义相似性来决定在哪里切分文本
from langchain_experimental.text_splitter import SemanticChunker, BreakpointThresholdType
# 导入 OpenAI 嵌入模型，用于将文本转换为向量
from langchain_openai.embeddings import OpenAIEmbeddings

# 将父目录添加到路径，因为我们使用笔记本工作
# 这行代码的作用是：让当前脚本可以导入上级目录中的模块
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# 从 .env 文件加载环境变量（例如 OpenAI API 密钥）
# .env 文件是一个存储敏感信息（如 API 密钥）的安全方式，避免硬编码在代码中
load_dotenv()
# 设置 OpenAI API 密钥，这是调用 OpenAI 服务所必需的
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


# ============================================================================
# SemanticChunkingRAG 类 - 使用语义分块的 RAG 系统
# ============================================================================
# 用于运行语义分块并返回分块和检索时间的函数
class SemanticChunkingRAG:
    """
    一个用于处理语义分块 RAG 过程的类，用于文档分块和查询检索。

    语义分块（Semantic Chunking）是一种智能的文本分割方法：
    - 传统分块：按照固定的字符数或句子数来分割文本
    - 语义分块：根据文本内容的语义相似性来决定在哪里分割

    为什么语义分块更好？
    - 它可以确保每个文本块在语义上是连贯的
    - 避免把一个完整的概念切分成两半
    - 检索时可以找到更完整、更有意义的上下文

    工作原理：
    1. 将文档分割成句子
    2. 计算相邻句子之间的语义相似度
    3. 在相似度突然下降的地方设置"断点"进行分割
    4. 每个语义块包含一组语义上相关的句子
    """

    def __init__(self, path, n_retrieved=2, embeddings=None, breakpoint_type: BreakpointThresholdType = "percentile",
                 breakpoint_amount=90):
        """
        通过使用语义分块器对内容进行编码来初始化 SemanticChunkingRAG。

        参数：
            path (str): 要编码的 PDF 文件路径。
            n_retrieved (int): 每个查询检索的块数（默认：2）。
                              每次查询会返回多少个最相关的文本块。
            embeddings: 要使用的嵌入模型。
                       如果未提供，会使用 OpenAI 的默认嵌入模型。
            breakpoint_type (str): 语义断点阈值的类型。
                                  决定如何计算断点的统计方法：
                                  - percentile: 百分位数（如第 90 百分位）
                                  - standard_deviation: 标准差
                                  - interquartile: 四分位数
                                  - gradient: 梯度
            breakpoint_amount (float): 语义断点阈值的数量。
                                      具体含义取决于 breakpoint_type：
                                      如果是 percentile，90 表示在第 90 百分位设置断点
        """
        print("\n--- 初始化语义分块 RAG ---")
        # 将 PDF 读取为字符串
        # read_pdf_to_string 函数读取 PDF 文件并将其全部内容转换为纯文本字符串
        content = read_pdf_to_string(path)

        # 使用提供的嵌入模型或初始化 OpenAI 嵌入
        # 如果调用者没有提供 embeddings 参数，就使用 OpenAI 的默认嵌入模型
        # 嵌入模型用于将文本转换为向量，以便计算语义相似度
        self.embeddings = embeddings if embeddings else OpenAIEmbeddings()

        # 初始化语义分块器
        # SemanticChunker 会自动分析文本，找到语义上的自然分界点
        self.semantic_chunker = SemanticChunker(
            self.embeddings,  # 使用上面初始化的嵌入模型
            # 断点阈值类型，决定如何判断"这里应该切分"
            breakpoint_threshold_type=breakpoint_type,
            # 断点阈值数量，值越大分块越细
            breakpoint_threshold_amount=breakpoint_amount
        )

        # 测量语义分块时间
        # 记录开始时间，用于计算分块过程的耗时
        start_time = time.time()
        # create_documents 方法执行实际的语义分块
        # 输入是一个字符串列表（这里只有一个字符串，所以用 [content]）
        # 输出是一个 Document 对象列表，每个对象包含：
        # - page_content: 文本块的内容
        # - metadata: 元数据（如页码等）
        self.semantic_docs = self.semantic_chunker.create_documents([content])
        # 记录分块过程所花费的时间
        self.time_records = {'Chunking': time.time() - start_time}
        print(f"语义分块时间：{self.time_records['Chunking']:.2f} 秒")

        # 从语义块创建向量存储和检索器
        # FAISS 是 Facebook AI 开发的向量相似度搜索库
        # from_documents 方法会：
        # 1. 将每个语义块转换为向量
        # 2. 将所有向量存入 FAISS 索引中，以便快速检索
        self.semantic_vectorstore = FAISS.from_documents(self.semantic_docs, self.embeddings)
        # 从向量存储创建检索器
        # search_kwargs={"k": n_retrieved} 指定每次检索返回 k 个最相似的结果
        self.semantic_retriever = self.semantic_vectorstore.as_retriever(search_kwargs={"k": n_retrieved})

    def run(self, query):
        """
        检索并显示给定查询的上下文。

        参数：
            query (str): 要检索上下文的查询。
                        这是用户提出的问题，比如"气候变化的主要原因是什么？"

        返回：
            tuple: 检索时间。
        """
        # 测量语义检索时间
        # 记录开始时间，用于计算检索过程的耗时
        start_time = time.time()
        # retrieve_context_per_question 函数会：
        # 1. 将查询转换为向量
        # 2. 在向量数据库中搜索最相似的文本块
        # 3. 返回检索到的文本块列表
        semantic_context = retrieve_context_per_question(query, self.semantic_retriever)
        # 记录检索过程所花费的时间
        self.time_records['Retrieval'] = time.time() - start_time
        print(f"语义检索时间：{self.time_records['Retrieval']:.2f} 秒")

        # 显示检索到的上下文
        # show_context 函数会将检索到的文本块格式化并打印出来
        show_context(semantic_context)
        # 返回时间记录，可用于性能分析
        return self.time_records


# ============================================================================
# 参数解析函数 - 解析用户在命令行中输入的参数
# ============================================================================
# 用于解析命令行参数的函数
def parse_args():
    """
    解析命令行参数。

    这个函数使用 argparse 模块来定义和解析命令行参数，
    让用户可以通过命令行来配置程序的行为，而不需要修改代码。

    返回：
        args: 包含所有命令行参数的对象
    """
    # 创建一个参数解析器，描述程序接受哪些参数
    parser = argparse.ArgumentParser(
        description="使用语义分块 RAG 处理 PDF 文档。")
    # 添加 --path 参数，指定要处理的 PDF 文件路径
    parser.add_argument("--path", type=str, default="../data/Understanding_Climate_Change.pdf",
                        help="要编码的 PDF 文件路径。")
    # 添加 --n_retrieved 参数，控制每次检索返回的结果数量
    parser.add_argument("--n_retrieved", type=int, default=2,
                        help="每个查询检索的块数（默认：2）。")
    # 添加 --breakpoint_threshold_type 参数，选择断点阈值类型
    # choices 参数限制了用户只能从这四个选项中选择一个
    parser.add_argument("--breakpoint_threshold_type", type=str,
                        choices=["percentile", "standard_deviation", "interquartile", "gradient"],
                        default="percentile",
                        help="用于分块的断点阈值类型（默认：百分位数）。")
    # 添加 --breakpoint_threshold_amount 参数，设置断点阈值数量
    parser.add_argument("--breakpoint_threshold_amount", type=float, default=90,
                        help="要使用的断点阈值数量（默认：90）。")
    # 添加 --chunk_size 参数（用于简单分块对比）
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="简单分块中每个文本块的大小（默认：1000）。")
    # 添加 --chunk_overlap 参数（用于简单分块对比）
    parser.add_argument("--chunk_overlap", type=int, default=200,
                        help="简单分块中连续块之间的重叠（默认：200）。")
    # 添加 --query 参数，指定要测试的查询语句
    parser.add_argument("--query", type=str, default="What is the main cause of climate change?",
                        help="用于测试检索器的查询（默认：'气候变化的主要原因是什么？'）。")
    # 添加 --experiment 标志，如果用户指定这个标志，就会运行对比实验
    parser.add_argument("--experiment", action="store_true",
                        help="运行实验以比较语义分块和简单分块之间的性能。")

    return parser.parse_args()


# ============================================================================
# 主函数 - 程序执行的入口
# ============================================================================
# 用于处理 PDF、分块文本和测试检索器的主函数
def main(args):
    """
    程序的主函数，协调整个语义分块 RAG 流程。

    这个函数是程序的入口点，它负责：
    1. 创建 SemanticChunkingRAG 实例（会自动处理文档）
    2. 执行查询检索

    参数：
        args: 包含所有命令行参数的对象
    """
    # 初始化 SemanticChunkingRAG
    # 创建实例时会自动：
    # - 读取 PDF 文档
    # - 使用语义分块器分割文本
    # - 将文本块转换为向量并存储
    # - 创建检索器
    semantic_rag = SemanticChunkingRAG(
        path=args.path,  # PDF 文件路径
        n_retrieved=args.n_retrieved,  # 检索结果数量
        breakpoint_type=args.breakpoint_threshold_type,  # 断点阈值类型
        breakpoint_amount=args.breakpoint_threshold_amount  # 断点阈值数量
    )

    # 运行查询
    # run 方法会执行检索并显示结果
    semantic_rag.run(args.query)


# ============================================================================
# 程序入口 - Python 脚本执行时的起点
# ============================================================================
if __name__ == '__main__':
    # 使用解析的参数调用主函数
    # 这行代码确保只有在直接运行此脚本时才会执行 main 函数
    # 如果这个文件被作为模块导入到其他文件中，main 函数不会自动执行
    main(parse_args())
