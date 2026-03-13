# ============================================================================
# 导入必要的库和模块
# ============================================================================
import os      # 用于操作系统相关的操作，如文件路径处理
import sys     # 用于操作系统相关的操作，如添加模块搜索路径
import argparse  # 用于解析命令行参数，让用户可以通过命令行配置程序行为
import time    # 用于时间相关操作，这里用来记录代码运行时间
from dotenv import load_dotenv  # 用于从.env 文件加载环境变量

# 将父目录添加到路径，因为我们使用笔记本工作
# 这行代码的作用是：让当前脚本可以导入上级目录中的模块
# os.getcwd() 获取当前工作目录，os.path.join 拼接路径，os.path.abspath 获取绝对路径
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# 导入自定义的辅助函数模块
# helper_functions 包含了一些常用的工具函数，如 encode_pdf、retrieve_context_per_question 等
from helper_functions import *
# 导入 RAG 评估模块，用于评估检索系统的性能
from evaluation.evalute_rag import *

# 从 .env 文件加载环境变量（例如 OpenAI API 密钥）
# .env 文件是一个存储敏感信息（如 API 密钥）的安全方式，避免硬编码在代码中
load_dotenv()
# 设置 OpenAI API 密钥，这是调用 OpenAI 服务所必需的
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


# ============================================================================
# SimpleRAG 类 - 实现基础的 RAG（检索增强生成）功能
# ============================================================================
class SimpleRAG:
    """
    一个用于处理简单 RAG 过程的类，用于文档分块和查询检索。

    RAG（Retrieval-Augmented Generation）检索增强生成是一种技术：
    1. 首先从文档中检索相关信息
    2. 然后将这些信息提供给大语言模型来生成答案

    这个类实现了最简单的 RAG 流程：
    - 将 PDF 文档分割成小块（分块）
    - 将每个块转换为向量（嵌入）
    - 根据查询检索最相关的块
    """

    def __init__(self, path, chunk_size=1000, chunk_overlap=200, n_retrieved=2):
        """
        通过对 PDF 文档进行编码并创建检索器来初始化 SimpleRAGRetriever。

        参数：
            path (str): 要编码的 PDF 文件路径。
            chunk_size (int): 每个文本块的大小（默认：1000）。
                          这个值决定了每个文本片段包含多少个字符。
                          太大会丢失细节，太小会丢失上下文。
            chunk_overlap (int): 连续块之间的重叠（默认：200）。
                               相邻的块之间会有部分重叠，这是为了确保
                               重要的信息不会被切分到两个块中而丢失语义。
            n_retrieved (int): 每个查询检索的块数（默认：2）。
                             每次查询会返回多少个最相关的文本块。
        """
        print("\n--- 初始化简单 RAG 检索器 ---")

        # 使用 OpenAI 嵌入将 PDF 文档编码到向量存储中
        # encode_pdf 函数会：
        # 1. 读取 PDF 文件
        # 2. 将文本分割成指定大小的块
        # 3. 使用 OpenAI 的嵌入模型将每个文本块转换为向量
        # 4. 将所有向量存储到 FAISS 向量数据库中
        start_time = time.time()  # 记录开始时间，用于计算耗时
        self.vector_store = encode_pdf(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # 记录分块过程所花费的时间
        self.time_records = {'Chunking': time.time() - start_time}
        print(f"分块时间：{self.time_records['Chunking']:.2f} 秒")

        # 从向量存储创建检索器
        # as_retriever() 方法将向量存储转换为检索器对象
        # search_kwargs={"k": n_retrieved} 指定每次检索返回 k 个最相似的结果
        # 检索器的作用是：给定一个查询，找出最相关的文本块
        self.chunks_query_retriever = self.vector_store.as_retriever(search_kwargs={"k": n_retrieved})

    def run(self, query):
        """
        检索并显示给定查询的上下文。

        参数：
            query (str): 要检索上下文的查询。
                        这是用户提出的问题，比如"气候变化的主要原因是什么？"

        返回：
            tuple: 检索时间。
        """
        # 测量检索时间
        # 记录检索开始时间，用于计算整个检索过程耗时
        start_time = time.time()
        # retrieve_context_per_question 函数会：
        # 1. 将查询转换为向量
        # 2. 在向量数据库中搜索最相似的文本块
        # 3. 返回检索到的文本块列表
        context = retrieve_context_per_question(query, self.chunks_query_retriever)
        # 记录检索过程所花费的时间
        self.time_records['Retrieval'] = time.time() - start_time
        print(f"检索时间：{self.time_records['Retrieval']:.2f} 秒")

        # 显示检索到的上下文
        # show_context 函数会将检索到的文本块格式化并打印出来
        show_context(context)


# ============================================================================
# 参数验证函数 - 确保用户输入的命令行参数合法
# ============================================================================
# 用于验证命令行输入的函数
def validate_args(args):
    """
    验证命令行参数是否合法。

    这个函数检查用户提供的参数是否有意义，
    如果参数不合法会抛出异常，阻止程序继续执行。

    参数：
        args: 解析后的命令行参数对象

    返回：
        args: 验证通过的参数对象
    """
    # chunk_size 必须是正数，因为文本块大小不能为 0 或负数
    if args.chunk_size <= 0:
        raise ValueError("chunk_size 必须是正整数。")
    # chunk_overlap 必须是非负数，因为重叠量不能为负
    if args.chunk_overlap < 0:
        raise ValueError("chunk_overlap 必须是非负整数。")
    # n_retrieved 必须是正数，因为检索数量不能为 0 或负数
    if args.n_retrieved <= 0:
        raise ValueError("n_retrieved 必须是正整数。")
    return args


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
    parser = argparse.ArgumentParser(description="编码 PDF 文档并测试简单 RAG。")
    # 添加 --path 参数，指定要处理的 PDF 文件路径
    parser.add_argument("--path", type=str, default="../data/Understanding_Climate_Change.pdf",
                        help="要编码的 PDF 文件路径。")
    # 添加 --chunk_size 参数，控制文本块的大小
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="每个文本块的大小（默认：1000）。")
    # 添加 --chunk_overlap 参数，控制文本块之间的重叠量
    parser.add_argument("--chunk_overlap", type=int, default=200,
                        help="连续块之间的重叠（默认：200）。")
    # 添加 --n_retrieved 参数，控制每次检索返回的结果数量
    parser.add_argument("--n_retrieved", type=int, default=2,
                        help="每个查询检索的块数（默认：2）。")
    # 添加 --query 参数，指定要测试的查询语句
    parser.add_argument("--query", type=str, default="What is the main cause of climate change?",
                        help="用于测试检索器的查询（默认：'气候变化的主要原因是什么？'）。")
    # 添加 --evaluate 标志，如果用户指定这个标志，就会运行评估
    parser.add_argument("--evaluate", action="store_true",
                        help="是否评估检索器的性能（默认：False）。")

    # 解析并验证参数
    # parse_args() 会从命令行读取参数并返回一个参数对象
    # validate_args() 会检查这些参数是否合法
    return validate_args(parser.parse_args())


# ============================================================================
# 主函数 - 程序执行的入口
# ============================================================================
# 用于处理参数解析并调用 SimpleRAG 类的主函数
def main(args):
    """
    程序的主函数，协调整个 RAG 流程。

    这个函数是程序的入口点，它负责：
    1. 创建 SimpleRAG 实例（会自动处理文档）
    2. 执行查询检索
    3. 可选地评估检索器性能

    参数：
        args: 包含所有命令行参数的对象
    """
    # 初始化 SimpleRAGRetriever
    # 创建 SimpleRAG 实例时，会自动：
    # - 读取并分块 PDF 文档
    # - 将文本块转换为向量并存储
    # - 创建检索器
    simple_rag = SimpleRAG(
        path=args.path,           # PDF 文件路径
        chunk_size=args.chunk_size,      # 文本块大小
        chunk_overlap=args.chunk_overlap, # 块重叠量
        n_retrieved=args.n_retrieved     # 检索结果数量
    )

    # 根据查询检索上下文
    # run 方法会执行检索并显示结果
    simple_rag.run(args.query)

    # 评估检索器的性能（如果请求）
    # 如果用户指定了 --evaluate 标志，就会运行评估
    # evaluate_rag 函数会使用预设的问题和答案来测试检索器的准确性
    if args.evaluate:
        evaluate_rag(simple_rag.chunks_query_retriever)


# ============================================================================
# 程序入口 - Python 脚本执行时的起点
# ============================================================================
if __name__ == '__main__':
    # 使用解析的参数调用主函数
    # 这行代码确保只有在直接运行此脚本时才会执行 main 函数
    # 如果这个文件被作为模块导入到其他文件中，main 函数不会自动执行
    main(parse_args())
