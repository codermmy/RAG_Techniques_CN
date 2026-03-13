# ============================================================================
# 导入必要的库和模块
# ============================================================================
import os    # 用于操作系统相关的操作，如文件路径处理
import sys   # 用于操作系统相关的操作，如添加模块搜索路径
import time  # 用于时间相关操作，这里用来记录代码运行时间
import argparse  # 用于解析命令行参数
from dotenv import load_dotenv  # 用于从.env 文件加载环境变量

# 从 langchain 导入上下文压缩相关的类
# LLMChainExtractor 是一个智能压缩器，它会从检索到的文档中提取与查询最相关的部分
from langchain.retrievers.document_compressors import LLMChainExtractor
# ContextualCompressionRetriever 是一个包装器，它在检索后对结果进行压缩
from langchain.retrievers import ContextualCompressionRetriever
# RetrievalQA 是一个用于基于检索的问答的链
from langchain.chains import RetrievalQA
# 导入自定义的辅助函数模块
from helper_functions import *
# 导入 RAG 评估模块
from evaluation.evalute_rag import *

# 将父目录添加到路径，因为我们使用 notebooks 工作
# 这行代码的作用是：让当前脚本可以导入上级目录中的模块
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# 从 .env 文件加载环境变量
# .env 文件是一个存储敏感信息（如 API 密钥）的安全方式，避免硬编码在代码中
load_dotenv()
# 设置 OpenAI API 密钥，这是调用 OpenAI 服务所必需的
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


# ============================================================================
# ContextualCompressionRAG 类 - 使用上下文压缩的 RAG 系统
# ============================================================================
class ContextualCompressionRAG:
    """
    一个用于创建基于检索的问答系统的类，带有上下文压缩检索器。

    什么是上下文压缩（Contextual Compression）？
    - 传统检索：检索完整的文本块，可能包含大量无关信息
    - 上下文压缩：先检索，然后用 LLM 提取与查询最相关的部分

    为什么需要上下文压缩？
    1. 减少噪声：原始检索的文本块可能包含无关句子
    2. 提高效率：更少的 token 意味着更快的响应和更低的成本
    3. 更好的答案：LLM 专注于相关信息，生成更准确的答案

    工作流程：
    1. 用户提出查询
    2. 检索器找到最相似的文本块
    3. 压缩器分析这些文本块，提取与查询相关的片段
    4. 将压缩后的上下文传递给 LLM 生成答案

    类比：
    - 传统检索：给你整本书让你找答案
    - 上下文压缩：先帮你划出书中与问题相关的重点段落
    """

    def __init__(self, path, model_name="gpt-4o-mini", temperature=0, max_tokens=4000):
        """
        初始化 ContextualCompressionRAG，设置文档存储和检索器。

        参数：
            path (str): 要处理的 PDF 文件路径。
                       这是包含源文档的 PDF 文件的路径。
            model_name (str): 要使用的语言模型名称（默认：gpt-4o-mini）。
                             gpt-4o-mini 是一个快速且经济的模型。
            temperature (float): 语言模型的温度参数（默认：0）。
                               温度控制输出的随机性：
                               - 0：最确定、最聚焦
                               - 1：更随机、更有创意
            max_tokens (int): 语言模型的最大 token 数（默认：4000）。
                             限制模型生成的最大 token 数量。
        """
        print("\n--- 初始化上下文压缩 RAG ---")
        # 保存初始化参数
        self.path = path  # PDF 文件路径
        self.model_name = model_name  # 模型名称
        self.temperature = temperature  # 温度参数
        self.max_tokens = max_tokens  # 最大 token 数

        # 步骤 1：创建向量存储
        # 调用辅助函数 encode_pdf 将 PDF 文档转换为向量存储
        # 这个函数会读取 PDF、分割文本、创建嵌入并存储到向量数据库中
        self.vector_store = self._encode_document()

        # 步骤 2：创建检索器
        # as_retriever() 方法将向量存储转换为检索器对象
        # 检索器的作用是：给定一个查询，找出最相关的文本块
        self.retriever = self.vector_store.as_retriever()

        # 步骤 3：初始化语言模型并创建上下文压缩器
        # 调用辅助函数初始化 LLM
        self.llm = self._initialize_llm()
        # LLMChainExtractor.from_llm 创建一个使用 LLM 的文档压缩器
        # 这个压缩器会分析检索到的文档，提取与查询最相关的部分
        self.compressor = LLMChainExtractor.from_llm(self.llm)

        # 步骤 4：将检索器与压缩器结合
        # ContextualCompressionRetriever 是一个包装器：
        # 1. 先用 base_retriever 检索文档
        # 2. 再用 base_compressor 压缩提取相关信息
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,  # 用于压缩的 LLM 提取器
            base_retriever=self.retriever     # 用于检索的向量检索器
        )

        # 步骤 5：使用压缩检索器创建 QA 链
        # RetrievalQA.from_chain_type 创建一个完整的问答链：
        # 1. 接收用户查询
        # 2. 使用检索器找到相关文档
        # 3. 使用压缩器提取相关信息
        # 4. 将信息和查询传递给 LLM 生成答案
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,                    # 用于生成答案的语言模型
            retriever=self.compression_retriever,  # 压缩检索器
            return_source_documents=True     # 返回源文档，便于追溯答案来源
        )

    def _encode_document(self):
        """
        辅助函数：将文档编码到向量存储中。

        这个函数调用 encode_pdf 辅助函数，它会：
        1. 读取 PDF 文件
        2. 将文本分割成指定大小的块
        3. 使用 OpenAI 的嵌入模型将每个文本块转换为向量
        4. 将所有向量存储到 FAISS 向量数据库中

        返回：
            VectorStore: 包含文档向量的向量存储。
        """
        return encode_pdf(self.path)

    def _initialize_llm(self):
        """
        辅助函数：初始化语言模型。

        创建 ChatOpenAI 实例，这是一个用于与 OpenAI 聊天模型交互的类。

        返回：
            ChatOpenAI: 配置好的语言模型实例。
        """
        # ChatOpenAI 创建一个 OpenAI 聊天模型实例
        # temperature: 控制输出随机性，0 表示最确定
        # model_name: 指定使用的模型，如 gpt-4o-mini
        # max_tokens: 限制生成的最大 token 数
        return ChatOpenAI(temperature=self.temperature, model_name=self.model_name, max_tokens=self.max_tokens)

    def run(self, query):
        """
        使用 QA 链执行查询并打印结果。

        这个方法执行完整的上下文压缩 RAG 流程：
        1. 接收用户查询
        2. 检索相关文档
        3. 压缩提取相关信息
        4. 生成答案
        5. 显示结果和性能指标

        参数：
            query (str): 针对文档运行的查询。
                        这是用户提出的问题，比如"文档的主题是什么？"

        返回：
            tuple: 包含结果字典和执行时间的元组。
        """
        print("\n--- 运行查询 ---")
        # 记录开始时间，用于计算查询执行耗时
        start_time = time.time()
        # invoke 方法执行 QA 链
        # {"query": query} 是输入，包含用户的查询
        # 返回的 result 包含：
        # - result: LLM 生成的答案
        # - source_documents: 用于生成答案的源文档
        result = self.qa_chain.invoke({"query": query})
        # 计算查询执行耗时
        elapsed_time = time.time() - start_time

        # 显示结果和源文档
        # result['result'] 是 LLM 生成的答案
        print(f"结果：{result['result']}")
        # result['source_documents'] 是用于生成答案的源文档
        # 这有助于验证答案的可信度
        print(f"源文档：{result['source_documents']}")
        # 显示查询执行时间
        print(f"查询执行时间：{elapsed_time:.2f} 秒")
        # 返回结果和执行时间，便于进一步分析
        return result, elapsed_time


# ============================================================================
# 参数解析函数 - 解析用户在命令行中输入的参数
# ============================================================================
# 解析命令行参数的函数
def parse_args():
    """
    解析命令行参数。

    这个函数使用 argparse 模块来定义和解析命令行参数，
    让用户可以通过命令行来配置程序的行为，而不需要修改代码。

    返回：
        args: 包含所有命令行参数的对象
    """
    # 创建参数解析器，描述程序接受哪些参数
    parser = argparse.ArgumentParser(description="使用上下文压缩 RAG 处理 PDF 文档。")
    # 添加 --model_name 参数，指定要使用的语言模型
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini",
                        help="要使用的语言模型名称（默认：gpt-4o-mini）。")
    # 添加 --path 参数，指定要处理的 PDF 文件路径
    parser.add_argument("--path", type=str, default="../data/Understanding_Climate_Change.pdf",
                        help="要处理的 PDF 文件路径。")
    # 添加 --query 参数，指定要测试的查询语句
    parser.add_argument("--query", type=str, default="文档的主题是什么？",
                        help="用于测试检索器的查询（默认：'文档的主题是什么？'）。")
    # 添加 --temperature 参数，控制语言模型的随机性
    parser.add_argument("--temperature", type=float, default=0,
                        help="语言模型的温度参数（默认：0）。")
    # 添加 --max_tokens 参数，限制生成的最大 token 数
    parser.add_argument("--max_tokens", type=int, default=4000,
                        help="语言模型的最大 token 数（默认：4000）。")

    return parser.parse_args()


# ============================================================================
# 主函数 - 程序执行的入口
# ============================================================================
# 运行 RAG 流程的主函数
def main(args):
    """
    程序的主函数，协调整个上下文压缩 RAG 流程。

    这个函数是程序的入口点，它负责：
    1. 创建 ContextualCompressionRAG 实例（会自动处理文档）
    2. 执行查询检索

    参数：
        args: 包含所有命令行参数的对象
    """
    # 初始化 ContextualCompressionRAG
    # 创建实例时会自动：
    # - 读取并编码 PDF 文档
    # - 创建向量存储和检索器
    # - 初始化语言模型和压缩器
    # - 创建上下文压缩检索器和 QA 链
    contextual_compression_rag = ContextualCompressionRAG(
        path=args.path,              # PDF 文件路径
        model_name=args.model_name,  # 语言模型名称
        temperature=args.temperature,  # 温度参数
        max_tokens=args.max_tokens   # 最大 token 数
    )

    # 运行查询
    # run 方法会执行检索、压缩和答案生成，并显示结果
    contextual_compression_rag.run(args.query)


# ============================================================================
# 程序入口 - Python 脚本执行时的起点
# ============================================================================
if __name__ == '__main__':
    # 使用解析的参数调用主函数
    # 这行代码确保只有在直接运行此脚本时才会执行 main 函数
    # 如果这个文件被作为模块导入到其他文件中，main 函数不会自动执行
    main(parse_args())
