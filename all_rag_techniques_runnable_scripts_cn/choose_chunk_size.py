# ============================================================================
# 导入必要的库和模块
# ============================================================================
# nest_asyncio 用于修复 Jupyter notebooks 中的 asyncio 事件循环问题
# 在 Jupyter 中运行异步代码时，需要这个库来避免事件循环冲突
import nest_asyncio
# random 用于随机抽样，这里用于从生成的问题中随机选择一部分
import random
# time 用于时间相关操作，这里用来记录代码运行时间
import time
# os 用于操作系统相关的操作，如文件路径处理
import os
# dotenv 用于从.env 文件加载环境变量
from dotenv import load_dotenv

# 从 llama_index 导入必要的类和函数
# llama_index 是一个用于构建 RAG 应用的框架，提供了数据加载、索引、查询等功能
from llama_index.core import (
    VectorStoreIndex,  # 向量索引，用于存储和检索文档向量
    SimpleDirectoryReader,  # 简单的目录读取器，用于加载文档
    Settings  # 全局设置，用于配置 LLM 等
)
from llama_index.core.prompts import PromptTemplate  # 提示模板，用于自定义 LLM 的输入格式
from llama_index.core.evaluation import (
    DatasetGenerator,  # 数据集生成器，用于从文档生成评估问题
    FaithfulnessEvaluator,  # 忠实度评估器，评估答案是否忠实于原文
    RelevancyEvaluator  # 相关性评估器，评估答案是否与问题相关
)
from llama_index.llms.openai import OpenAI  # OpenAI 语言模型接口
from llama_index.core.node_parser import SentenceSplitter  # 句子分割器，用于按指定 chunk_size 分割文本

# 应用 asyncio 修复以适用于 Jupyter notebooks
# 这行代码允许在 Jupyter 的异步环境中运行同步代码
nest_asyncio.apply()

# 加载环境变量
# 从 .env 文件中读取环境变量，如 OPENAI_API_KEY
load_dotenv()

# 设置 OpenAI API 密钥环境变量
# 这是调用 OpenAI 服务所必需的认证信息
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


# ============================================================================
# 评估函数 - 评估不同分块大小对 RAG 系统性能的影响
# ============================================================================
# 工具函数
def evaluate_response_time_and_accuracy(chunk_size, eval_questions, eval_documents, faithfulness_evaluator,
                                        relevancy_evaluator):
    """
    评估 GPT-3.5-turbo 生成的回答在给定分块大小下的平均响应时间、忠实度和相关性。

    这个函数的目的是：测试不同的 chunk_size（分块大小）如何影响 RAG 系统的性能。
    chunk_size 是 RAG 系统中的一个重要参数：
    - 太小的 chunk_size：每个块包含的信息太少，可能丢失上下文
    - 太大的 chunk_size：每个块包含太多无关信息，影响检索准确性

    参数：
    chunk_size (int): 处理的数据分块大小。
                     这个参数决定了每个文本块包含多少个字符。
    eval_questions (list): 评估问题列表。
                          这些是用于测试系统的问题，比如"气候变化的原因是什么？"
    eval_documents (list): 用于评估的文档。
                          这些是系统可以检索的源文档。
    faithfulness_evaluator (FaithfulnessEvaluator): 忠实度评估器。
                                                   用于评估生成的答案是否忠实于源文档。
    relevancy_evaluator (RelevancyEvaluator): 相关性评估器。
                                             用于评估生成的答案是否与问题相关。

    返回：
    tuple: 包含三个指标的元组：
        - average_response_time: 平均响应时间（秒）
        - average_faithfulness: 平均忠实度（0-1 之间，1 表示完全忠实）
        - average_relevancy: 平均相关性（0-1 之间，1 表示完全相关）
    """

    # 初始化累加器，用于累加多个问题的指标
    # 最后会除以问题数量得到平均值
    total_response_time = 0  # 总响应时间
    total_faithfulness = 0   # 总忠实度分数
    total_relevancy = 0      # 总相关性分数

    # 设置全局 LLM 为 GPT-3.5-turbo
    # GPT-3.5 是一个较快且经济的模型，适合批量测试
    # OpenAI(model="gpt-3.5-turbo") 创建一个使用 GPT-3.5 的 LLM 实例
    llm = OpenAI(model="gpt-3.5-turbo")
    # 将 LLM 设置为全局设置，这样后续的索引和查询都会使用这个模型
    Settings.llm = llm

    # 创建向量索引
    # SentenceSplitter 是一个文本分割器，它按照指定的 chunk_size 分割文本
    # 这与简单分块类似，但这里是用于 llama_index 框架
    splitter = SentenceSplitter(chunk_size=chunk_size)
    # VectorStoreIndex.from_documents 方法会：
    # 1. 使用 splitter 将文档分割成节点（每个节点是一个文本块）
    # 2. 将每个节点转换为向量
    # 3. 构建向量索引，以便后续检索
    vector_index = VectorStoreIndex.from_documents(eval_documents, transformations=[splitter])

    # 构建查询引擎
    # as_query_engine 方法将索引转换为可以查询的引擎
    # similarity_top_k=5 表示每次查询会检索 5 个最相似的文本块
    query_engine = vector_index.as_query_engine(similarity_top_k=5)
    # 记录评估问题的数量，用于后续计算平均值
    num_questions = len(eval_questions)

    # 遍历 eval_questions 中的每个问题以计算指标
    # 这是一个循环，逐个处理每个评估问题
    for question in eval_questions:
        # 记录开始时间
        start_time = time.time()
        # 执行查询，获取答案
        # query 方法会：
        # 1. 将问题转换为向量
        # 2. 检索最相似的文本块
        # 3. 使用 LLM 生成答案
        response_vector = query_engine.query(question)
        # 计算查询耗时
        elapsed_time = time.time() - start_time

        # 评估答案的忠实度
        # evaluate_response 方法检查答案是否忠实于源文档
        # passing 属性是一个布尔值，表示是否通过评估
        faithfulness_result = faithfulness_evaluator.evaluate_response(response=response_vector).passing
        # 评估答案的相关性
        # evaluate_response 方法检查答案是否与问题相关
        relevancy_result = relevancy_evaluator.evaluate_response(query=question, response=response_vector).passing

        # 累加各项指标
        total_response_time += elapsed_time
        total_faithfulness += faithfulness_result
        total_relevancy += relevancy_result

    # 计算平均值
    # 将总和除以问题数量，得到每个问题的平均指标
    average_response_time = total_response_time / num_questions
    average_faithfulness = total_faithfulness / num_questions
    average_relevancy = total_relevancy / num_questions

    # 返回三个指标，用于后续分析
    return average_response_time, average_faithfulness, average_relevancy


# ============================================================================
# RAGEvaluator 类 - 评估不同分块大小对 RAG 系统性能的影响
# ============================================================================
# 定义 RAG 方法的主类

class RAGEvaluator:
    """
    RAG 评估器类，用于系统地评估不同分块大小对 RAG 系统性能的影响。

    这个类的主要功能是：
    1. 加载文档数据
    2. 自动生成评估问题
    3. 使用不同的分块大小进行测试
    4. 比较各分块大小下的响应时间、忠实度和相关性

    为什么需要评估不同的分块大小？
    - 分块大小是 RAG 系统中最重要的参数之一
    - 太小的分块：检索速度快但可能丢失上下文，答案质量低
    - 太大的分块：包含更多上下文但检索速度慢，且可能包含无关信息
    - 最佳分块大小取决于具体的应用场景和文档类型

    使用流程：
    1. 创建 RAGEvaluator 实例，指定文档目录和要测试的分块大小
    2. 调用 run() 方法，自动运行所有测试
    3. 查看输出结果，选择最佳的分块大小
    """

    def __init__(self, data_dir, num_eval_questions, chunk_sizes):
        """
        初始化 RAGEvaluator。

        参数：
            data_dir (str): 文档目录路径。
                           这个目录应该包含要用于评估的 PDF 或其他文档文件。
            num_eval_questions (int): 评估问题数量。
                                     生成的问题越多，评估结果越可靠，但测试时间越长。
            chunk_sizes (list): 要测试的分块大小列表。
                               例如 [128, 256, 512, 1024] 会测试这 4 种分块大小。
        """
        # 保存初始化参数
        self.data_dir = data_dir  # 文档目录
        self.num_eval_questions = num_eval_questions  # 评估问题数量
        self.chunk_sizes = chunk_sizes  # 要测试的分块大小列表

        # 加载文档
        # 调用 load_documents 方法读取指定目录下的所有文档
        self.documents = self.load_documents()
        # 生成评估问题
        # 调用 generate_eval_questions 方法自动生成用于测试的问题
        self.eval_questions = self.generate_eval_questions()

        # 设置 GPT-4o 作为本地配置用于评估
        # GPT-4o 是一个更强大的模型，用于评估答案质量更可靠
        self.llm_gpt4 = OpenAI(model="gpt-4o")
        # 创建忠实度评估器
        self.faithfulness_evaluator = self.create_faithfulness_evaluator()
        # 创建相关性评估器
        self.relevancy_evaluator = self.create_relevancy_evaluator()

    def load_documents(self):
        """
        加载文档。

        使用 SimpleDirectoryReader 读取指定目录下的所有文档。
        SimpleDirectoryReader 支持多种格式，包括 PDF、TXT、DOCX 等。

        返回：
            list: 包含所有加载的文档的列表。
        """
        # SimpleDirectoryReader 会读取 data_dir 目录下的所有文档
        # load_data() 方法加载文档并返回 Document 对象列表
        return SimpleDirectoryReader(self.data_dir).load_data()

    def generate_eval_questions(self):
        """
        生成评估问题。

        这个方法使用 DatasetGenerator 自动从文档中生成问题。
        工作原理：
        1. 取前 20 个文档作为生成问题的源材料
        2. DatasetGenerator 分析文档内容，自动生成相关问题
        3. 从生成的所有问题中随机选择指定数量的问题

        为什么只使用前 20 个文档？
        - 避免生成过多问题，导致测试时间过长
        - 20 个文档通常足以生成足够多的评估问题

        返回：
            list: 随机选择的评估问题列表。
        """
        # 取前 20 个文档用于生成问题
        eval_documents = self.documents[0:20]
        # 从文档创建数据集生成器
        # from_documents 方法分析文档内容，准备生成问题
        data_generator = DatasetGenerator.from_documents(eval_documents)
        # generate_questions_from_nodes 方法自动生成问题
        # 它会分析每个文本节点，生成与该节点内容相关的问题
        eval_questions = data_generator.generate_questions_from_nodes()
        # 从所有生成的问题中随机选择指定数量的问题
        # random.sample 确保不重复选择
        return random.sample(eval_questions, self.num_eval_questions)

    def create_faithfulness_evaluator(self):
        """
        创建忠实度评估器。

        忠实度（Faithfulness）评估的是：生成的答案是否忠实于源文档？
        也就是说，答案中的信息是否都能在源文档中找到依据？

        为什么需要自定义提示模板？
        - 默认的评估提示可能不够精确
        - 自定义提示可以让评估标准更明确
        - 这里的提示要求评估者回答 YES 或 NO，简化判断

        返回：
            FaithfulnessEvaluator: 配置好的忠实度评估器。
        """
        # 使用 GPT-4 创建忠实度评估器
        # 使用更强大的模型可以让评估更准确
        faithfulness_evaluator = FaithfulnessEvaluator(llm=self.llm_gpt4)

        # 定义新的提示模板
        # 这个模板告诉评估器如何判断答案是否忠实
        faithfulness_new_prompt_template = PromptTemplate("""
            请判断给定的信息是否直接由上下文支持。
            你需要回答 YES 或 NO。
            如果上下文的任何部分明确支持该信息，即使大多数上下文无关，也回答 YES。
            如果上下文没有明确支持该信息，回答 NO。以下是一些示例。
            ...
            """)
        # update_prompts 方法用自定义模板替换默认的评估提示
        # "your_prompt_key" 是要替换的提示的键
        faithfulness_evaluator.update_prompts({"your_prompt_key": faithfulness_new_prompt_template})
        return faithfulness_evaluator

    def create_relevancy_evaluator(self):
        """
        创建相关性评估器。

        相关性（Relevancy）评估的是：生成的答案是否与问题相关？
        即使答案是忠实的，但如果答非所问，也是低质量的回答。

        为什么相关性评估器不需要自定义提示？
        - 相关性的定义比较直观：答案是否回应了问题
        - 默认的评估提示通常已经足够好

        返回：
            RelevancyEvaluator: 配置好的相关性评估器。
        """
        # 使用 GPT-4 创建相关性评估器
        return RelevancyEvaluator(llm=self.llm_gpt4)

    def run(self):
        """
        运行评估。

        这个方法会：
        1. 遍历所有指定的分块大小
        2. 对每个分块大小，使用 evaluate_response_time_and_accuracy 函数进行评估
        3. 打印每个分块大小下的平均响应时间、忠实度和相关性

        结果解读：
        - 响应时间：越短越好
        - 忠实度：越高越好（1.0 表示完全忠实）
        - 相关性：越高越好（1.0 表示完全相关）

        选择最佳分块大小的策略：
        - 在忠实度和相关性达到可接受水平的前提下
        - 选择响应时间最短的分块大小
        """
        # 遍历所有要测试的分块大小
        for chunk_size in self.chunk_sizes:
            # 调用评估函数，获取当前分块大小下的三个指标
            avg_response_time, avg_faithfulness, avg_relevancy = evaluate_response_time_and_accuracy(
                chunk_size,              # 当前测试的分块大小
                self.eval_questions,     # 评估问题列表
                self.documents[0:20],    # 用于评估的文档（前 20 个）
                self.faithfulness_evaluator,  # 忠实度评估器
                self.relevancy_evaluator      # 相关性评估器
            )
            # 打印结果
            # .2f 表示保留两位小数
            print(f"分块大小 {chunk_size} - 平均响应时间：{avg_response_time:.2f}秒，"
                  f"平均忠实度：{avg_faithfulness:.2f}，平均相关性：{avg_relevancy:.2f}")


# ============================================================================
# 参数解析函数 - 解析用户在命令行中输入的参数
# ============================================================================
# 参数解析

def parse_args():
    """
    解析命令行参数。

    这个函数使用 argparse 模块来定义和解析命令行参数，
    让用户可以通过命令行来配置程序的行为，而不需要修改代码。

    返回：
        args: 包含所有命令行参数的对象
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='RAG 方法评估')
    # 添加 --data_dir 参数，指定文档目录
    parser.add_argument('--data_dir', type=str, default='../data', help='文档目录')
    # 添加 --num_eval_questions 参数，指定评估问题数量
    parser.add_argument('--num_eval_questions', type=int, default=25, help='评估问题数量')
    # 添加 --chunk_sizes 参数，指定要测试的分块大小列表
    # nargs='+' 表示可以接受多个值，例如 --chunk_sizes 128 256 512
    parser.add_argument('--chunk_sizes', nargs='+', type=int, default=[128, 256], help='分块大小列表')
    return parser.parse_args()


# ============================================================================
# 程序入口 - Python 脚本执行时的起点
# ============================================================================
if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    # 创建 RAGEvaluator 实例
    # 使用命令行参数或默认值来配置评估器
    evaluator = RAGEvaluator(
        data_dir=args.data_dir,           # 文档目录
        num_eval_questions=args.num_eval_questions,  # 评估问题数量
        chunk_sizes=args.chunk_sizes      # 要测试的分块大小列表
    )
    # 运行评估
    # 这会遍历所有分块大小，测试并打印结果
    evaluator.run()
