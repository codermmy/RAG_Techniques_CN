# 导入必要的库和模块
import os  # 用于操作系统相关的操作，如读取环境变量
from dotenv import load_dotenv  # 用于从.env 文件加载环境变量到系统环境中
from langchain_openai import ChatOpenAI  # OpenAI 的聊天模型
from langchain_core.prompts import PromptTemplate  # LangChain 的提示模板

# 从 .env 文件加载环境变量
# .env 文件通常包含敏感信息如 API 密钥，不应该直接写在代码中
load_dotenv()

# 设置 OpenAI API 密钥环境变量
# 这样后续的 OpenAI 模型调用会自动使用这个密钥进行认证
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


# 用于重写查询以改进检索的函数
def rewrite_query(original_query, llm_chain):
    """
    重写原始查询以改进检索。

    什么是查询重写（Query Rewrite）？
    - 用户的问题可能过于简单或模糊，直接检索效果不好
    - 查询重写是将用户的原始问题改写得更加具体、详细
    - 例如："气候变化" -> "气候变化对环境、经济和社会的影响"

    为什么要重写查询？
    - 更具体的查询能检索到更相关的文档
    - 可以补充用户问题中缺失的关键信息
    - 提高检索的准确性和召回率

    Args:
    original_query (str): 原始用户查询
    llm_chain: 用于生成重写查询的链（提示模板+LLM）

    Returns:
    str: 重写后的查询，更加具体和详细
    """
    # 调用 LLM 链来处理原始查询，生成重写后的查询
    response = llm_chain.invoke(original_query)
    return response.content


# 用于生成回溯查询以获取更广泛上下文的函数
def generate_step_back_query(original_query, llm_chain):
    """
    生成回溯查询以获取更广泛的上下文。

    什么是回溯查询（Step-back Query）？
    - 与查询重写相反，回溯是生成一个更一般、更宽泛的问题
    - 例如："地球与太阳的距离如何影响气候" -> "影响气候的因素有哪些"

    为什么要生成回溯查询？
    - 有时需要更广泛的背景信息来回答具体问题
    - 可以检索到相关的上下文知识
    - 特别适用于需要背景知识的复杂问题

    Args:
    original_query (str): 原始用户查询
    llm_chain: 用于生成回溯查询的链（提示模板+LLM）

    Returns:
    str: 回溯查询，更加一般和宽泛
    """
    # 调用 LLM 链来处理原始查询，生成回溯查询
    response = llm_chain.invoke(original_query)
    return response.content


# 用于将查询分解为更简单的子查询的函数
def decompose_query(original_query, llm_chain):
    """
    将原始查询分解为更简单的子查询。

    什么是查询分解（Query Decomposition）？
    - 将复杂问题拆分成多个简单的子问题
    - 例如："气候变化对环境有什么影响" 拆分为：
      1. 气候变化对生物多样性有什么影响？
      2. 气候变化如何影响海洋？
      3. 气候变化对农业有什么影响？
      4. 气候变化对人类健康有什么影响？

    为什么要分解查询？
    - 复杂问题可能涉及多个方面，单一检索可能遗漏重要信息
    - 子问题更容易检索到精准的答案
    - 可以分别检索每个子问题，然后整合答案

    Args:
    original_query (str): 原始复杂查询
    llm_chain: 用于生成子查询的链（提示模板+LLM）

    Returns:
    List[str]: 更简单的子查询列表，每个子查询针对原问题的一个方面
    """
    # 调用 LLM 链来处理原始查询，获取分解后的内容
    response = llm_chain.invoke(original_query).content

    # 解析响应，提取子查询列表
    # 按行分割，过滤空行和标题行
    sub_queries = [q.strip() for q in response.split('\n') if q.strip() and not q.strip().startswith('Sub-queries:')]
    return sub_queries


# RAG 方法的主类
class RAGQueryProcessor:
    """
    RAG 查询处理器：整合多种查询转换技术

    这个类提供了三种查询转换方法：
    1. 查询重写（Rewrite）：将问题改写得更加具体详细
    2. 回溯查询（Step-back）：生成更宽泛的问题获取背景知识
    3. 查询分解（Decompose）：将复杂问题拆分为多个简单子问题

    使用流程：
    1. 初始化时会自动创建三个 LLM 链，分别用于三种转换
    2. 调用 run 方法传入原始查询，会展示三种转换的结果
    """
    def __init__(self):
        # 初始化三个 LLM 模型，分别用于三种查询转换
        # temperature=0 确保输出稳定一致
        # model_name="gpt-4o" 使用强大的 GPT-4o 模型
        self.re_write_llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
        self.step_back_llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
        self.sub_query_llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)

        # 初始化提示模板
        # 提示模板告诉 AI 如何执行每种查询转换任务
        query_rewrite_template = """你是一个 AI 助手，任务是重新表述用户查询以改进 RAG 系统中的检索。
        给定原始查询，将其重写为更具体、更详细且更可能检索到相关信息的查询。

        原始查询：{original_query}

        重写后的查询："""
        step_back_template = """你是一个 AI 助手，任务是生成更广泛、更一般的查询以改进 RAG 系统中的上下文检索。
        给定原始查询，生成一个更一般的回溯查询，帮助检索相关的背景信息。

        原始查询：{original_query}

        回溯查询："""
        subquery_decomposition_template = """你是一个 AI 助手，任务是将复杂查询分解为 RAG 系统的更简单子查询。
        给定原始查询，将其分解为 2-4 个更简单的子查询，当这些子查询一起回答时，可以提供对原始查询的全面回答。

        原始查询：{original_query}

        示例：气候变化对环境有什么影响？

        子查询：
        1. 气候变化对生物多样性有什么影响？
        2. 气候变化如何影响海洋？
        3. 气候变化对农业有什么影响？
        4. 气候变化对人类健康有什么影响？"""

        # 创建 LLMChains
        # 将提示模板和对应的 LLM 组合成链
        # PromptTemplate 定义了输入格式，| 操作符将模板和 LLM 连接起来
        self.query_rewriter = PromptTemplate(input_variables=["original_query"],
                                             template=query_rewrite_template) | self.re_write_llm
        self.step_back_chain = PromptTemplate(input_variables=["original_query"],
                                              template=step_back_template) | self.step_back_llm
        self.subquery_decomposer_chain = PromptTemplate(input_variables=["original_query"],
                                                        template=subquery_decomposition_template) | self.sub_query_llm

    def run(self, original_query):
        """
        运行完整的 RAG 查询处理流水线。

        这个方法会展示三种查询转换技术：
        1. 重写查询：让问题更具体
        2. 生成回溯查询：获取更广泛的背景
        3. 分解查询：拆分为多个子问题

        Args:
        original_query (str): 要处理的原始查询
        """
        # 第一步：重写查询
        # 使用 query_rewriter 链将原始查询改写得更加具体
        rewritten_query = rewrite_query(original_query, self.query_rewriter)
        print("原始查询:", original_query)
        print("\n重写后的查询:", rewritten_query)

        # 第二步：生成回溯查询
        # 使用 step_back_chain 生成更宽泛的查询，用于获取背景知识
        step_back_query = generate_step_back_query(original_query, self.step_back_chain)
        print("\n回溯查询:", step_back_query)

        # 第三步：将查询分解为子查询
        # 使用 subquery_decomposer_chain 将复杂问题拆分为多个简单子问题
        sub_queries = decompose_query(original_query, self.subquery_decomposer_chain)
        print("\n子查询:")
        # 逐个打印每个子查询
        for i, sub_query in enumerate(sub_queries, 1):
            print(f"{i}. {sub_query}")


# 参数解析
def parse_args():
    """
    解析命令行参数

    允许用户通过命令行传递参数来运行脚本
    例如：python query_transformations.py --query "什么是气候变化"

    返回：
        包含解析后参数的对象
    """
    import argparse  # Python 内置的命令行参数解析库
    parser = argparse.ArgumentParser(description="使用 RAG 方法处理查询。")
    parser.add_argument("--query", type=str, default='What are the impacts of climate change on the environment?',
                        help="要处理的原始查询")
    return parser.parse_args()


# 主执行入口
if __name__ == "__main__":
    """
    程序主入口

    当直接运行这个脚本时（而不是作为模块导入），会执行这里的代码
    """
    # 解析命令行参数
    args = parse_args()

    # 创建 RAG 查询处理器实例
    processor = RAGQueryProcessor()

    # 运行查询处理流水线，展示三种转换方法的结果
    processor.run(args.query)
