# ============================================================================
# 导入必要的库和模块
# ============================================================================
import os    # 用于操作系统相关的操作，如文件路径处理
import sys   # 用于操作系统相关的操作，如添加模块搜索路径
from dotenv import load_dotenv  # 用于从.env 文件加载环境变量

# 将父目录添加到路径，以便可以导入上级目录中的模块
# 这行代码让当前脚本可以使用 helper_functions 和 evaluation 模块
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
# 导入自定义的辅助函数模块
from helper_functions import *
# 导入 RAG 评估模块
from evaluation.evalute_rag import *

# 从 .env 文件加载环境变量
# .env 文件是一个存储敏感信息（如 API 密钥）的安全方式
load_dotenv()

# 设置 OpenAI API 密钥环境变量
# 这是调用 OpenAI 服务所必需的认证信息
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


# ============================================================================
# ExplainableRetriever 类 - 可解释的检索器
# ============================================================================
# 定义实用工具类/函数
class ExplainableRetriever:
    """
    可解释检索器：不仅能检索相关信息，还能解释为什么这些信息是相关的。

    传统的 RAG 检索器只返回检索到的文档片段，用户不知道：
    - 为什么这段内容与我的问题相关？
    - 这段内容如何帮助回答我的问题？

    ExplainableRetriever 解决了这个问题：
    1. 检索与查询相关的文档片段
    2. 使用 LLM 分析查询和上下文之间的关系
    3. 生成解释，说明为什么这段内容是相关的

    应用场景：
    - 需要透明度的系统（如医疗、法律）
    - 帮助用户理解检索结果
    - 调试和优化 RAG 系统

    工作原理：
    1. 使用 FAISS 向量存储进行相似性检索
    2. 使用自定义提示模板让 LLM 分析查询与上下文的关系
    3. 返回包含内容和解释的结果
    """

    def __init__(self, texts):
        """
        初始化 ExplainableRetriever。

        参数：
            texts (list): 文本列表。
                         这些是用于检索的源文本，可以是句子、段落或文档。
                         例如：["天空是蓝色的因为...", "光合作用是..."]
        """
        # 初始化 OpenAI 嵌入模型
        # OpenAIEmbeddings() 使用 OpenAI 的嵌入模型将文本转换为向量
        # 这些向量用于计算文本之间的语义相似度
        self.embeddings = OpenAIEmbeddings()

        # 创建 FAISS 向量存储
        # FAISS.from_texts 方法会：
        # 1. 将每个文本转换为向量
        # 2. 将所有向量存入 FAISS 索引中
        # texts 是源文本列表，self.embeddings 是嵌入模型
        self.vectorstore = FAISS.from_texts(texts, self.embeddings)

        # 初始化 ChatOpenAI 语言模型
        # ChatOpenAI 是一个用于与 OpenAI 聊天模型交互的类
        # temperature=0：输出最确定、最聚焦
        # model_name="gpt-4o-mini"：使用快速经济的 gpt-4o-mini 模型
        # max_tokens=4000：限制生成的最大 token 数
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=4000)

        # 从向量存储创建检索器
        # search_kwargs={"k": 5} 指定每次检索返回 5 个最相似的结果
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        # 创建解释提示模板
        # PromptTemplate 用于定义 LLM 的输入格式
        # input_variables=["query", "context"] 指定模板需要两个变量
        explain_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""
            分析以下查询和检索到的上下文之间的关系。
            解释为什么这个上下文与查询相关，以及它如何帮助回答查询。

            查询：{query}

            上下文：{context}

            解释：
            """
        )
        # 将提示模板与 LLM 组合成一个链
        # | 是 Python 的管道运算符，这里用于将提示模板和 LLM 连接起来
        # 这个链可以先用提示模板格式化输入，然后用 LLM 生成输出
        self.explain_chain = explain_prompt | self.llm

    def retrieve_and_explain(self, query):
        """
        检索相关文档并为每个结果生成解释。

        这个方法执行以下步骤：
        1. 使用检索器找到与查询相关的文档
        2. 对每个文档，使用 LLM 生成解释
        3. 返回包含内容和解释的结果列表

        参数：
            query (str): 要检索的查询。
                        这是用户提出的问题，比如"为什么天空是蓝色的？"

        返回：
            list: 包含解释结果的列表，每个结果是一个字典：
                  {
                      "content": "检索到的文本内容",
                      "explanation": "LLM 生成的解释"
                  }
        """
        # 获取与查询相关的文档
        # get_relevant_documents 方法返回一个 Document 对象列表
        # 每个 Document 包含：
        # - page_content: 文本内容
        # - metadata: 元数据（如页码等）
        docs = self.retriever.get_relevant_documents(query)
        # 初始化结果列表
        explained_results = []

        # 遍历每个检索到的文档
        for doc in docs:
            # 准备输入数据
            # query 是用户的查询，context 是文档的内容
            input_data = {"query": query, "context": doc.page_content}
            # 使用解释链生成解释
            # invoke 方法执行链，返回 LLM 的响应
            # .content 提取响应的文本内容
            explanation = self.explain_chain.invoke(input_data).content
            # 将内容和解释添加到结果列表
            explained_results.append({
                "content": doc.page_content,      # 检索到的文本
                "explanation": explanation        # 生成的解释
            })
        # 返回所有解释结果
        return explained_results


# ============================================================================
# ExplainableRAGMethod 类 - 可解释 RAG 方法的包装类
# ============================================================================
class ExplainableRAGMethod:
    """
    可解释 RAG 方法包装类。

    这个类是一个简单的包装器，它将 ExplainableRetriever 封装成一个
    更易于使用的方法类。

    设计目的：
    - 提供统一的接口
    - 简化调用流程
    - 便于扩展和测试
    """

    def __init__(self, texts):
        """
        初始化 ExplainableRAGMethod。

        参数：
            texts (list): 文本列表。
                         这些是用于检索的源文本。
        """
        # 创建一个 ExplainableRetriever 实例
        # texts 是用于检索的源文本列表
        self.explainable_retriever = ExplainableRetriever(texts)

    def run(self, query):
        """
        运行可解释 RAG 查询。

        这个方法接收一个查询，返回带有解释的检索结果。

        参数：
            query (str): 要检索的查询。

        返回：
            list: 包含解释结果的列表。
        """
        # 调用底层检索器的 retrieve_and_explain 方法
        # 这会检索相关文档并为每个结果生成解释
        return self.explainable_retriever.retrieve_and_explain(query)


# ============================================================================
# 参数解析函数 - 解析用户在命令行中输入的参数
# ============================================================================
# 参数解析
def parse_args():
    """
    解析命令行参数。

    这个函数使用 argparse 模块来定义和解析命令行参数，
    让用户可以通过命令行来配置程序的行为。

    返回：
        args: 包含所有命令行参数的对象
    """
    # 导入 argparse 模块（在这里导入是因为它只在函数内部使用）
    import argparse
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="可解释的 RAG 方法")
    # 添加 --query 参数，指定要测试的查询语句
    parser.add_argument('--query', type=str, default='为什么天空是蓝色的？', help="检索器的查询")
    return parser.parse_args()


# ============================================================================
# 程序入口 - Python 脚本执行时的起点
# ============================================================================
if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    # 示例文本（这些可以替换为实际数据）
    # 在实际应用中，这些文本可以来自文档、数据库或其他数据源
    texts = [
        "天空是蓝色的是因为阳光与大气层相互作用的方式。",
        "光合作用是植物利用阳光产生能量的过程。",
        "全球变暖是由地球大气层中温室气体增加引起的。"
    ]

    # 创建 ExplainableRAGMethod 实例
    # 使用上面定义的示例文本初始化
    explainable_rag = ExplainableRAGMethod(texts)
    # 运行查询
    # run 方法会检索相关文本并为每个结果生成解释
    results = explainable_rag.run(args.query)

    # 遍历并打印所有结果
    for i, result in enumerate(results, 1):
        # 打印结果编号
        print(f"结果 {i}:")
        # 打印检索到的内容
        print(f"内容：{result['content']}")
        # 打印 LLM 生成的解释
        print(f"解释：{result['explanation']}")
        # 打印空行分隔不同结果
        print()
