# ============================================================================
# 导入必要的库和模块
# ============================================================================
import os
import sys
import argparse
from dotenv import load_dotenv

# 将父目录添加到路径，因为我们使用 notebooks
# 这样可以导入上级目录中的 helper_functions 等模块
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from helper_functions import *
from evaluation.evalute_rag import *

# 从 .env 文件加载环境变量
# .env 文件通常包含敏感信息如 API 密钥，不应提交到版本控制
load_dotenv()

# 设置 OpenAI API 密钥环境变量
# 后续使用 OpenAI 服务时会自动读取这个环境变量
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


# ============================================================================
# HyDe（假设文档嵌入）检索器类
# ============================================================================
# HyDe 的核心思想是：先用 AI 生成一个"假设的理想答案"，
# 然后用这个假设答案去向量库中搜索相似的真实文档
# 这样做的好处是可以克服用户查询简短、向量库中的文档较长导致的语义不匹配问题
class HyDERetriever:
    """
    HyDe 检索器：通过生成假设文档来提升检索效果

    工作原理：
    1. 用户提出问题
    2. AI 生成一个假设的"理想答案文档"
    3. 用这个假设文档在向量库中搜索相似的真实文档
    4. 返回最相关的真实文档作为检索结果
    """

    def __init__(self, files_path, chunk_size=500, chunk_overlap=100):
        """
        初始化 HyDe 检索器

        参数：
            files_path: PDF 文件路径，用于构建向量知识库
            chunk_size: 文本块大小，控制每个文档片段的长度（字符数）
            chunk_overlap: 块重叠大小，相邻文本块之间的重叠部分，避免信息被切断
        """
        # 初始化大语言模型，用于生成假设文档
        # temperature=0 让输出更稳定、确定性更高
        # gpt-4o-mini 是 OpenAI 的轻量级模型，速度快、成本低
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=4000)

        # 初始化嵌入模型，用于将文本转换为向量
        # 向量是文本的数学表示，用于计算文本之间的相似度
        self.embeddings = OpenAIEmbeddings()

        # 保存文本分块参数
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 构建向量存储
        # encode_pdf 函数会：读取 PDF -> 分割成块 -> 转换为向量 -> 存储到向量数据库
        self.vectorstore = encode_pdf(files_path, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

        # 定义 HyDe 提示模板
        # 这个模板告诉 AI：根据用户问题，生成一个详细的假设答案
        self.hyde_prompt = PromptTemplate(
            input_variables=["query", "chunk_size"],  # 模板需要的变量
            template="""给定问题 '{query}'，生成一个直接回答此问题的假设文档。该文档应该详细且深入。
            文档大小必须恰好是 {chunk_size} 个字符。""",
        )
        # 将提示模板和 LLM 连接成一个处理链
        # 输入问题 -> 提示模板 -> LLM -> 输出假设文档
        self.hyde_chain = self.hyde_prompt | self.llm

    def generate_hypothetical_document(self, query):
        """
        根据用户查询生成假设文档

        参数：
            query: 用户的问题

        返回：
            假设文档的文本内容

        举例：
            用户问："气候变化的原因是什么？"
            AI 生成的假设文档可能是："气候变化的主要原因是人类活动，包括燃烧化石燃料、
             deforestation 等，这些活动释放了大量温室气体..."
        """
        # 准备输入变量
        input_variables = {"query": query, "chunk_size": self.chunk_size}
        # 调用处理链生成假设文档，.content 提取纯文本
        return self.hyde_chain.invoke(input_variables).content

    def retrieve(self, query, k=3):
        """
        执行检索操作

        参数：
            query: 用户查询
            k: 返回最相关的 k 个文档

        返回：
            similar_docs: 最相关的 k 个真实文档
            hypothetical_doc: 生成的假设文档
        """
        # 第一步：生成假设文档
        hypothetical_doc = self.generate_hypothetical_document(query)
        # 第二步：用假设文档在向量库中搜索相似的真实文档
        # similarity_search 会计算向量相似度，返回最接近的 k 个文档
        similar_docs = self.vectorstore.similarity_search(hypothetical_doc, k=k)
        return similar_docs, hypothetical_doc


# ============================================================================
# 气候变化 RAG 主类 - 用于演示和运行检索过程
# ============================================================================
class ClimateChangeRAG:
    """
    气候变化 RAG（检索增强生成）系统的主类

    这个类封装了完整的 RAG 流程：
    1. 加载 PDF 文档并构建知识库
    2. 使用 HyDe 技术进行检索
    3. 展示检索结果
    """

    def __init__(self, path, query):
        """
        初始化 RAG 系统

        参数：
            path: PDF 文件路径
            query: 用户要查询的问题
        """
        # 创建 HyDe 检索器实例
        self.retriever = HyDERetriever(path)
        # 保存用户查询
        self.query = query

    def run(self):
        """
        运行完整的 RAG 检索流程并输出结果
        """
        # 检索结果和假设文档
        # results 包含最相关的文档列表，hypothetical_doc 是 AI 生成的假设答案
        results, hypothetical_doc = self.retriever.retrieve(self.query)

        # 提取文档内容，方便展示
        # doc.page_content 是每个文档的文本内容
        docs_content = [doc.page_content for doc in results]

        # 打印假设文档（换行格式化）
        print("假设文档:\n")
        # text_wrap 函数用于格式化长文本，使其更易读
        print(text_wrap(hypothetical_doc) + "\n")

        # 展示检索到的真实文档
        show_context(docs_content)


# ============================================================================
# 命令行参数解析函数
# ============================================================================
def parse_args():
    """
    解析命令行参数

    允许用户通过命令行指定：
    - 要处理的 PDF 文件路径
    - 要查询的问题

    返回：
        包含所有参数的命名空间对象
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="运行气候变化 RAG 方法。")

    # 添加路径参数
    parser.add_argument("--path", type=str, default="../data/Understanding_Climate_Change.pdf",
                        help="要处理的 PDF 文件路径。")

    # 添加查询参数
    parser.add_argument("--query", type=str, default="气候变化的主要原因是什么？",
                        help="测试检索器的查询（默认：'文档的主题是什么？'）。")

    # 解析并返回参数
    return parser.parse_args()


# ============================================================================
# 程序主入口
# ============================================================================
if __name__ == "__main__":
    # 解析命令行参数
    # args.path 和 args.query 分别对应传入的路径和查询
    args = parse_args()

    # 创建 RAG 系统实例
    rag_runner = ClimateChangeRAG(args.path, args.query)

    # 运行 RAG 系统，输出检索结果
    rag_runner.run()
