# 导入必要的库和模块
import os  # 用于操作系统相关的操作，如读取环境变量
import sys  # 用于系统特定的参数和函数
from dotenv import load_dotenv  # 用于从.env 文件加载环境变量到系统环境中
from langchain_core.documents import Document  # LangChain 的文档类，用于封装文本内容
from typing import List, Any  # 类型注解工具
from langchain_openai import ChatOpenAI  # OpenAI 的聊天模型
from langchain.chains import RetrievalQA  # LangChain 的检索问答链
from langchain_core.retrievers import BaseRetriever  # LangChain 检索器的基类
from sentence_transformers import CrossEncoder  # 交叉编码器模型，用于文档重排序
from pydantic import BaseModel, Field  # 用于定义结构化输出的数据模型
import argparse  # Python 内置的命令行参数解析库

# 将父目录添加到 Python 路径，这样可以导入上级目录中的模块
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from helper_functions import *  # 导入自定义的辅助函数
from evaluation.evalute_rag import *  # 导入 RAG 评估相关的函数

# 从 .env 文件加载环境变量
# .env 文件通常包含敏感信息如 API 密钥，不应该直接写在代码中
load_dotenv()

# 设置 OpenAI API 密钥环境变量
# 这样后续的 OpenAI 模型调用会自动使用这个密钥进行认证
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


# 辅助类和函数

# 定义用于存储相关性评分的数据模型
class RatingScore(BaseModel):
    """
    相关性评分的数据模型

    用于存储 AI 对文档与查询相关性的打分
    继承自 BaseModel，确保输出格式的一致性
    """
    relevance_score: float = Field(..., description="文档与查询的相关性分数。")
    # 分数范围通常是 1-10，分数越高表示文档与查询越相关


def rerank_documents(query: str, docs: List[Document], top_n: int = 3) -> List[Document]:
    """
    使用 AI 对文档进行重新排序

    什么是重排序（Reranking）？
    - 首先用快速但较粗糙的方法（如向量相似度）检索出较多候选文档（如 30 个）
    - 然后用更精确但更慢的方法（如 AI 评分）对这些文档逐一评估
    - 按评分重新排序，返回最相关的 top N 个文档

    为什么要重排序？
    - 向量检索速度快但精度有限
    - AI 评分更准确但速度慢、成本高
    - 先用向量检索筛选，再用 AI 精排，兼顾速度和精度

    参数：
        query: 用户的查询
        docs: 待排序的文档列表
        top_n: 返回最相关的 top_n 个文档，默认 3 个

    返回：
        按相关性排序后的文档列表
    """
    # 定义评分提示模板，指导 AI 如何评估文档相关性
    prompt_template = PromptTemplate(
        input_variables=["query", "doc"],
        template="""在 1-10 分的范围内，评价以下文档与查询的相关性。考虑查询的具体上下文和意图，而不仅仅是关键词匹配。
        查询：{query}
        文档：{doc}
        相关性分数："""
    )

    # 初始化 LLM 模型
    # temperature=0 确保输出稳定一致
    # model_name="gpt-4o" 使用强大的 GPT-4o 模型
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)

    # 将提示模板和 LLM 组合成链，并指定输出格式为 RatingScore
    llm_chain = prompt_template | llm.with_structured_output(RatingScore)

    scored_docs = []  # 存储（文档，分数）元组的列表
    # 遍历每个文档，让 AI 给出相关性分数
    for doc in docs:
        input_data = {"query": query, "doc": doc.page_content}
        # 调用 AI 链获取评分
        score = llm_chain.invoke(input_data).relevance_score
        try:
            score = float(score)  # 将评分转换为浮点数
        except ValueError:
            score = 0  # 如果解析失败，使用默认分数 0
        scored_docs.append((doc, score))  # 存储文档和对应的分数

    # 按分数从高到低排序
    reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)

    # 返回前 top_n 个文档（只返回文档对象，不返回分数）
    return [doc for doc, _ in reranked_docs[:top_n]]


class CustomRetriever(BaseRetriever, BaseModel):
    """
    自定义检索器：结合向量检索和 AI 重排序

    这个检索器的工作流程：
    1. 使用向量相似度检索出 30 个候选文档
    2. 使用 AI 对这 30 个文档逐一评分
    3. 按评分排序，返回最相关的 2 个文档

    继承自 BaseRetriever 和 BaseModel：
    - BaseRetriever 提供了 LangChain 检索器的标准接口
    - BaseModel 提供了 Pydantic 的数据模型功能
    """
    # 向量存储，用于初始检索
    vectorstore: Any = Field(description="用于初始检索的向量存储")

    # 配置类，允许任意类型的字段
    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str, num_docs=2) -> List[Document]:
        """
        获取与查询相关的文档

        参数：
            query: 用户的查询
            num_docs: 返回的文档数量，默认 2 个

        返回：
            经过重排序的相关文档列表
        """
        # 第一步：使用向量相似度检索 30 个候选文档
        initial_docs = self.vectorstore.similarity_search(query, k=30)

        # 第二步：使用 AI 对文档进行重排序，返回最相关的 num_docs 个文档
        return rerank_documents(query, initial_docs, top_n=num_docs)


class CrossEncoderRetriever(BaseRetriever, BaseModel):
    """
    交叉编码器检索器：使用专门的 CrossEncoder 模型进行重排序

    CrossEncoder 是什么？
    - 是一种专门用于文本匹配任务的深度学习模型
    - 与普通的向量检索（Bi-Encoder）不同，CrossEncoder 会同时看到查询和文档
    - 能捕捉更精细的交互，精度更高但速度较慢

    与 CustomRetriever 的区别：
    - CustomRetriever 使用 LLM（如 GPT-4）进行评分
    - CrossEncoderRetriever 使用专门的 CrossEncoder 模型进行评分
    - CrossEncoder 更快、更便宜，但需要预先训练
    """
    # 向量存储，用于初始检索
    vectorstore: Any = Field(description="用于初始检索的向量存储")
    # 交叉编码器模型，用于重排序
    cross_encoder: Any = Field(description="用于重新排序的交叉编码器模型")
    # 初始检索的文档数量
    k: int = Field(default=5, description="初始检索的文档数量")
    # 重排序后返回的文档数量
    rerank_top_k: int = Field(default=3, description="重新排序后返回的文档数量")

    # 配置类，允许任意类型的字段
    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        使用 CrossEncoder 获取相关文档

        参数：
            query: 用户的查询

        返回：
            经过 CrossEncoder 重排序的相关文档列表
        """
        # 第一步：使用向量相似度检索 k 个候选文档
        initial_docs = self.vectorstore.similarity_search(query, k=self.k)

        # 第二步：准备 CrossEncoder 的输入
        # CrossEncoder 需要 [query, document] 格式的输入对
        pairs = [[query, doc.page_content] for doc in initial_docs]

        # 第三步：使用 CrossEncoder 预测每对的相关性分数
        scores = self.cross_encoder.predict(pairs)

        # 第四步：将文档与分数配对，并按分数排序
        scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)

        # 返回重排序后的前 rerank_top_k 个文档
        return [doc for doc, _ in scored_docs[:self.rerank_top_k]]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """
        异步获取相关文档（未实现）

        这是一个异步方法，用于异步场景
        当前未实现，抛出 NotImplementedError 异常
        """
        raise NotImplementedError("异步检索未实现")


def compare_rag_techniques(query: str, docs: List[Document]) -> None:
    """
    比较不同检索技术的结果

    这个函数演示了基础检索和高级重排序检索的区别：
    1. 基线检索：直接使用向量相似度检索
    2. 高级检索：使用自定义检索器（向量检索 + AI 重排序）

    参数：
        query: 测试查询
        docs: 测试文档列表
    """
    # 创建嵌入模型和向量存储
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    print("检索技术比较")
    print("==================================")
    print(f"查询：{query}\n")

    # 基线检索：直接使用向量相似度
    print("基线检索结果:")
    baseline_docs = vectorstore.similarity_search(query, k=2)
    for i, doc in enumerate(baseline_docs):
        print(f"\n文档 {i + 1}:")
        print(doc.page_content)

    # 高级检索：使用自定义检索器（带重排序）
    print("\n高级检索结果:")
    custom_retriever = CustomRetriever(vectorstore=vectorstore)
    advanced_docs = custom_retriever.get_relevant_documents(query)
    for i, doc in enumerate(advanced_docs):
        print(f"\n文档 {i + 1}:")
        print(doc.page_content)


# 主类
class RAGPipeline:
    """
    RAG 流水线：整合文档处理和问答功能

    这个类封装了完整的 RAG（检索增强生成）流程：
    1. 加载和处理 PDF 文档
    2. 创建向量存储
    3. 根据选择的检索器类型创建检索器（支持重排序和 CrossEncoder）
    4. 使用检索到的文档回答用户问题

    使用流程：
    1. 初始化时传入 PDF 文件路径
    2. 调用 run 方法传入查询和检索器类型，获取答案
    """
    def __init__(self, path: str):
        """
        初始化 RAG 流水线

        参数：
            path: PDF 文档的路径
        """
        # 处理 PDF 并创建向量存储
        # encode_pdf 是 helper_functions 中定义的辅助函数
        self.vectorstore = encode_pdf(path)

        # 初始化用于生成答案的 LLM
        # temperature=0 确保输出稳定，model_name="gpt-4o" 使用强大的 GPT-4 模型
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

    def run(self, query: str, retriever_type: str = "reranker"):
        """
        运行 RAG 流水线，回答用户问题

        参数：
            query: 用户的问题
            retriever_type: 检索器类型，可选 "reranker" 或 "cross_encoder"
                - reranker: 使用 AI 进行重排序
                - cross_encoder: 使用 CrossEncoder 模型进行重排序
        """
        # 根据选择的类型创建检索器
        if retriever_type == "reranker":
            # 使用自定义检索器（AI 重排序）
            retriever = CustomRetriever(vectorstore=self.vectorstore)
        elif retriever_type == "cross_encoder":
            # 使用 CrossEncoder 检索器
            # 'cross-encoder/ms-marco-MiniLM-L-6-v2' 是一个预训练的 CrossEncoder 模型
            cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            retriever = CrossEncoderRetriever(
                vectorstore=self.vectorstore,
                cross_encoder=cross_encoder,
                k=10,  # 初始检索 10 个文档
                rerank_top_k=5  # 重排序后返回 5 个
            )
        else:
            # 未知的检索器类型，抛出异常
            raise ValueError("未知的检索器类型。使用 'reranker' 或 'cross_encoder'。")

        # 创建检索问答链
        # RetrievalQA 是 LangChain 提供的完整 RAG 流程封装
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,  # 用于生成答案的语言模型
            chain_type="stuff",  # "stuff" 表示将所有文档内容拼接到提示中
            retriever=retriever,  # 使用的检索器
            return_source_documents=True  # 返回源文档以便查看
        )

        # 执行问答链，获取结果
        result = qa_chain({"query": query})

        # 打印结果
        print(f"\n问题：{query}")
        print(f"答案：{result['result']}")
        print("\n相关源文档:")
        for i, doc in enumerate(result["source_documents"]):
            print(f"\n文档 {i + 1}:")
            # 只显示前 200 个字符，避免输出过长
            print(doc.page_content[:200] + "...")


# 参数解析
def parse_args():
    """
    解析命令行参数

    允许用户通过命令行传递参数来运行脚本
    例如：python reranking.py --path ./book.pdf --query "什么是气候变化" --retriever_type reranker

    返回：
        包含解析后参数的对象
    """
    parser = argparse.ArgumentParser(description="RAG 流水线")
    parser.add_argument("--path", type=str, default="../data/Understanding_Climate_Change.pdf", help="文档路径")
    parser.add_argument("--query", type=str, default='What are the impacts of climate change?', help="要询问的查询")
    parser.add_argument("--retriever_type", type=str, default="reranker", choices=["reranker", "cross_encoder"],
                        help="要使用的检索器类型")
    return parser.parse_args()


if __name__ == "__main__":
    """
    程序主入口

    当直接运行这个脚本时（而不是作为模块导入），会执行这里的代码
    """
    # 解析命令行参数
    args = parse_args()

    # 创建 RAG 流水线实例
    pipeline = RAGPipeline(path=args.path)

    # 运行流水线，回答用户问题
    pipeline.run(query=args.query, retriever_type=args.retriever_type)

    # 演示重新排序的比较
    # 这部分代码展示了为什么要使用重排序
    # 通过对比基础检索和重排序检索的结果，可以看出重排序的优势
    chunks = [
        "The capital of France is great.",
        "The capital of France is huge.",
        "The capital of France is beautiful.",
        """Have you ever visited Paris? It is a beautiful city where you can eat delicious food and see the Eiffel Tower.
        I really enjoyed all the cities in France, but its capital with the Eiffel Tower is my favorite city.""",
        "I really enjoyed my trip to Paris, France. The city is beautiful and the food is delicious. I would love to visit again. Such a great capital city."
    ]
    # 将字符串列表转换为 Document 对象列表
    docs = [Document(page_content=sentence) for sentence in chunks]

    # 运行比较函数，展示两种检索方法的差异
    # 查询是"法国的首都是哪里"，看看哪种方法能更好地找到包含答案的文档
    compare_rag_techniques(query="what is the capital of france?", docs=docs)
