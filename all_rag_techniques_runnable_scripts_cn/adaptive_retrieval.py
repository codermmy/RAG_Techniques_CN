# 导入必要的库和模块
import os  # 用于操作系统相关的操作，如读取环境变量
import sys  # 用于系统特定的参数和函数
from dotenv import load_dotenv  # 用于从.env 文件加载环境变量到系统环境中
from langchain_core.prompts import PromptTemplate  # LangChain 的提示模板，用于定义 AI 输入的格式
from langchain_community.vectorstores import FAISS  # Facebook 的高效相似度搜索库，用于存储和检索文档向量
from langchain_openai import OpenAIEmbeddings  # OpenAI 的文本嵌入模型，将文本转换为向量
from langchain_text_splitters import CharacterTextSplitter  # 文本分割器，按字符分割长文本

from langchain_core.retrievers import BaseRetriever  # LangChain 检索器的基类
from typing import List, Dict, Any  # 类型注解工具
from langchain.docstore.document import Document  # LangChain 的文档类，用于封装文本内容
from langchain_openai import ChatOpenAI  # OpenAI 的聊天模型
from langchain_core.pydantic_v1 import BaseModel, Field  # 用于定义结构化输出的数据模型

# 将父目录添加到 Python 路径，这样可以导入上级目录中的模块
# 因为我们使用的是 notebook 项目结构，需要访问上级目录的 helper_functions 等模块
sys.path.append(os.path.abspath(
    os.path.join(os.getcwd(), '..')))
from helper_functions import *  # 导入自定义的辅助函数
from evaluation.evalute_rag import *  # 导入 RAG 评估相关的函数

# 从 .env 文件加载环境变量
# .env 文件通常包含敏感信息如 API 密钥，不应该直接写在代码中
load_dotenv()

# 设置 OpenAI API 密钥环境变量
# 这样后续的 OpenAI 模型调用会自动使用这个密钥进行认证
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


# 定义所有需要的类和策略

# 下面是四个数据模型类，用于定义 AI 模型的结构化输出格式
# 使用 Pydantic BaseModel 可以确保 AI 返回的数据符合我们定义的格式

class CategoriesOptions(BaseModel):
    """
    查询类别分类结果的数据模型

    这个类定义了查询分类的输出格式，AI 会返回一个包含 category 字段的对象
    继承自 BaseModel，可以确保输出格式的一致性
    """
    category: str = Field(
        description="查询的类别，可选项有：Factual（事实性）、Analytical（分析性）、Opinion（观点性）或 Contextual（上下文相关）",
        example="Factual"
        # Factual = 事实性问题，如"地球和太阳的距离是多少"
        # Analytical = 分析性问题，如"地球与太阳的距离如何影响气候"
        # Opinion = 观点性问题，如"关于地球形成的不同理论有哪些"
        # Contextual = 需要上下文的问题，如"根据前面的信息，地球的位置如何"
    )


class RelevantScore(BaseModel):
    """
    相关性评分的数据模型

    用于存储文档与查询的相关性得分，AI 会返回一个 0-10 之间的分数
    分数越高表示文档与查询越相关
    """
    score: float = Field(description="文档与查询的相关性得分", example=8.0)
    # 得分范围通常是 1-10，10 分表示最相关


class SelectedIndices(BaseModel):
    """
    选中文档索引的数据模型

    用于存储从多个候选文档中选择出的文档索引列表
    例如从 10 个文档中选择最相关的 3 个，返回它们的索引 [0, 2, 5]
    """
    indices: List[int] = Field(description="选中文档的索引列表", example=[0, 1, 2, 3])


class SubQueries(BaseModel):
    """
    子查询列表的数据模型

    用于将一个复杂查询分解为多个简单的子查询
    例如将"地球的位置如何影响其气候"分解为多个子问题分别检索
    """
    sub_queries: List[str] = Field(description="用于综合分析的子查询列表",
                                   example=["What is the population of New York?", "What is the GDP of New York?"])


class QueryClassifier:
    """
    查询分类器：负责分析用户提出的问题，判断其属于哪种类型

    为什么要分类？因为不同类型的问题需要不同的检索策略
    比如事实性问题需要精确匹配，而观点性问题需要收集不同视角
    """
    def __init__(self):
        # 初始化时使用 GPT-4o 模型，temperature=0 确保输出稳定一致
        # max_tokens=4000 设置最大输出长度
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)

        # 定义分类提示模板，告诉 AI 如何将查询分类到四个类别中
        self.prompt = PromptTemplate(
            input_variables=["query"],  # 输入变量是 query（用户的问题）
            template="将以下查询分类为以下类别之一：Factual（事实性）、Analytical（分析性）、Opinion（观点性）或 Contextual（上下文相关）。\n查询：{query}\n类别："
        )
        # 将提示模板和 LLM 组合成一个链，并指定输出格式为 CategoriesOptions
        # with_structured_output 确保 AI 返回我们定义的结构化数据
        self.chain = self.prompt | self.llm.with_structured_output(CategoriesOptions)

    def classify(self, query):
        """
        对用户查询进行分类

        参数：
            query: 用户提出的问题

        返回：
            查询的类别（Factual/Analytical/Opinion/Contextual）
        """
        print("正在分类查询...")
        # 调用链来处理查询，返回结构化的分类结果
        return self.chain.invoke(query).category


class BaseRetrievalStrategy:
    """
    基础检索策略类：所有具体检索策略的父类

    这个类定义了检索器的基本功能：
    1. 将文本分割成小块
    2. 使用嵌入模型将文本转换为向量
    3. 存储到 FAISS 向量数据库中
    4. 提供基础的相似度搜索功能
    """
    def __init__(self, texts):
        # 初始化 OpenAI 嵌入模型，用于将文本转换为向量表示
        # 向量是一种数学表示，相似的文本会有相近的向量
        self.embeddings = OpenAIEmbeddings()

        # 创建文本分割器：设置每块 800 字符，块之间不重叠
        # 分割文本是因为 AI 模型有输入长度限制，小块更容易处理
        text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=0)
        # 将输入文本列表分割成文档对象
        self.documents = text_splitter.create_documents(texts)
        # 使用 FAISS 创建向量数据库，存储文档及其向量表示
        self.db = FAISS.from_documents(self.documents, self.embeddings)

        # 初始化 LLM 用于后续的查询处理和文档评分
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)

    def retrieve(self, query, k=4):
        """
        基础检索方法：使用向量相似度搜索

        参数：
            query: 搜索查询
            k: 返回最相关的 k 个文档

        返回：
            最相关的 k 个文档
        """
        # similarity_search 是 FAISS 的核心方法，找到与查询向量最接近的文档向量
        return self.db.similarity_search(query, k=k)


class FactualRetrievalStrategy(BaseRetrievalStrategy):
    """
    事实性检索策略：专门用于检索事实性信息

    事实性问题的特点：
    - 有明确的、客观的答案
    - 如"地球和太阳之间的距离是多少"
    - 需要精确匹配和高度相关的文档

    策略步骤：
    1. 使用 AI 增强原始查询，使其更具体
    2. 检索更多候选文档（k*2）
    3. 使用 AI 对每个文档进行相关性评分
    4. 按评分排序，返回得分最高的 k 个文档
    """
    def retrieve(self, query, k=4):
        """
        执行事实性检索

        参数：
            query: 用户的查询
            k: 返回的文档数量

        返回：
            经过评分排序的最相关文档列表
        """
        print("正在检索事实信息...")

        # 第一步：增强查询
        # 使用 AI 将原始查询改写得更加明确和具体，有助于检索到更相关的文档
        enhanced_query_prompt = PromptTemplate(
            input_variables=["query"],
            template="增强此事实性查询以获得更好的信息检索效果：{query}"
        )
        query_chain = enhanced_query_prompt | self.llm
        enhanced_query = query_chain.invoke(query).content
        print(f'增强后的查询：{enhanced_query}')

        # 第二步：使用增强后的查询检索更多候选文档（k*2 个）
        docs = self.db.similarity_search(enhanced_query, k=k * 2)

        # 第三步：使用 AI 对每个文档进行相关性评分
        ranking_prompt = PromptTemplate(
            input_variables=["query", "doc"],
            template="在 1-10 的评分范围内，此文档与查询 '{query}' 的相关性如何？\n文档：{doc}\n相关性得分："
        )
        ranking_chain = ranking_prompt | self.llm.with_structured_output(RelevantScore)

        ranked_docs = []
        print("正在对文档进行排序...")
        # 遍历每个文档，让 AI 给出相关性分数
        for doc in docs:
            input_data = {"query": enhanced_query, "doc": doc.page_content}
            score = float(ranking_chain.invoke(input_data).score)
            ranked_docs.append((doc, score))  # 存储文档和对应的分数

        # 第四步：按分数从高到低排序
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        # 返回前 k 个文档（只返回文档对象，不返回分数）
        return [doc for doc, _ in ranked_docs[:k]]


class AnalyticalRetrievalStrategy(BaseRetrievalStrategy):
    """
    分析性检索策略：专门用于检索分析性信息

    分析性问题的特点：
    - 需要分析、比较、推理
    - 如"地球与太阳的距离如何影响其气候"
    - 需要从多个角度收集信息

    策略步骤：
    1. 将复杂查询分解为多个子查询
    2. 对每个子查询分别检索文档
    3. 使用 AI 选择最多样化和最相关的文档
    """
    def retrieve(self, query, k=4):
        """
        执行分析性检索

        参数：
            query: 用户的查询
            k: 返回的文档数量

        返回：
            多样化且相关的文档列表
        """
        print("正在检索分析信息...")

        # 第一步：生成子查询
        # 将复杂问题分解为多个简单的子问题，从不同角度检索信息
        sub_queries_prompt = PromptTemplate(
            input_variables=["query", "k"],
            template="为以下查询生成 {k} 个子问题：{query}"
        )
        sub_queries_chain = sub_queries_prompt | self.llm.with_structured_output(SubQueries)
        input_data = {"query": query, "k": k}
        sub_queries = sub_queries_chain.invoke(input_data).sub_queries
        print(f'子查询：{sub_queries}')

        # 第二步：对每个子查询分别检索，收集所有相关文档
        all_docs = []
        for sub_query in sub_queries:
            all_docs.extend(self.db.similarity_search(sub_query, k=2))  # 每个子查询检索 2 个文档

        # 第三步：从所有文档中选择最多样化和最相关的 k 个文档
        # 多样性确保覆盖不同的角度和信息源
        diversity_prompt = PromptTemplate(
            input_variables=["query", "docs", "k"],
            template="为查询 '{query}' 选择最多样化和最相关的 {k} 个文档\n文档：{docs}\n"
        )
        diversity_chain = diversity_prompt | self.llm.with_structured_output(SelectedIndices)

        # 将文档格式化为字符串，只显示前 50 个字符以保持简洁
        docs_text = "\n".join([f"{i}: {doc.page_content[:50]}..." for i, doc in enumerate(all_docs)])
        input_data = {"query": query, "docs": docs_text, "k": k}
        # AI 返回选中的文档索引列表
        selected_indices = diversity_chain.invoke(input_data).indices

        # 根据索引返回选中的文档，检查索引不越界
        return [all_docs[i] for i in selected_indices if i < len(all_docs)]


class OpinionRetrievalStrategy(BaseRetrievalStrategy):
    """
    观点性检索策略：专门用于检索不同观点和视角

    观点性问题的特点：
    - 没有唯一正确答案
    - 如"关于地球上生命起源的不同理论有哪些"
    - 需要收集多种不同的观点和立场

    策略步骤：
    1. 识别关于主题的不同观点/视角
    2. 针对每个观点分别检索文档
    3. 使用 AI 选择最具代表性和多样性的观点文档
    """
    def retrieve(self, query, k=3):
        """
        执行观点性检索

        参数：
            query: 用户的查询
            k: 返回的文档数量（默认 3 个，因为观点通常不需要太多）

        返回：
            代表不同观点的文档列表
        """
        print("正在检索观点...")

        # 第一步：识别不同的观点
        # 让 AI 列出关于这个主题的 k 个不同观点或视角
        viewpoints_prompt = PromptTemplate(
            input_variables=["query", "k"],
            template="识别关于主题 '{query}' 的 {k} 个不同观点或视角"
        )
        viewpoints_chain = viewpoints_prompt | self.llm
        input_data = {"query": query, "k": k}
        # 调用 AI 获取观点列表，按行分割
        viewpoints = viewpoints_chain.invoke(input_data).content.split('\n')
        print(f'观点：{viewpoints}')

        # 第二步：针对每个观点分别检索文档
        # 将原始查询与每个观点结合，检索支持该观点的文档
        all_docs = []
        for viewpoint in viewpoints:
            all_docs.extend(self.db.similarity_search(f"{query} {viewpoint}", k=2))

        # 第三步：选择最具代表性和多样性的观点文档
        opinion_prompt = PromptTemplate(
            input_variables=["query", "docs", "k"],
            template="将这些文档分类为关于 '{query}' 的不同观点，并选择 {k} 个最具代表性和多样性的观点：\n文档：{docs}\n选中索引："
        )
        opinion_chain = opinion_prompt | self.llm.with_structured_output(SelectedIndices)

        # 将文档格式化为字符串，显示前 100 个字符以便 AI 更好判断
        docs_text = "\n".join([f"{i}: {doc.page_content[:100]}..." for i, doc in enumerate(all_docs)])
        input_data = {"query": query, "docs": docs_text, "k": k}
        selected_indices = opinion_chain.invoke(input_data).indices

        # 根据索引返回选中的文档
        # 检查索引是否为有效数字且不越界
        return [all_docs[int(i)] for i in selected_indices if i.isdigit() and int(i) < len(all_docs)]


class ContextualRetrievalStrategy(BaseRetrievalStrategy):
    """
    上下文相关检索策略：根据用户上下文个性化检索

    上下文相关问题的特点：
    - 需要结合用户的背景、偏好或历史
    - 如"根据我的需求，哪款产品最适合我"
    - 同样的问题，不同用户可能需要不同的答案

    策略步骤：
    1. 结合用户上下文重新表述查询
    2. 使用重新表述后的查询检索更多候选文档
    3. 结合查询和上下文对文档进行相关性评分
    4. 返回评分最高的文档
    """
    def retrieve(self, query, k=4, user_context=None):
        """
        执行上下文相关检索

        参数：
            query: 用户的查询
            k: 返回的文档数量
            user_context: 用户上下文信息（可选），如用户偏好、历史等

        返回：
            考虑上下文后最相关的文档列表
        """
        print("正在检索上下文信息...")

        # 第一步：结合用户上下文重新表述查询
        # 让 AI 根据用户背景将通用查询改写为个性化查询
        context_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="给定用户上下文：{context}\n重新表述查询以最好地满足用户需求：{query}"
        )
        context_chain = context_prompt | self.llm
        # 如果没有提供上下文，使用默认提示
        input_data = {"query": query, "context": user_context or "未提供特定上下文"}
        contextualized_query = context_chain.invoke(input_data).content
        print(f'上下文化查询：{contextualized_query}')

        # 第二步：使用重新表述后的查询检索更多候选文档
        docs = self.db.similarity_search(contextualized_query, k=k * 2)

        # 第三步：结合查询和上下文对每个文档进行相关性评分
        ranking_prompt = PromptTemplate(
            input_variables=["query", "context", "doc"],
            template="给定查询：'{query}' 和用户上下文：'{context}'，在 1-10 的评分范围内评估此文档的相关性：\n文档：{doc}\n相关性得分："
        )
        ranking_chain = ranking_prompt | self.llm.with_structured_output(RelevantScore)

        ranked_docs = []
        for doc in docs:
            # 为每个文档计算相关性分数
            input_data = {"query": contextualized_query, "context": user_context or "未提供特定上下文",
                          "doc": doc.page_content}
            score = float(ranking_chain.invoke(input_data).score)
            ranked_docs.append((doc, score))

        # 第四步：按分数排序并返回前 k 个文档
        ranked_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in ranked_docs[:k]]


# 定义主自适应 RAG 类
class AdaptiveRAG:
    """
    自适应 RAG 系统：根据查询类型自动选择最合适的检索策略

    这是整个系统的核心类，它：
    1. 使用 QueryClassifier 对查询进行分类
    2. 根据分类结果选择对应的检索策略（事实性/分析性/观点性/上下文）
    3. 使用选定的策略检索相关文档
    4. 基于检索到的文档生成答案

    这种设计的优势：
    - 不同类型的问题自动使用最适合的检索方法
    - 比单一检索策略更灵活、更准确
    """
    def __init__(self, texts: List[str]):
        """
        初始化自适应 RAG 系统

        参数：
            texts: 用于检索的文本列表，这些文本会被分割并存储到向量数据库中
        """
        # 初始化查询分类器
        self.classifier = QueryClassifier()

        # 初始化四种检索策略，每种策略针对不同类型的查询
        self.strategies = {
            "Factual": FactualRetrievalStrategy(texts),      # 事实性检索
            "Analytical": AnalyticalRetrievalStrategy(texts),  # 分析性检索
            "Opinion": OpinionRetrievalStrategy(texts),       # 观点性检索
            "Contextual": ContextualRetrievalStrategy(texts)   # 上下文检索
        }

        # 初始化用于生成最终答案的 LLM
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)

        # 定义回答问题的提示模板
        prompt_template = """使用以下背景段落来回答最后的问题。
        如果你不知道答案，就说你不知道，不要试图编造答案。

        {context}

        问题：{question}
        答案："""
        # 将提示模板和 LLM 组合成链
        self.prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        self.llm_chain = self.prompt | self.llm

    def answer(self, query: str) -> str:
        """
        回答用户查询的主方法

        参数：
            query: 用户提出的问题

        返回：
            基于检索到的文档生成的答案
        """
        # 步骤 1：分类查询，确定问题类型
        category = self.classifier.classify(query)

        # 步骤 2：根据分类选择对应的检索策略
        strategy = self.strategies[category]

        # 步骤 3：使用选定的策略检索相关文档
        docs = strategy.retrieve(query)

        # 步骤 4：将检索到的文档作为上下文，让 LLM 生成答案
        input_data = {"context": "\n".join([doc.page_content for doc in docs]), "question": query}
        return self.llm_chain.invoke(input_data).content


# 参数解析函数
def parse_args():
    """
    解析命令行参数

    这个函数允许用户通过命令行传递参数来运行脚本
    例如：python adaptive_retrieval.py --texts "文本 1" "文本 2"

    返回：
        包含解析后参数的对象
    """
    import argparse  # Python 内置的命令行参数解析库
    parser = argparse.ArgumentParser(description="运行自适应 RAG 系统。")
    # 定义 --texts 参数，可以接收多个文本作为输入
    parser.add_argument('--texts', nargs='+', help="用于检索的输入文本")
    return parser.parse_args()


if __name__ == "__main__":
    """
    程序主入口

    当直接运行这个脚本时（而不是作为模块导入），会执行这里的代码
    """
    # 解析命令行参数
    args = parse_args()

    # 如果没有提供文本，使用默认示例文本
    texts = args.texts or [
        "地球是距离太阳第三远的行星，也是唯一已知存在生命的天体。"]

    # 创建自适应 RAG 系统实例
    rag_system = AdaptiveRAG(texts)

    # 定义一系列测试查询，展示不同类型的查询如何处理
    queries = [
        "地球和太阳之间的距离是多少？",           # 事实性问题
        "地球与太阳的距离如何影响其气候？",       # 分析性问题
        "关于地球上生命起源的不同理论有哪些？",   # 观点性问题
        "地球在太阳系中的位置如何影响其宜居性？"   # 分析性问题
    ]

    # 逐个处理每个查询并打印结果
    for query in queries:
        print(f"查询：{query}")
        result = rag_system.answer(query)
        print(f"答案：{result}")
