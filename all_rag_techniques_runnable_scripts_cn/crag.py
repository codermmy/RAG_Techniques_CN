# ==================== 导入必要的库 ====================
import os
import sys
import argparse
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools import DuckDuckGoSearchResults
from helper_functions import encode_pdf
import json

# 将父目录添加到系统路径，因为项目使用笔记本工作方式，需要引用上级目录的模块
sys.path.append(os.path.abspath(
    os.path.join(os.getcwd(), '..')))

# 从 .env 文件加载环境变量（主要加载 OPENAI_API_KEY 等配置）
load_dotenv()
# 设置 OpenAI API 密钥，后续所有调用 OpenAI 的操作都需要这个密钥
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


# ==================== 数据模型定义 ====================
# 下面是三个用于结构化输出的数据模型，它们定义了 LLM 返回结果的格式

class RetrievalEvaluatorInput(BaseModel):
    """
    用于捕获文档与查询相关性评分的模型。
    这个模型让 AI 返回一个 0-1 之间的分数，表示文档和查询有多相关。
    """
    relevance_score: float = Field(..., description="0 到 1 之间的相关性评分，"
                                                    "表示文档与查询的相关性。1 表示完全相关，0 表示完全不相关")


class QueryRewriterInput(BaseModel):
    """
    用于捕获重写的查询以适应网络搜索的模型。
    当文档不够相关时，系统会重写查询以便更好地进行网络搜索。
    """
    query: str = Field(..., description="重写的查询以获得更好的网络搜索结果。")


class KnowledgeRefinementInput(BaseModel):
    """
    用于从文档中提取要点的模型。
    这个模型让 AI 从长文档中提取关键信息，以要点形式返回。
    """
    key_points: str = Field(..., description="以要点形式从文档中提取的关键信息。")


class CRAG:
    """
    一个用于处理 CRAG 过程的类，用于文档检索、评估和知识提炼。

    CRAG (Corrective Retrieval-Augmented Generation) 是一种智能检索技术：
    - 当检索到的文档质量高时，直接使用文档内容
    - 当文档质量低时，转而使用网络搜索
    - 当文档质量一般时，结合文档和网络搜索

    这种技术可以让系统更可靠地回答问题，避免基于不相关文档生成错误答案。
    """

    def __init__(self, path, model="gpt-4o-mini", max_tokens=1000, temperature=0, lower_threshold=0.3,
                 upper_threshold=0.7):
        """
        通过对 PDF 文档进行编码并创建必要的模型和搜索工具来初始化 CRAG 检索器。

        CRAG 系统的核心思想：根据检索文档的质量评分，决定三种不同的处理策略：
        1. 评分 > upper_threshold (0.7): 文档质量好，直接使用
        2. 评分 < lower_threshold (0.3): 文档质量差，进行网络搜索
        3. 评分介于两者之间：结合文档和网络搜索

        参数：
            path (str): 要编码的 PDF 文件路径。
            model (str): 用于 CRAG 过程的语言模型，默认使用 GPT-4o-mini。
            max_tokens (int): LLM 响应中使用的最大令牌数（默认：1000），控制回答长度。
            temperature (float): 用于 LLM 响应的温度（默认：0），0 表示最确定、最一致的回答。
            lower_threshold (float): 文档评估评分的下阈值（默认：0.3），低于此值认为文档不相关。
            upper_threshold (float): 文档评估评分的上阈值（默认：0.7），高于此值认为文档非常相关。
        """
        print("\n--- 初始化 CRAG 过程 ---")
        print("CRAG 系统将根据检索文档的质量评分，智能选择：使用文档、网络搜索或两者结合")

        self.lower_threshold = lower_threshold  # 下阈值：低于此分数则进行网络搜索
        self.upper_threshold = upper_threshold  # 上阈值：高于此分数则直接使用文档

        # 将 PDF 文档编码到向量存储中
        # encode_pdf 函数会读取 PDF，分割成块，并为每块创建向量嵌入
        self.vectorstore = encode_pdf(path)

        # 初始化 OpenAI 语言模型
        # 这个模型将用于：评估文档相关性、提取关键点、重写查询、生成最终回答
        self.llm = ChatOpenAI(model=model, max_tokens=max_tokens, temperature=temperature)

        # 初始化搜索工具（DuckDuckGo 搜索引擎）
        # 当本地文档不够相关时，系统会使用这个工具进行网络搜索
        self.search = DuckDuckGoSearchResults()

    @staticmethod
    def retrieve_documents(query, faiss_index, k=3):
        """
        从 FAISS 向量索引中检索与查询最相关的文档。

        工作原理：
        - FAISS 是 Facebook 开发的高效相似度搜索库
        - 它将查询向量化后，找到与之最相似的 k 个文档块
        - 相似度通常使用余弦相似度或欧氏距离计算

        参数：
            query: 用户的查询字符串
            faiss_index: 已构建好的 FAISS 向量索引
            k: 返回最相关文档的数量，默认 3 个

        返回：
            文档内容列表（仅返回文本内容，不包含元数据）
        """
        # similarity_search 返回按相似度排序的文档对象列表
        docs = faiss_index.similarity_search(query, k=k)
        # 只提取文档的文本内容，忽略其他元数据
        return [doc.page_content for doc in docs]

    def evaluate_documents(self, query, documents):
        """
        评估多个文档与查询的相关性。

        参数：
            query: 用户查询
            documents: 待评估的文档列表

        返回：
            每个文档的相关性评分列表（0-1 之间的浮点数）
        """
        # 对每个文档调用 retrieval_evaluator 进行评分
        return [self.retrieval_evaluator(query, doc) for doc in documents]

    def retrieval_evaluator(self, query, document):
        """
        使用 LLM 评估单个文档与查询的相关性。

        工作原理：
        - 构造一个提示词模板，包含查询和文档内容
        - 让 LLM 根据内容相关性给出 0-1 的评分
        - 使用结构化输出确保返回格式正确

        参数：
            query: 用户查询
            document: 待评估的文档内容

        返回：
            相关性评分（0-1 之间的浮点数）
        """
        # 定义评估提示词模板
        # {query} 和 {document} 是占位符，稍后会被实际值替换
        prompt = PromptTemplate(
            input_variables=["query", "document"],
            template="在 0 到 1 的评分范围内，以下文档与查询的相关性如何？"
                     "查询：{query}\n文档：{document}\n相关性评分："
        )
        # 构建处理链：提示词 -> LLM（使用结构化输出）
        # with_structured_output 确保 LLM 返回符合 RetrievalEvaluatorInput 格式的结果
        chain = prompt | self.llm.with_structured_output(RetrievalEvaluatorInput)
        input_variables = {"query": query, "document": document}
        # 调用 LLM 并获取评分
        result = chain.invoke(input_variables).relevance_score
        return result

    def knowledge_refinement(self, document):
        """
        从文档中提取关键信息要点。

        工作原理：
        - 使用 LLM 阅读文档内容
        - 提取最重要的信息，以要点形式返回
        - 用于压缩文档内容，保留核心信息

        参数：
            document: 需要提取关键点的文档内容

        返回：
            关键点列表，每个关键点是一个字符串
        """
        # 定义关键点提取提示词模板
        prompt = PromptTemplate(
            input_variables=["document"],
            template="以要点形式从以下文档中提取关键信息："
                     "\n{document}\n关键点："
        )
        # 构建处理链，使用结构化输出确保返回格式正确
        chain = prompt | self.llm.with_structured_output(KnowledgeRefinementInput)
        input_variables = {"document": document}
        # 调用 LLM 并获取结果
        result = chain.invoke(input_variables).key_points
        # 将结果按行分割，过滤空行，返回关键点列表
        return [point.strip() for point in result.split('\n') if point.strip()]

    def rewrite_query(self, query):
        """
        重写查询以更适合网络搜索。

        为什么要重写查询？
        - 原始查询可能太口语化或不完整
        - 网络搜索引擎更喜欢简洁、关键词式的查询
        - 重写可以提高网络搜索的准确性

        参数：
            query: 用户的原始查询

        返回：
            重写后的查询字符串
        """
        # 定义查询重写提示词模板
        prompt = PromptTemplate(
            input_variables=["query"],
            template="重写以下查询以使其更适合网络搜索："
                     "\n{query}\n重写后的查询："
        )
        # 构建处理链，使用结构化输出
        chain = prompt | self.llm.with_structured_output(QueryRewriterInput)
        input_variables = {"query": query}
        # 调用 LLM 并返回重写后的查询（去除首尾空格）
        return chain.invoke(input_variables).query.strip()

    @staticmethod
    def parse_search_results(results_string):
        """
        解析网络搜索返回的 JSON 结果。

        参数：
            results_string: 搜索引擎返回的 JSON 格式字符串

        返回：
            包含 (标题，链接) 元组的列表

        异常处理：
            如果 JSON 解析失败，返回空列表并打印错误信息
        """
        try:
            # 将 JSON 字符串解析为 Python 对象（通常是字典列表）
            results = json.loads(results_string)
            # 提取每个结果的标题和链接，如果不存在则使用默认值
            return [(result.get('title', '无标题'), result.get('link', '')) for result in results]
        except json.JSONDecodeError:
            # JSON 解析失败时的错误处理
            print("解析搜索结果时出错。返回空列表。")
            return []

    def perform_web_search(self, query):
        """
        执行网络搜索并提取知识。

        完整流程：
        1. 重写查询以优化搜索结果
        2. 使用 DuckDuckGo 执行搜索
        3. 从搜索结果中提取关键点

        参数：
            query: 用户查询

        返回：
            - web_knowledge: 从搜索结果中提取的关键点列表
            - sources: 来源列表（标题，链接）
        """
        # 第一步：重写查询以获得更好的搜索结果
        rewritten_query = self.rewrite_query(query)
        print(f"重写后的查询：{rewritten_query}")

        # 第二步：执行网络搜索
        web_results = self.search.run(rewritten_query)

        # 第三步：从搜索结果中提取关键点
        web_knowledge = self.knowledge_refinement(web_results)

        # 第四步：解析搜索结果以获取来源信息
        sources = self.parse_search_results(web_results)

        return web_knowledge, sources

    def generate_response(self, query, knowledge, sources):
        """
        基于检索到的知识生成最终回答。

        工作原理：
        - 使用提示词模板组织查询、知识和来源
        - 让 LLM 基于提供的知识生成准确的回答
        - 在回答末尾附上来源信息

        参数：
            query: 用户原始查询
            knowledge: 检索到的知识（可能是文档内容、网络搜索结果或两者结合）
            sources: 来源列表（标题，链接）

        返回：
            LLM 生成的回答内容
        """
        # 定义回答生成提示词模板
        response_prompt = PromptTemplate(
            input_variables=["query", "knowledge", "sources"],
            template="根据以下知识回答查询。"
                     "在答案末尾附上来源及其链接（如果可用）："
                     "\n查询：{query}\n知识：{knowledge}\n来源：{sources}\n答案："
        )
        # 准备输入变量
        input_variables = {
            "query": query,
            "knowledge": knowledge,
            # 将来源格式化为"标题：链接"的形式
            "sources": "\n".join([f"{title}: {link}" if link else title for title, link in sources])
        }
        # 构建处理链（不需要结构化输出，因为直接返回文本）
        response_chain = response_prompt | self.llm
        # 调用 LLM 并返回回答内容
        return response_chain.invoke(input_variables).content

    def run(self, query):
        """
        CRAG 系统的主运行流程，处理用户查询并生成回答。

        完整工作流程：
        1. 从向量存储中检索相关文档
        2. 评估每个文档与查询的相关性评分
        3. 根据最高评分决定处理策略：
           - 评分 > 上阈值：使用检索到的文档（正确）
           - 评分 < 下阈值：执行网络搜索（不正确）
           - 评分介于两者之间：结合文档和网络搜索（模糊）
        4. 基于选定策略获取知识
        5. 生成最终回答

        参数：
            query: 用户查询

        返回：
            生成的回答字符串
        """
        print(f"\n处理查询：{query}")

        # 步骤 1：从 FAISS 向量存储中检索最相关的文档（默认 3 个）
        retrieved_docs = self.retrieve_documents(query, self.vectorstore)

        # 步骤 2：使用 LLM 评估每个文档与查询的相关性
        eval_scores = self.evaluate_documents(query, retrieved_docs)

        # 打印调试信息
        print(f"\n检索到 {len(retrieved_docs)} 个文档")
        print(f"评估评分：{eval_scores}")

        # 步骤 3：根据评估评分确定操作策略
        # 取最高评分作为决策依据
        max_score = max(eval_scores)
        sources = []  # 用于记录知识来源

        # 根据最高评分与阈值的关系，决定三种不同的处理策略
        if max_score > self.upper_threshold:
            # 情况 1：文档质量高，直接使用检索到的文档
            print("\n操作：正确 - 使用检索到的文档")
            # 找到评分最高的文档
            best_doc = retrieved_docs[eval_scores.index(max_score)]
            final_knowledge = best_doc  # 直接使用文档原文
            sources.append(("检索到的文档", ""))

        elif max_score < self.lower_threshold:
            # 情况 2：文档质量差，执行网络搜索
            print("\n操作：不正确 - 执行网络搜索")
            final_knowledge, sources = self.perform_web_search(query)

        else:
            # 情况 3：文档质量一般，结合本地文档和网络搜索
            print("\n操作：模糊 - 结合检索到的文档和网络搜索")
            best_doc = retrieved_docs[eval_scores.index(max_score)]
            # 从检索到的文档中提取关键点
            retrieved_knowledge = self.knowledge_refinement(best_doc)
            # 同时执行网络搜索获取额外信息
            web_knowledge, web_sources = self.perform_web_search(query)
            # 合并两种来源的知识
            final_knowledge = "\n".join(retrieved_knowledge + web_knowledge)
            sources = [("检索到的文档", "")] + web_sources

        # 打印最终知识和来源（用于调试）
        print("\n最终知识：")
        print(final_knowledge)

        print("\n来源：")
        for title, link in sources:
            print(f"{title}: {link}" if link else title)

        # 步骤 4：基于收集的知识生成最终回答
        print("\n生成响应...")
        response = self.generate_response(query, final_knowledge, sources)
        print("\n响应已生成")
        return response


# ==================== 命令行参数处理函数 ====================

def validate_args(args):
    """
    验证命令行参数的有效性。

    验证规则：
    - max_tokens 必须是正整数（令牌数不能为负或零）
    - temperature 必须在 0 到 1 之间（温度参数控制随机性）

    参数：
        args: 命令行参数对象

    返回：
        验证通过后的参数对象

    异常：
        ValueError: 当参数值超出有效范围时抛出
    """
    if args.max_tokens <= 0:
        raise ValueError("max_tokens 必须是正整数。")
    if args.temperature < 0 or args.temperature > 1:
        raise ValueError("temperature 必须在 0 和 1 之间。")
    return args


def parse_args():
    """
    解析命令行参数。

    允许的命令行参数：
    --path: PDF 文件路径
    --model: 使用的语言模型
    --max_tokens: LLM 响应最大令牌数
    --temperature: LLM 响应温度
    --query: 测试查询
    --lower_threshold: 下阈值
    --upper_threshold: 上阈值

    返回：
        解析后的参数对象
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="用于文档检索和查询回答的 CRAG 过程。")

    # 添加各个参数的定义
    parser.add_argument("--path", type=str, default="../data/Understanding_Climate_Change.pdf",
                        help="要编码的 PDF 文件路径。")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="要使用的语言模型（默认：gpt-4o-mini）。")
    parser.add_argument("--max_tokens", type=int, default=1000,
                        help="LLM 响应中使用的最大令牌数（默认：1000）。")
    parser.add_argument("--temperature", type=float, default=0,
                        help="用于 LLM 响应的温度（默认：0）。")
    parser.add_argument("--query", type=str, default="What are the main causes of climate change?",
                        help="用于测试 CRAG 过程的查询。")
    parser.add_argument("--lower_threshold", type=float, default=0.3,
                        help="评分评估的下阈值（默认：0.3）。")
    parser.add_argument("--upper_threshold", type=float, default=0.7,
                        help="评分评估的上阈值（默认：0.7）。")

    # 解析参数并验证
    return validate_args(parser.parse_args())


def main(args):
    """
    主函数：处理参数解析并调用 CRAG 类。

    工作流程：
    1. 使用命令行参数初始化 CRAG 实例
    2. 运行 CRAG 处理查询
    3. 打印查询和答案

    参数：
        args: 解析后的命令行参数
    """
    # 初始化 CRAG 过程
    # 传入所有配置参数，创建 CRAG 实例
    crag = CRAG(
        path=args.path,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        lower_threshold=args.lower_threshold,
        upper_threshold=args.upper_threshold
    )

    # 处理查询并获取回答
    response = crag.run(args.query)

    # 打印结果
    print(f"查询：{args.query}")
    print(f"答案：{response}")


if __name__ == '__main__':
    # 程序入口：解析参数并运行主函数
    main(parse_args())
