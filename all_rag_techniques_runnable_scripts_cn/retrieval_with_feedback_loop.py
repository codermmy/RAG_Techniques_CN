# ============================================================================
# 导入必要的库和模块
# ============================================================================
import os
import sys
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# 将父目录添加到路径，因为我们使用 notebooks
# 这样可以导入上级目录中的 helper_functions 等模块
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from helper_functions import *
from evaluation.evalute_rag import *

# 从 .env 文件加载环境变量
# .env 文件包含敏感配置信息，如 API 密钥
load_dotenv()

# 设置 OpenAI API 密钥环境变量
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
# 解决某些平台上的库冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ============================================================================
# 数据模型定义
# ============================================================================
# 定义响应类，用于规范 LLM 的输出格式
class Response(BaseModel):
    """
    LLM 响应数据模型

    使用 Pydantic 的 BaseModel 可以：
    1. 定义字段类型和验证规则
    2. 强制 LLM 输出结构化数据
    3. 自动解析 LLM 响应

    字段：
        answer: 是/否类型的回答
    """
    # Field 定义字段的元数据
    # ... 表示这是必填字段
    # title 是字段的描述
    answer: str = Field(..., title="问题的答案。选项只能是 'Yes' 或 'No'")


# ============================================================================
# 工具函数
# ============================================================================
def get_user_feedback(query, response, relevance, quality, comments=""):
    """
    构造用户反馈数据结构

    参数：
        query: 用户的原始查询
        response: RAG 系统生成的回答
        relevance: 相关性评分（1-5 分）
        quality: 质量评分（1-5 分）
        comments: 可选的额外评论

    返回：
        包含反馈信息的字典

    用途：
        这个函数将用户的反馈整理成统一的格式，方便存储和后续分析
    """
    return {
        "query": query,              # 用户问了什么
        "response": response,         # 系统回答了什麼
        "relevance": int(relevance),  # 相关性得分（整数）
        "quality": int(quality),      # 质量得分（整数）
        "comments": comments          # 额外评论
    }


def store_feedback(feedback):
    """
    将反馈数据存储到 JSON 文件

    参数：
        feedback: 反馈数据字典

    存储格式：
        每行一个 JSON 对象（JSON Lines 格式）
        这种格式便于追加写入，也方便逐行读取

    文件位置：
        ../data/feedback_data.json
    """
    # 以追加模式打开文件（"a" 表示 append）
    with open("../data/feedback_data.json", "a") as f:
        # 将反馈数据转换为 JSON 字符串
        json.dump(feedback, f)
        # 写入换行符，每条反馈占一行
        f.write("\n")


def load_feedback_data():
    """
    从文件加载历史反馈数据

    返回：
        反馈数据列表，每个元素是一个字典

    异常处理：
        如果文件不存在，返回空列表并打印提示信息

    用途：
        在调整检索结果或微调模型时，需要使用历史反馈数据
    """
    # 初始化空列表
    feedback_data = []
    try:
        # 打开反馈文件
        with open("../data/feedback_data.json", "r") as f:
            # 逐行读取（JSON Lines 格式）
            for line in f:
                # 解析每行的 JSON 数据
                feedback_data.append(json.loads(line.strip()))
    except FileNotFoundError:
        # 文件不存在时的处理
        print("未找到反馈数据文件。从空反馈开始。")
    return feedback_data


def adjust_relevance_scores(query: str, docs: List[Any], feedback_data: List[Dict[str, Any]]) -> List[Any]:
    """
    根据历史反馈调整文档的相关性分数

    核心思想：
        如果某个历史反馈与当前查询相关，且该反馈的评分很高，
        那么包含该反馈的文档应该获得更高的相关性分数

    参数：
        query: 当前查询
        docs: 检索到的文档列表
        feedback_data: 历史反馈数据

    返回：
        按调整后的相关性分数排序的文档列表
    """
    # 定义相关性判断的提示模板
    relevance_prompt = PromptTemplate(
        # 模板需要的变量
        input_variables=["query", "feedback_query", "doc_content", "feedback_response"],
        # 模板内容
        template="""
        确定以下反馈响应是否与当前查询和文档内容相关。
        还提供用于生成反馈响应的反馈原始查询。
        当前查询：{query}
        反馈查询：{feedback_query}
        文档内容：{doc_content}
        反馈响应：{feedback_response}

        此反馈是否相关？仅回答 'Yes' 或 'No'。
        """
    )
    # 初始化 LLM，使用 gpt-4o 模型
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
    # 将提示模板和 LLM 连接成处理链
    # with_structured_output(Response) 强制输出为 Response 类型
    relevance_chain = relevance_prompt | llm.with_structured_output(Response)

    # 遍历每个文档，调整其相关性分数
    for doc in docs:
        # 存储与当前文档相关的反馈
        relevant_feedback = []

        # 检查每条历史反馈
        for feedback in feedback_data:
            # 准备输入数据
            input_data = {
                "query": query,                    # 当前查询
                "feedback_query": feedback['query'],  # 历史反馈的原始查询
                "doc_content": doc.page_content[:1000],  # 文档内容（截取前 1000 字符）
                "feedback_response": feedback['response']  # 历史反馈的回答
            }
            # 调用 LLM 判断反馈是否相关
            result = relevance_chain.invoke(input_data).answer

            # 如果 LLM 认为相关，添加到列表
            if result == 'yes':
                relevant_feedback.append(feedback)

        # 如果有相关反馈，调整文档的相关性分数
        if relevant_feedback:
            # 计算相关反馈的平均相关性分数
            avg_relevance = sum(f['relevance'] for f in relevant_feedback) / len(relevant_feedback)
            # 调整原始相关性分数
            # 如果平均分数是 3（中等），分数不变
            # 如果平均分数 > 3，分数提高；如果 < 3，分数降低
            doc.metadata['relevance_score'] *= (avg_relevance / 3)

    # 按调整后的相关性分数降序排序
    # lambda x: x.metadata['relevance_score'] 是排序键
    return sorted(docs, key=lambda x: x.metadata['relevance_score'], reverse=True)


def fine_tune_index(feedback_data: List[Dict[str, Any]], texts: List[str]) -> Any:
    """
    基于优质反馈微调向量索引

    核心思想：
        将高质量的问答对添加到训练数据中，重新构建向量库
        这样可以让向量库包含更多"问题 - 答案"模式，提升检索效果

    参数：
        feedback_data: 历史反馈数据列表
        texts: 原始文本列表

    返回：
        新的向量存储

    流程：
        1. 筛选出高质量反馈（相关性和质量都 >= 4 分）
        2. 将这些反馈的"问题 + 回答"添加到训练数据
        3. 重新编码构建向量库
    """
    # 筛选出高质量反馈
    # 相关性 >= 4 且质量 >= 4 的反馈被认为是优质数据
    good_responses = [f for f in feedback_data if f['relevance'] >= 4 and f['quality'] >= 4]

    # 构造额外的训练文本
    # 将每个优质反馈的"问题 + 回答"拼接成文本
    additional_texts = " ".join([f['query'] + " " + f['response'] for f in good_responses])

    # 合并原始文本和额外文本
    all_texts = texts + additional_texts

    # 使用合并后的数据重新编码向量库
    # encode_from_string 将文本转换为向量存储
    new_vectorstore = encode_from_string(all_texts)
    return new_vectorstore


# ============================================================================
# 主 RAG 类
# ============================================================================
class RetrievalAugmentedGeneration:
    """
    带反馈回路的 RAG（检索增强生成）系统

    这个类的特点：
    1. 收集用户反馈（相关性和质量评分）
    2. 使用反馈调整检索结果的相关性分数
    3. 定期使用反馈数据微调向量索引

    RAG 工作流程：
        用户查询 -> 检索相关文档 -> 生成回答 -> 收集反馈 -> 优化检索
    """

    def __init__(self, path: str):
        """
        初始化 RAG 系统

        参数：
            path: PDF 文档路径
        """
        # 保存文档路径
        self.path = path
        # 读取 PDF 内容
        self.content = read_pdf_to_string(self.path)
        # 构建向量存储
        self.vectorstore = encode_from_string(self.content)
        # 创建检索器
        self.retriever = self.vectorstore.as_retriever()
        # 初始化 LLM
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
        # 创建问答链
        # RetrievalQA.from_chain_type 将 LLM 和检索器组合成问答系统
        self.qa_chain = RetrievalQA.from_chain_type(self.llm, retriever=self.retriever)

    def run(self, query: str, relevance: int, quality: int):
        """
        运行 RAG 系统并收集反馈

        参数：
            query: 用户查询
            relevance: 相关性评分（1-5）
            quality: 质量评分（1-5）

        返回：
            系统生成的回答

        流程：
            1. 使用 RAG 生成回答
            2. 存储用户反馈
            3. 加载历史反馈
            4. 调整检索结果的相关性分数
            5. 更新检索器配置
        """
        # 使用 RAG 系统生成回答
        # ["result"] 提取回答文本
        response = self.qa_chain(query)["result"]

        # 构造反馈数据
        feedback = get_user_feedback(query, response, relevance, quality)
        # 存储反馈到文件
        store_feedback(feedback)

        # 检索相关文档
        docs = self.retriever.get_relevant_documents(query)
        # 加载历史反馈数据
        # 使用反馈调整文档的相关性分数
        adjusted_docs = adjust_relevance_scores(query, docs, load_feedback_data())

        # 更新检索器的搜索参数
        # k: 返回的文档数量
        self.retriever.search_kwargs['k'] = len(adjusted_docs)
        # docs: 按调整后分数排序的文档
        self.retriever.search_kwargs['docs'] = adjusted_docs

        return response


# ============================================================================
# 参数解析函数
# ============================================================================
def parse_args():
    """
    解析命令行参数

    允许用户配置：
    - PDF 文档路径
    - 测试查询
    - 反馈的相关性和质量评分

    返回：
        解析后的参数对象
    """
    import argparse
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="运行带有反馈集成的 RAG 系统。")

    # 文档路径参数
    parser.add_argument('--path', type=str, default="../data/Understanding_Climate_Change.pdf",
                        help="文档路径。")

    # 查询参数
    parser.add_argument('--query', type=str, default='What is the greenhouse effect?',
                        help="向 RAG 系统提出的问题。")

    # 相关性评分参数
    parser.add_argument('--relevance', type=int, default=5, help="反馈的相关性分数。")

    # 质量评分参数
    parser.add_argument('--quality', type=int, default=5, help="反馈的质量分数。")
    return parser.parse_args()


# ============================================================================
# 程序主入口
# ============================================================================
if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    # 创建 RAG 系统实例
    rag = RetrievalAugmentedGeneration(args.path)

    # 运行 RAG 系统，使用指定的相关性/质量评分
    result = rag.run(args.query, args.relevance, args.quality)
    # 打印生成的回答
    print(f"响应：{result}")

    # 定期微调向量存储
    # 使用历史反馈数据优化向量索引
    new_vectorstore = fine_tune_index(load_feedback_data(), rag.content)
    # 更新检索器使用新的向量库
    rag.retriever = new_vectorstore.as_retriever()
