# ============================================================================
# 导入必要的库和模块
# ============================================================================
import os
import sys
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field

# 将父目录添加到路径，因为我们使用 notebooks
# 这样可以导入上级目录中的 helper_functions 等模块
sys.path.append(os.path.abspath(
    os.path.join(os.getcwd(), '..')))
from helper_functions import *
from evaluation.evalute_rag import *

# 从 .env 文件加载环境变量
# .env 文件包含敏感配置信息，如 API 密钥
load_dotenv()

# 设置 OpenAI API 密钥环境变量
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


# ============================================================================
# 响应数据模型定义
# ============================================================================
# Self-RAG 使用多个判断步骤，每个步骤都有特定的输出格式
# 这些类定义了每个步骤的合法输出

class RetrievalResponse(BaseModel):
    """
    检索决策响应

    用于判断是否需要检索外部知识
    选项：'Yes'（需要检索）或 'No'（不需要检索）
    """
    response: str = Field(..., title="确定是否需要检索", description="仅输出 'Yes' 或 'No'。")


class RelevanceResponse(BaseModel):
    """
    相关性判断响应

    用于判断检索到的文档是否与查询相关
    选项：'Relevant'（相关）或 'Irrelevant'（不相关）
    """
    response: str = Field(..., title="确定上下文是否相关",
                          description="仅输出 'Relevant' 或 'Irrelevant'。")


class GenerationResponse(BaseModel):
    """
    生成响应

    用于存储最终生成的答案
    """
    response: str = Field(..., title="生成的响应", description="生成的响应。")


class SupportResponse(BaseModel):
    """
    支持度判断响应

    用于判断生成的回答是否有上下文支持
    选项：
        'Fully supported'（完全支持）：回答完全基于上下文
        'Partially supported'（部分支持）：回答部分基于上下文
        'No support'（无支持）：回答没有上下文支持
    """
    response: str = Field(..., title="确定响应是否得到支持",
                          description="输出 'Fully supported'、'Partially supported' 或 'No support'。")


class UtilityResponse(BaseModel):
    """
    效用评分响应

    用于对生成回答的质量进行评分（1-5 分）
    5 分最高，1 分最低
    """
    response: int = Field(..., title="效用评分", description="对响应的效用进行 1 到 5 的评分。")


# ============================================================================
# 提示模板定义
# ============================================================================
# Self-RAG 使用多个提示模板，每个模板对应一个判断步骤

# 检索决策提示：判断是否需要检索
retrieval_prompt = PromptTemplate(
    # 模板需要的变量
    input_variables=["query"],
    # 模板内容
    template="给定查询 '{query}'，确定是否需要检索。仅输出 'Yes' 或 'No'。"
)

# 相关性判断提示：判断文档是否相关
relevance_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="给定查询 '{query}' 和上下文 '{context}'，确定上下文是否相关。仅输出 'Relevant' 或 'Irrelevant'。"
)

# 生成提示：使用上下文生成回答
generation_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="给定查询 '{query}' 和上下文 '{context}'，生成响应。"
)

# 支持度判断提示：判断回答是否有上下文支持
support_prompt = PromptTemplate(
    input_variables=["response", "context"],
    template="给定响应 '{response}' 和上下文 '{context}'，确定响应是否得到上下文支持。输出 'Fully supported'、'Partially supported' 或 'No support'。"
)

# 效用评分提示：对回答质量评分
utility_prompt = PromptTemplate(
    input_variables=["query", "response"],
    template="给定查询 '{query}' 和响应 '{response}'，对响应的效用进行 1 到 5 的评分。"
)


# ============================================================================
# Self-RAG 主类
# ============================================================================
class SelfRAG:
    """
    Self-RAG（自我反思式检索增强生成）系统

    Self-RAG 的核心特点：
    1. 自主判断是否需要检索（不是所有问题都需要检索）
    2. 对检索到的文档进行相关性过滤
    3. 对生成的回答进行自我评估（支持度和效用）
    4. 选择最佳的回答输出

    工作流程（6 个步骤）：
    步骤 1：判断是否需要检索
    步骤 2：如果需要，检索相关文档
    步骤 3：评估每个文档的相关性
    步骤 4：使用相关文档生成候选回答
    步骤 5：评估每个回答的支持度
    步骤 6：评估每个回答的效用
    最终：选择最佳回答
    """

    def __init__(self, path, top_k=3):
        """
        初始化 Self-RAG 系统

        参数：
            path: PDF 文件路径，用于构建向量知识库
            top_k: 检索时返回的文档数量，默认 3 个
        """
        # 加载 PDF 并构建向量存储
        # encode_pdf 函数：读取 PDF -> 分割成块 -> 转换为向量 -> 存储
        self.vectorstore = encode_pdf(path)
        # 保存检索数量
        self.top_k = top_k
        # 初始化 LLM，使用 gpt-4o-mini 模型
        self.llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1000, temperature=0)

        # 为每个步骤创建处理链
        # 每个链 = 提示模板 + LLM + 结构化输出
        self.retrieval_chain = retrieval_prompt | self.llm.with_structured_output(RetrievalResponse)
        self.relevance_chain = relevance_prompt | self.llm.with_structured_output(RelevanceResponse)
        self.generation_chain = generation_prompt | self.llm.with_structured_output(GenerationResponse)
        self.support_chain = support_prompt | self.llm.with_structured_output(SupportResponse)
        self.utility_chain = utility_prompt | self.llm.with_structured_output(UtilityResponse)

    def run(self, query):
        """
        运行 Self-RAG 系统

        参数：
            query: 用户查询

        返回：
            最终生成的回答

        详细流程：
            1. 判断是否需要检索
            2. 如果需要，检索相关文档
            3. 评估文档相关性，过滤不相关文档
            4. 对每个相关文档生成回答
            5. 评估每个回答的支持度
            6. 评估每个回答的效用
            7. 选择最佳回答
        """
        # 打印查询信息
        print(f"\n正在处理查询：{query}")

        # ========== 步骤 1：确定是否需要检索 ==========
        print("步骤 1：确定是否需要检索...")
        input_data = {"query": query}
        # 调用检索决策链，获取判断结果
        # .response 提取响应字段，.strip() 去除空白，.lower() 转小写
        retrieval_decision = self.retrieval_chain.invoke(input_data).response.strip().lower()
        print(f"检索决策：{retrieval_decision}")

        # 判断结果为"需要检索"
        if retrieval_decision == 'yes':
            # ========== 步骤 2：检索相关文档 ==========
            print("步骤 2：检索相关文档...")
            # similarity_search 返回与查询最相似的 k 个文档
            docs = self.vectorstore.similarity_search(query, k=self.top_k)
            # 提取文档内容
            contexts = [doc.page_content for doc in docs]
            print(f"检索到 {len(contexts)} 个文档")

            # ========== 步骤 3：评估检索到的文档的相关性 ==========
            print("步骤 3：评估检索到的文档的相关性...")
            # 存储相关文档的列表
            relevant_contexts = []
            # 遍历每个检索到的文档
            for i, context in enumerate(contexts):
                # 准备输入数据
                input_data = {"query": query, "context": context}
                # 调用相关性判断链
                relevance = self.relevance_chain.invoke(input_data).response.strip().lower()
                print(f"文档 {i + 1} 相关性：{relevance}")
                # 如果相关，添加到列表
                if relevance == 'relevant':
                    relevant_contexts.append(context)

            print(f"相关上下文数量：{len(relevant_contexts)}")

            # 如果没有找到相关上下文，直接生成回答（不进行检索增强）
            if not relevant_contexts:
                print("未找到相关上下文。无需检索生成...")
                input_data = {"query": query, "context": "未找到相关上下文。"}
                return self.generation_chain.invoke(input_data).response

            # ========== 步骤 4：使用相关上下文生成响应 ==========
            print("步骤 4：使用相关上下文生成响应...")
            # 存储所有候选回答及其评估结果
            responses = []
            # 遍历每个相关上下文
            for i, context in enumerate(relevant_contexts):
                print(f"正在为上下文 {i + 1} 生成响应...")
                input_data = {"query": query, "context": context}
                # 生成回答
                response = self.generation_chain.invoke(input_data).response

                # ========== 步骤 5：评估支持度 ==========
                print(f"步骤 5：评估响应 {i + 1} 的支持度...")
                input_data = {"response": response, "context": context}
                # 判断回答是否有上下文支持
                support = self.support_chain.invoke(input_data).response.strip().lower()
                print(f"支持度评估：{support}")

                # ========== 步骤 6：评估效用 ==========
                print(f"步骤 6：评估响应 {i + 1} 的效用...")
                input_data = {"query": query, "response": response}
                # 对回答质量进行评分
                utility = int(self.utility_chain.invoke(input_data).response)
                print(f"效用分数：{utility}")

                # 保存回答及其评估结果
                # (回答，支持度，效用分数)
                responses.append((response, support, utility))

            # ========== 选择最佳响应 ==========
            print("选择最佳响应...")
            # 根据支持度和效用选择最佳回答
            # 排序规则：
            #   1. 优先选择"完全支持"的回答
            #   2. 在支持度相同的情况下，选择效用分数最高的
            best_response = max(responses, key=lambda x: (x[1] == 'fully supported', x[2]))
            print(f"最佳响应支持度：{best_response[1]}, 效用：{best_response[2]}")
            return best_response[0]
        else:
            # ========== 不需要检索，直接生成 ==========
            print("无需检索生成...")
            input_data = {"query": query, "context": "无需检索。"}
            # 直接让 LLM 根据自身知识回答
            return self.generation_chain.invoke(input_data).response


# ============================================================================
# 参数解析函数
# ============================================================================
def parse_args():
    """
    解析命令行参数

    允许用户配置：
    - PDF 路径
    - 测试查询

    返回：
        解析后的参数对象
    """
    import argparse
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Self-RAG 方法")

    # PDF 路径参数
    parser.add_argument('--path', type=str, default='../data/Understanding_Climate_Change.pdf',
                        help='用于向量存储的 PDF 文件路径')

    # 查询参数
    parser.add_argument('--query', type=str, default='What is the impact of climate change on the environment?',
                        help='要处理的查询')
    return parser.parse_args()


# ============================================================================
# 程序主入口
# ============================================================================
if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    # 创建 Self-RAG 实例
    rag = SelfRAG(path=args.path)

    # 运行 Self-RAG 系统
    response = rag.run(args.query)

    # 打印最终回答
    print("\n最终响应:")
    print(response)
