# ============================================================================
# 导入必要的库和模块
# ============================================================================
import os
import sys
import argparse
import time
import faiss
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.docstore.in_memory import InMemoryDocstore

# 将父目录添加到路径，因为我们使用 notebooks
# 这样可以导入上级目录中的 helper_functions 等模块
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from helper_functions import *
from evaluation.evalute_rag import *

# 从 .env 文件加载环境变量（例如 OpenAI API 密钥）
# .env 文件包含敏感配置信息，不应提交到版本控制
load_dotenv()
# 设置 OpenAI API 密钥环境变量
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


# ============================================================================
# HyPE（假设提示嵌入）RAG 检索器类
# ============================================================================
# HyPE 的核心思想是：
# 1. 对于每个文档块，让 AI 生成多个"可能被问到的问题"
# 2. 将这些问题转换为向量，作为文档块的"代理表示"
# 3. 检索时，用用户的问题去匹配这些"问题向量"
# 这样做的优势：用户问题更容易与"问题形式的向量"匹配，而不是与"陈述形式的文档"匹配
class HyPE:
    """
    HyPE RAG 检索器：通过假设性问题增强文档检索

    与 HyDe 不同，HyPE 是在"索引阶段"做文章：
    - HyDe：检索时生成假设答案
    - HyPE：索引时为每个文档块生成假设性问题

    工作流程：
    1. 读取 PDF 并分割成块
    2. 对每个块，让 AI 生成多个相关问题
    3. 将这些问题的向量表示存储到向量库
    4. 检索时，用用户问题匹配问题向量，找到对应文档
    """

    def __init__(self, path, chunk_size=1000, chunk_overlap=200, n_retrieved=3):
        """
        初始化 HyPE RAG 检索器

        参数：
            path (str): PDF 文件路径
            chunk_size (int): 每个文本块的大小（字符数），默认 1000
            chunk_overlap (int): 块与块之间的重叠，避免信息被切断，默认 200
            n_retrieved (int): 检索时返回的文档块数量，默认 3
        """
        print("\n--- 初始化 HyPE RAG 检索器 ---")

        # 使用假设提示嵌入将 PDF 文档编码到 FAISS 向量存储中
        # 记录开始时间，用于性能分析
        start_time = time.time()
        # 调用 encode_pdf 方法构建向量存储
        # 这个过程包括：加载 PDF -> 分块 -> 生成假设性问题 -> 嵌入向量 -> 存储
        self.vector_store = self.encode_pdf(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # 记录分块阶段耗时
        self.time_records = {'分块': time.time() - start_time}
        print(f"分块时间：{self.time_records['分块']:.2f} 秒")

        # 从向量存储创建检索器
        # as_retriever() 将向量库转换为检索器接口
        # search_kwargs={"k": n_retrieved} 指定检索时返回 k 个结果
        self.chunks_query_retriever = self.vector_store.as_retriever(search_kwargs={"k": n_retrieved})

    def generate_hypothetical_prompt_embeddings(self, chunk_text):
        """
        为单个文档块生成多个假设性问题及其向量表示

        这个方法的核心作用：
        1. 分析文档块内容
        2. 生成多个能概括该块内容的问题
        3. 将这些问题转换为向量（数学表示）

        参数：
            chunk_text (str): 文档块的文本内容

        返回：
            tuple: (原始块文本，从问题生成的嵌入向量列表)

        举例：
            如果块内容是"温室气体导致全球变暖..."
            生成的问题可能是：
            - "什么是温室气体？"
            - "全球变暖的原因是什么？"
            - "温室效应如何影响气候？"
        """
        # 初始化大语言模型，用于生成问题
        # temperature=0 使输出更稳定、可预测
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
        # 初始化嵌入模型，用于将文本转换为向量
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

        # 定义问题生成的提示模板
        # from_template 从模板字符串创建 PromptTemplate
        question_gen_prompt = PromptTemplate.from_template(
            "分析输入文本并生成基本问题，当回答这些问题时，\
            捕捉文本的主要观点。每个问题应该是一行，\
            没有编号或前缀。\n\n \
            文本:\n{chunk_text}\n\n问题:\n"
        )
        # 构建处理链：输入文本 -> 提示模板 -> LLM 生成 -> 字符串输出
        # | 是链式操作符，将多个组件连接成一个处理流水线
        question_chain = question_gen_prompt | llm | StrOutputParser()

        # 调用处理链生成问题
        # .invoke() 执行处理链，传入文档块作为输入
        # .replace("\n\n", "\n") 规范化换行符
        # .split("\n") 将多行输出分割成问题列表
        questions = question_chain.invoke({"chunk_text": chunk_text}).replace("\n\n", "\n").split("\n")

        # 将问题列表转换为向量
        # embed_documents 批量将多个文本转换为向量
        # 返回：原始块文本和对应的向量列表
        return chunk_text, embedding_model.embed_documents(questions)

    def prepare_vector_store(self, chunks):
        """
        创建和填充 FAISS 向量存储

        FAISS 是 Facebook AI 开发的高效相似度搜索库
        它使用向量距离（如欧几里得距离）来衡量文本相似度

        参数：
            chunks (List[str]): 要处理的文档块列表

        返回：
            FAISS: 包含嵌入向量的 FAISS 向量存储
        """
        vector_store = None  # 等待初始化以确定向量大小

        # 使用线程池并行处理，大幅加速嵌入生成
        # ThreadPoolExecutor 自动管理多个工作线程
        with ThreadPoolExecutor() as pool:
            # 为每个块提交嵌入生成任务
            # pool.submit() 提交任务到线程池
            # as_completed() 按完成顺序返回 futures
            futures = [pool.submit(self.generate_hypothetical_prompt_embeddings, c) for c in chunks]

            # tqdm 显示进度条，方便查看处理进度
            for f in tqdm(as_completed(futures), total=len(chunks)):
                # 获取任务结果：处理后的块和嵌入向量列表
                chunk, vectors = f.result()

                # 一旦知道向量大小就初始化 FAISS 存储
                # 需要知道向量维度才能创建正确的索引
                if vector_store is None:
                    # 创建 FAISS 向量存储
                    vector_store = FAISS(
                        # 嵌入函数：用于将文本转换为向量
                        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
                        # FAISS 索引类型：L2 表示欧几里得距离（常用）
                        # len(vectors[0]) 是向量的维度
                        index=faiss.IndexFlatL2(len(vectors[0])),
                        # 内存文档存储：存储原始文本
                        docstore=InMemoryDocstore(),
                        # 索引到文档 ID 的映射
                        index_to_docstore_id={}
                    )

                # 为每个块存储多个向量表示（一个问题一个向量）
                # 这样检索时，任何一个问题匹配都能找到这个块
                chunks_with_embedding_vectors = [(chunk.page_content, vec) for vec in vectors]
                # 将向量和对应的原始文本添加到向量库
                vector_store.add_embeddings(chunks_with_embedding_vectors)

        return vector_store

    def encode_pdf(self, path, chunk_size=1000, chunk_overlap=200):
        """
        将 PDF 文档编码到向量存储中

        这是 HyPE 的核心流程：
        1. 加载 PDF
        2. 分割成块
        3. 为每个块生成假设性问题向量
        4. 存储到 FAISS 向量库

        参数：
            path: PDF 文件路径
            chunk_size: 每个文本块的大小
            chunk_overlap: 连续块之间的重叠

        返回：
            包含编码后书籍内容的 FAISS 向量存储
        """
        # 加载 PDF 文档
        # PyPDFLoader 是 LangChain 的 PDF 加载器
        loader = PyPDFLoader(path)
        # load() 读取 PDF 并返回文档列表
        documents = loader.load()

        # 将文档分割成块
        # RecursiveCharacterTextSplitter 按字符递归分割文本
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,  # 目标块大小
            chunk_overlap=chunk_overlap,  # 块重叠
            length_function=len  # 计算长度的函数
        )
        # split_documents 分割文档
        texts = text_splitter.split_documents(documents)
        # replace_t_with_space 清理特殊字符（可能是 PDF 转换产生的乱码）
        cleaned_texts = replace_t_with_space(texts)

        # 调用 prepare_vector_store 创建包含假设问题向量的向量库
        return self.prepare_vector_store(cleaned_texts)

    def run(self, query):
        """
        执行检索并显示结果

        参数：
            query (str): 用户查询

        返回：
            None（直接打印结果）
        """
        # 测量检索时间
        start_time = time.time()
        # retrieve_context_per_question 执行实际检索
        # 返回与查询最匹配的文档块内容列表
        context = retrieve_context_per_question(query, self.chunks_query_retriever)
        # 记录检索耗时
        self.time_records['检索'] = time.time() - start_time
        print(f"检索时间：{self.time_records['检索']:.2f} 秒")

        # 去重上下文（去除重复的文档块）
        # set() 去重，list() 转回列表
        context = list(set(context))
        # 展示检索结果
        show_context(context)


# ============================================================================
# 参数验证函数
# ============================================================================
def validate_args(args):
    """
    验证命令行参数的有效性

    参数：
        args: 解析后的参数对象

    返回：
        验证通过的参数对象

    异常：
        ValueError: 参数无效时抛出异常
    """
    # chunk_size 必须是正数（至少 1 个字符）
    if args.chunk_size <= 0:
        raise ValueError("chunk_size 必须是正整数。")
    # chunk_overlap 不能是负数（重叠不能为负）
    if args.chunk_overlap < 0:
        raise ValueError("chunk_overlap 必须是非负整数。")
    # n_retrieved 必须是正数（至少要检索 1 个结果）
    if args.n_retrieved <= 0:
        raise ValueError("n_retrieved 必须是正整数。")
    return args


# ============================================================================
# 参数解析函数
# ============================================================================
def parse_args():
    """
    解析命令行参数

    允许用户配置：
    - PDF 路径
    - 分块大小和重叠
    - 检索结果数量
    - 测试查询
    - 是否评估性能

    返回：
        解析后的参数对象
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="编码 PDF 文档并测试基于 HyPE 的 RAG 系统。")

    # PDF 路径参数
    parser.add_argument("--path", type=str, default="../data/Understanding_Climate_Change.pdf",
                        help="要编码的 PDF 文件路径。")

    # 分块大小参数
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="每个文本块的大小（默认：1000）。")

    # 分块重叠参数
    parser.add_argument("--chunk_overlap", type=int, default=200,
                        help="连续块之间的重叠（默认：200）。")

    # 检索数量参数
    parser.add_argument("--n_retrieved", type=int, default=3,
                        help="每个查询要检索的块数（默认：3）。")

    # 测试查询参数
    parser.add_argument("--query", type=str, default="气候变化的主要原因是什么？",
                        help="测试检索器的查询（默认：'气候变化的主要原因是什么？'）。")

    # 评估标志
    parser.add_argument("--evaluate", action="store_true",
                        help="是否评估检索器的性能（默认：False）。")

    # 验证并返回参数
    return validate_args(parser.parse_args())


# ============================================================================
# 主函数
# ============================================================================
def main(args):
    """
    主程序入口函数

    参数：
        args: 解析后的命令行参数

    流程：
        1. 初始化 HyPE 检索器
        2. 执行查询检索
        3. 可选：评估性能
    """
    # 初始化基于 HyPE 的 RAG 检索器
    # 这个过程会加载 PDF、分块、生成假设问题、构建向量库
    hyperag = HyPE(
        path=args.path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        n_retrieved=args.n_retrieved
    )

    # 基于查询检索上下文
    # 这个过程使用 HyPE 向量库找到最相关的文档块
    hyperag.run(args.query)

    # 评估检索器的性能（如果请求）
    # evaluate_rag 函数会计算准确率、召回率等指标
    if args.evaluate:
        evaluate_rag(hyperag.chunks_query_retriever)


# ============================================================================
# 程序主入口
# ============================================================================
if __name__ == '__main__':
    # 使用解析后的参数调用主函数
    # parse_args() 解析命令行参数
    # main() 执行主要逻辑
    main(parse_args())
