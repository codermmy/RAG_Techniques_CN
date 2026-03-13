# ==================== 导入必要的库 ====================
import os
import sys
from dotenv import load_dotenv
from langchain_core.documents import Document
from helper_functions import *
from evaluation.evalute_rag import *
from typing import List

# 从 .env 文件加载环境变量（主要加载 OPENAI_API_KEY）
load_dotenv()

# 设置 OpenAI API 密钥环境变量，后续所有调用 OpenAI 的操作都需要这个密钥
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


# ==================== 文本分块函数 ====================

def split_text_to_chunks_with_indices(text: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    将文本分割成带有索引元数据的 chunks。

    为什么要保留索引？
    - 索引用于标识每个 chunk 在原文中的位置
    - 后续可以通过索引检索相邻的 chunks
    - 实现"窗口式"上下文扩展

    分块策略：
    - chunk_size: 每个块的大小（字符数）
    - chunk_overlap: 相邻块之间的重叠部分（字符数）
    - 重叠可以保持上下文的连贯性，避免信息被切断

    参数：
        text (str): 要分割的完整文本。
        chunk_size (int): 每个块的大小（字符数）。
        chunk_overlap (int): 相邻块之间的重叠（字符数）。

    返回：
        List[Document]: Document 对象列表，每个包含：
            - page_content: 文本块内容
            - metadata: {"index": 块索引，"text": 完整原文}
    """
    chunks = []
    start = 0
    while start < len(text):
        # 计算当前块的结束位置
        end = start + chunk_size
        # 提取当前块的文本
        chunk = text[start:end]
        # 创建 Document 对象，包含内容和元数据
        # index: 当前块的序号
        # text: 完整原文（用于后续 reconstruct）
        chunks.append(Document(page_content=chunk, metadata={"index": len(chunks), "text": text}))
        # 移动起始位置（减去重叠部分）
        start += chunk_size - chunk_overlap
    return chunks


# ==================== 按索引检索分块函数 ====================

def get_chunk_by_index(vectorstore, target_index: int) -> Document:
    """
    根据元数据中的索引从向量存储中检索分块。

    为什么需要这个函数？
    - 当我们需要获取特定索引位置的分块时
    - 用于检索相邻分块以扩展上下文

    工作原理：
    1. 获取向量存储中的所有文档
    2. 遍历查找匹配目标索引的文档

    参数：
        vectorstore: FAISS 向量存储。
        target_index (int): 目标分块的索引。

    返回：
        Document: 匹配索引的文档，如果未找到则返回 None。
    """
    # 获取向量存储中的所有文档
    # similarity_search("", k=ntotal) 返回所有文档（空字符串查询）
    all_docs = vectorstore.similarity_search("", k=vectorstore.index.ntotal)
    # 遍历查找匹配索引的文档
    for doc in all_docs:
        if doc.metadata.get('index') == target_index:
            return doc
    return None


# ==================== 带上下文窗口的检索函数 ====================

def retrieve_with_context_overlap(vectorstore, retriever, query: str, num_neighbors: int = 1, chunk_size: int = 200,
                                  chunk_overlap: int = 20) -> List[str]:
    """
    基于语义相似度检索，并用相邻分块扩展每个检索到的分块。

    核心思想（窗口式上下文扩展）：
    - 传统 RAG 只返回检索到的分块，可能丢失上下文
    - 此方法同时获取相邻分块，提供更完整的信息
    - 例如：如果检索到第 5 块，同时获取第 4、5、6 块

    工作流程：
    1. 使用语义相似度检索相关分块
    2. 对每个相关分块，获取其相邻分块（前后各 num_neighbors 个）
    3. 按顺序连接这些分块，处理重叠部分
    4. 返回扩展后的文本序列

    参数：
        vectorstore: FAISS 向量存储。
        retriever: 基础检索器。
        query (str): 用户查询。
        num_neighbors (int): 每侧获取的相邻分块数量，默认 1。
        chunk_size (int): 分块大小。
        chunk_overlap (int): 分块间重叠。

    返回：
        List[str]: 扩展后的文本序列列表。
    """
    # 使用基础检索器获取相关分块
    relevant_chunks = retriever.get_relevant_documents(query)
    result_sequences = []

    # 处理每个相关分块
    for chunk in relevant_chunks:
        # 获取当前分块的索引
        current_index = chunk.metadata.get('index')
        if current_index is None:
            continue

        # 确定要检索的分块范围
        # 确保起始索引不小于 0
        start_index = max(0, current_index - num_neighbors)
        # 结束索引（+1 因为 range 是左闭右开）
        end_index = current_index + num_neighbors + 1

        # 检索范围内的所有分块
        neighbor_chunks = []
        for i in range(start_index, end_index):
            neighbor_chunk = get_chunk_by_index(vectorstore, i)
            if neighbor_chunk:
                neighbor_chunks.append(neighbor_chunk)

        # 按索引对分块进行排序以确保正确的顺序
        neighbor_chunks.sort(key=lambda x: x.metadata.get('index', 0))

        # 连接分块，考虑重叠部分
        concatenated_text = neighbor_chunks[0].page_content
        for i in range(1, len(neighbor_chunks)):
            current_chunk = neighbor_chunks[i].page_content
            # 计算重叠开始位置
            overlap_start = max(0, len(concatenated_text) - chunk_overlap)
            # 连接：保留第一部分（去掉重叠）+ 当前分块
            concatenated_text = concatenated_text[:overlap_start] + current_chunk

        result_sequences.append(concatenated_text)

    return result_sequences


# ==================== RAG 方法主类 ====================

class RAGMethod:
    """
    封装上下文窗口增强的 RAG 方法。

    这个方法演示了：
    1. 如何将文本分块并保留索引
    2. 如何构建向量存储
    3. 如何检索并用相邻上下文扩展结果

    与传统 RAG 的区别：
    - 传统 RAG：只返回最相似的分块
    - 本方法：返回分块 + 相邻分块，提供更完整的上下文
    """

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 200):
        """
        初始化 RAG 方法。

        参数：
            chunk_size (int): 分块大小，默认 400 字符。
            chunk_overlap (int): 分块重叠，默认 200 字符。
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 准备文档和检索器
        self.docs = self._prepare_docs()
        self.vectorstore, self.retriever = self._prepare_retriever()

    def _prepare_docs(self) -> List[Document]:
        """
        准备示例文档并分块。

        文档内容：人工智能历史简介

        返回：
            List[Document]: 分块后的文档列表。
        """
        # 示例文本：关于 AI 历史的简介
        content = """
            人工智能（AI）的历史可以追溯到 20 世纪中叶。"人工智能"一词于 1956 年在达特茅斯会议上被提出，标志着该领域的正式诞生。

            在 1950 年代和 1960 年代，AI 研究专注于符号方法和问题解决。1955 年由 Allen Newell 和 Herbert A. Simon 创建的 Logic Theorist 通常被认为是第一个 AI 程序。

            1960 年代见证了专家系统的发展，它使用预定义的规则来解决复杂问题。DENDRAL 创建于 1965 年，是最早的专家系统之一，用于分析化学化合物。

            然而，1970 年代带来了第一个"AI 寒冬"，这是一个资金减少和 AI 研究兴趣降低的时期，主要是因为承诺过多但交付不足。

            1980 年代，随着专家系统在企业中的普及，AI 研究重新焕发活力。日本政府第五代计算机项目也刺激了全球对 AI 研究的投资增加。

            神经网络在 1980 年代和 1990 年代开始兴起。反向传播算法虽然更早被发现，但在此期间被广泛用于训练多层网络。

            1990 年代后期和 2000 年代标志着机器学习方法的兴起。支持向量机（SVM）和随机森林在各种分类和回归任务中变得流行。

            深度学习作为机器学习的一个子集，使用具有多层的神经网络，在 2010 年代初期开始显示出有希望的结果。突破性进展出现在 2012 年，当时深度神经网络在 ImageNet 竞赛中显著优于其他机器学习方法。

            从那时起，深度学习彻底改变了许多 AI 应用，包括图像和语音识别、自然语言处理和游戏。2016 年，谷歌的 AlphaGo 击败了世界冠军围棋选手，这是 AI 领域的里程碑式成就。

            当前的 AI 时代特征是将深度学习与其他 AI 技术相结合，开发更高效和强大的硬件，以及对 AI 部署的伦理考虑。

            Transformers 于 2017 年推出，已成为自然语言处理领域的主导架构，使 GPT（生成式预训练 Transformer）等模型能够生成类似人类的文本。

            随着 AI 的不断发展，新的挑战和机遇也随之出现。可解释 AI、稳健和公平的机器学习以及人工通用智能（AGI）是当前和未来该领域研究的关键方向。
            """
        # 使用带索引的分块函数分割文本
        return split_text_to_chunks_with_indices(content, self.chunk_size, self.chunk_overlap)

    def _prepare_retriever(self):
        """
        准备向量存储和检索器。

        流程：
        1. 创建 OpenAI 嵌入模型
        2. 从文档构建 FAISS 向量存储
        3. 创建检索器（每次返回 1 个最相关分块）

        返回：
            tuple: (vectorstore, retriever)
        """
        # 创建嵌入模型
        embeddings = OpenAIEmbeddings()
        # 从文档构建 FAISS 索引
        vectorstore = FAISS.from_documents(self.docs, embeddings)
        # 创建检索器，search_kwargs={"k": 1} 表示只返回 1 个最相关结果
        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
        return vectorstore, retriever

    def run(self, query: str, num_neighbors: int = 1):
        """
        运行 RAG 检索，比较基线和增强结果。

        参数：
            query (str): 用户查询。
            num_neighbors (int): 每侧相邻分块数量。

        返回：
            tuple: (baseline_chunk, enriched_chunk)
                - baseline_chunk: 仅语义检索的结果
                - enriched_chunk: 带上下文扩展的结果
        """
        # 获取基线结果（仅语义检索）
        baseline_chunk = self.retriever.get_relevant_documents(query)
        # 获取增强结果（带上下文窗口）
        enriched_chunks = retrieve_with_context_overlap(self.vectorstore, self.retriever, query, num_neighbors,
                                                        self.chunk_size, self.chunk_overlap)
        # 返回第一个结果的两种版本
        return baseline_chunk[0].page_content, enriched_chunks[0]


# ==================== 命令行参数解析 ====================

def parse_args():
    """
    解析命令行参数。

    返回：
        解析后的参数对象
    """
    import argparse
    parser = argparse.ArgumentParser(description="在给定 PDF 和查询上运行 RAG 方法。")
    parser.add_argument("--query", type=str, default="深度学习何时在 AI 领域变得突出？",
                        help="用于测试检索器的查询（默认：'文档的主题是什么？'）。")
    parser.add_argument('--chunk_size', type=int, default=400, help="文本分块的大小。")
    parser.add_argument('--chunk_overlap', type=int, default=200, help="分块之间的重叠。")
    parser.add_argument('--num_neighbors', type=int, default=1, help="上下文的相邻分块数量。")
    return parser.parse_args()


# ==================== 主程序入口 ====================

# 主执行
if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    # 初始化并运行 RAG 方法
    rag_method = RAGMethod(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    # 执行检索，获取基线和增强结果
    baseline, enriched = rag_method.run(args.query, num_neighbors=args.num_neighbors)

    # 打印结果对比
    print("基线分块（仅语义检索）：")
    print(baseline)

    print("\n增强分块（带上下文窗口）：")
    print(enriched)
