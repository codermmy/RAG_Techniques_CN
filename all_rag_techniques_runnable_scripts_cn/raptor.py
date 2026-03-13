# ==================== 导入必要的库 ====================
# 数值计算库
import numpy as np
# 数据处理库
import pandas as pd
# 类型提示
from typing import List, Dict, Any
# LangChain 组件
from langchain.chains import LLMChain
from sklearn.mixture import GaussianMixture
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.messages import AIMessage
from langchain_core.documents import Document
# 可视化库
import matplotlib.pyplot as plt
# 日志库
import logging
# 系统操作库
import os
import sys
# 环境变量加载
from dotenv import load_dotenv

# 将父目录添加到系统路径，因为项目使用笔记本工作方式，需要引用上级目录的模块
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from helper_functions import *
from evaluation.evalute_rag import *

# 从 .env 文件加载环境变量（主要加载 OPENAI_API_KEY）
load_dotenv()

# 设置 OpenAI API 密钥环境变量，后续所有调用 OpenAI 的操作都需要这个密钥
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


# ==================== 辅助函数 ====================
# 这些函数为 RAPTOR 算法提供基础支持

def extract_text(item):
    """
    从字符串或 AIMessage 对象中提取文本内容。

    为什么需要这个函数？
    - LangChain 的某些组件可能返回 AIMessage 对象而不是纯字符串
    - 这个函数确保无论输入是什么类型，都能提取出文本

    参数：
        item: 可能是字符串或 AIMessage 对象

    返回：
        文本内容字符串
    """
    if isinstance(item, AIMessage):
        return item.content
    return item


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    使用 OpenAIEmbeddings 对文本进行嵌入。

    嵌入（Embedding）是什么？
    - 将文本转换为固定长度的向量（通常是几百到几千维）
    - 语义相似的文本在向量空间中距离更近
    - 这是后续聚类和相似度搜索的基础

    参数：
        texts (List[str]): 要嵌入的文本列表。

    返回：
        List[List[float]]: 嵌入向量列表，每个向量是一个浮点数列表。
    """
    embeddings = OpenAIEmbeddings()
    logging.info(f"正在嵌入 {len(texts)} 个文本")
    # 先提取文本内容（处理可能的 AIMessage 对象），然后调用嵌入模型
    return embeddings.embed_documents([extract_text(text) for text in texts])


def perform_clustering(embeddings: np.ndarray, n_clusters: int = 10) -> np.ndarray:
    """
    使用高斯混合模型（GMM）对嵌入进行聚类。

    什么是高斯混合模型？
    - 一种概率聚类算法，假设数据来自多个高斯分布的混合
    - 相比 K-means，GMM 可以考虑簇的协方差结构
    - 每个数据点属于每个簇都有一定的概率

    参数：
        embeddings (np.ndarray): 嵌入数组，形状为 (n_samples, n_features)。
        n_clusters (int): 聚类数量，默认 10 个。

    返回：
        np.ndarray: 每个样本的聚类标签数组。
    """
    logging.info(f"正在执行聚类，聚类数为 {n_clusters}")
    # 创建 GMM 模型，random_state=42 确保结果可重复
    gm = GaussianMixture(n_components=n_clusters, random_state=42)
    # 拟合并预测每个样本属于哪个簇
    return gm.fit_predict(embeddings)


def summarize_texts(texts: List[str], llm: ChatOpenAI) -> str:
    """
    使用 OpenAI 总结文本列表。

    为什么要总结？
    - RAPTOR 算法的核心思想是构建层次化的摘要树
    - 底层的原始文本块被聚类，每类生成一个摘要
    - 这些摘要又成为下一层的输入，递归构建

    参数：
        texts (List[str]): 要总结的文本列表。
        llm (ChatOpenAI): 用于生成总结的语言模型。

    返回：
        str: 生成的总结文本。
    """
    logging.info(f"正在总结 {len(texts)} 个文本")
    # 创建提示词模板
    prompt = ChatPromptTemplate.from_template(
        "简洁地总结以下文本：\n\n{text}"
    )
    # 构建处理链
    chain = prompt | llm
    input_data = {"text": texts}
    return chain.invoke(input_data)


def visualize_clusters(embeddings: np.ndarray, labels: np.ndarray, level: int):
    """
    使用 PCA 可视化聚类结果。

    为什么要可视化？
    - 帮助理解聚类效果
    - 观察不同簇在向量空间中的分布
    - 调试和优化聚类参数

    工作原理：
    1. 使用 PCA 将高维嵌入降维到 2D
    2. 用散点图展示，不同簇用不同颜色

    参数：
        embeddings (np.ndarray): 嵌入数组。
        labels (np.ndarray): 聚类标签数组。
        level (int): 当前层级（用于标题显示）。

    返回：
        None（直接显示 matplotlib 图形）
    """
    from sklearn.decomposition import PCA
    # 创建 PCA 模型，降维到 2 个成分
    pca = PCA(n_components=2)
    # 拟合并转换嵌入数据
    reduced_embeddings = pca.fit_transform(embeddings)

    # 创建图形
    plt.figure(figsize=(10, 8))
    # 绘制散点图，颜色表示簇标签
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis')
    # 添加颜色条
    plt.colorbar(scatter)
    plt.title(f'聚类可视化 - 层级 {level}')
    plt.xlabel('第一主成分')
    plt.ylabel('第二主成分')
    plt.show()


def build_vectorstore(tree_results: Dict[int, pd.DataFrame], embeddings) -> FAISS:
    """
    从 RAPTOR 树中的所有文本构建 FAISS 向量存储。

    RAPTOR 的存储策略：
    - 不仅存储原始文本块，还存储所有层级的摘要
    - 这样检索时可以同时利用细节信息和高层概括
    - 提高检索的全面性和准确性

    参数：
        tree_results (Dict[int, pd.DataFrame]): RAPTOR 树的结果字典，
            键是层级，值是包含 text、embedding、metadata 的 DataFrame。
        embeddings: 嵌入模型实例。

    返回：
        FAISS: 构建好的 FAISS 向量存储。
    """
    all_texts = []
    all_embeddings = []
    all_metadatas = []

    # 遍历所有层级的数据
    for level, df in tree_results.items():
        # 收集所有文本
        all_texts.extend([str(text) for text in df['text'].tolist()])
        # 收集所有嵌入（处理 numpy 数组和列表两种格式）
        all_embeddings.extend([embedding.tolist() if isinstance(embedding, np.ndarray) else embedding for embedding in
                               df['embedding'].tolist()])
        # 收集所有元数据
        all_metadatas.extend(df['metadata'].tolist())

    logging.info(f"正在构建包含 {len(all_texts)} 个文本的向量存储")
    # 创建 Document 对象列表
    documents = [Document(page_content=str(text), metadata=metadata)
                 for text, metadata in zip(all_texts, all_metadatas)]
    # 从文档和嵌入构建 FAISS 索引
    return FAISS.from_documents(documents, embeddings)


def create_retriever(vectorstore: FAISS, llm: ChatOpenAI) -> ContextualCompressionRetriever:
    """
    创建带有上下文压缩的检索器。

    什么是上下文压缩检索器？
    - 先检索出相关文档
    - 然后使用 LLM 从这些文档中提取与查询最相关的部分
    - 去除噪声，保留精华，提高回答质量

    参数：
        vectorstore (FAISS): 向量存储。
        llm (ChatOpenAI): 用于提取的语言模型。

    返回：
        ContextualCompressionRetriever: 配置好的上下文压缩检索器。
    """
    logging.info("正在创建上下文压缩检索器")
    # 创建基础检索器
    base_retriever = vectorstore.as_retriever()

    # 创建提取提示词模板
    prompt = ChatPromptTemplate.from_template(
        "给定以下上下文和问题，仅提取与回答问题相关的信息：\n\n"
        "上下文：{context}\n"
        "问题：{question}\n\n"
        "相关信息："
    )

    # 从 LLM 创建提取器
    extractor = LLMChainExtractor.from_llm(llm, prompt=prompt)
    # 组合压缩器和基础检索器
    return ContextualCompressionRetriever(
        base_compressor=extractor,
        base_retriever=base_retriever
    )


# ==================== RAPTOR 主类 ====================
# RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval
# 一种通过构建层次化摘要树来增强检索效果的方法

# 主类 RAPTORMethod
class RAPTORMethod:
    def __init__(self, texts: List[str], max_levels: int = 3):
        """
        初始化 RAPTOR 方法。

        RAPTOR 的核心思想：
        1. 将文档分成多个文本块
        2. 对文本块进行聚类，相似内容的块归为一类
        3. 为每类生成摘要
        4. 对摘要重复上述过程，形成树状结构
        5. 检索时同时利用原始块和各层级摘要

        这种层次化结构的优势：
        - 既能检索到细节信息，也能获取高层概括
        - 避免遗漏重要信息
        - 提高回答的全面性和准确性

        参数：
            texts (List[str]): 要处理的文本列表。
            max_levels (int): RAPTOR 树的最大层级数，默认 3 层。
                层级 0：原始文本块
                层级 1：第一层摘要
                层级 2：第二层摘要（摘要的摘要）
                ...

        属性：
            - texts: 原始文本列表
            - max_levels: 最大层级数
            - embeddings: 嵌入模型
            - llm: 语言模型
            - tree_results: 构建好的 RAPTOR 树结果
        """
        self.texts = texts
        self.max_levels = max_levels
        # OpenAI 嵌入模型，用于将文本转换为向量
        self.embeddings = OpenAIEmbeddings()
        # GPT-4o-mini 模型，用于生成摘要
        self.llm = ChatOpenAI(model_name="gpt-4o-mini")
        # 构建 RAPTOR 树
        self.tree_results = self.build_raptor_tree()

    def build_raptor_tree(self) -> Dict[int, pd.DataFrame]:
        """
        构建 RAPTOR 树结构，包含层级元数据和父子关系。

        详细构建流程：

        层级 0（底层）：
        - 输入：原始文本块
        - 处理：嵌入 + 聚类
        - 输出：带聚类标签的文本块

        层级 1：
        - 输入：层级 0 的聚类结果
        - 处理：为每个簇生成摘要
        - 输出：摘要及其元数据

        层级 2+：
        - 重复层级 1 的过程，直到达到 max_levels 或只剩一个摘要

        返回：
            Dict[int, pd.DataFrame]: 层级到 DataFrame 的映射字典。
                每个 DataFrame 包含：
                - text: 文本内容（原始块或摘要）
                - embedding: 向量嵌入
                - cluster: 所属簇 ID
                - metadata: 元数据（层级、来源、父子关系等）
        """
        results = {}  # 存储每个层级的结果
        # 初始化：使用原始文本
        current_texts = [extract_text(text) for text in self.texts]
        # 原始文本的元数据（层级 0，无父节点）
        current_metadata = [{"level": 0, "origin": "original", "parent_id": None} for _ in self.texts]

        # 迭代构建每一层
        for level in range(1, self.max_levels + 1):
            logging.info(f"正在处理层级 {level}")

            # 步骤 1：为当前层级的文本创建嵌入
            embeddings = embed_texts(current_texts)

            # 步骤 2：确定聚类数量
            # 聚类数 = min(10, 文本数/2)，确保每个簇至少有 2 个文本
            n_clusters = min(10, len(current_texts) // 2)

            # 步骤 3：执行聚类
            cluster_labels = perform_clustering(np.array(embeddings), n_clusters)

            # 步骤 4：创建 DataFrame 存储当前层级数据
            df = pd.DataFrame({
                'text': current_texts,
                'embedding': embeddings,
                'cluster': cluster_labels,
                'metadata': current_metadata
            })

            # 存储上一层级的结果
            results[level - 1] = df

            # 步骤 5：为每个簇生成摘要
            summaries = []
            new_metadata = []
            for cluster in df['cluster'].unique():
                # 获取当前簇的所有文档
                cluster_docs = df[df['cluster'] == cluster]
                cluster_texts = cluster_docs['text'].tolist()
                cluster_metadata = cluster_docs['metadata'].tolist()

                # 使用 LLM 总结该簇的所有文本
                summary = summarize_texts(cluster_texts, self.llm)
                summaries.append(summary)

                # 创建新摘要的元数据
                new_metadata.append({
                    "level": level,
                    "origin": f"summary_of_cluster_{cluster}_level_{level - 1}",
                    "child_ids": [meta.get('id') for meta in cluster_metadata],
                    "id": f"summary_{level}_{cluster}"
                })

            # 更新当前文本和元数据为摘要，用于下一轮迭代
            current_texts = summaries
            current_metadata = new_metadata

            # 终止条件：如果只剩一个摘要，无需继续
            if len(current_texts) <= 1:
                results[level] = pd.DataFrame({
                    'text': current_texts,
                    'embedding': embed_texts(current_texts),
                    'cluster': [0],
                    'metadata': current_metadata
                })
                logging.info(f"在层级 {level} 停止，因为只有一个摘要")
                break

        return results

    def run(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        运行 RAPTOR 查询流水线。

        查询处理流程：
        1. 从 RAPTOR 树构建向量存储（包含所有层级的文本和摘要）
        2. 创建上下文压缩检索器
        3. 检索与查询相关的文档
        4. 使用 LLM 基于检索到的内容生成答案

        参数：
            query (str): 用户查询。
            k (int): 检索的相关文档数量，默认 3 个。

        返回：
            Dict[str, Any]: 包含以下字段的字典：
                - query: 原始查询
                - retrieved_documents: 检索到的文档列表
                - context_used: 用于生成答案的上下文
                - answer: 生成的答案
                - model_used: 使用的模型名称
        """
        # 步骤 1：从树结构构建向量存储
        vectorstore = build_vectorstore(self.tree_results, self.embeddings)

        # 步骤 2：创建上下文压缩检索器
        retriever = create_retriever(vectorstore, self.llm)

        # 步骤 3：执行检索
        logging.info(f"正在处理查询：{query}")
        relevant_docs = retriever.get_relevant_documents(query)

        # 步骤 4：准备文档详情
        doc_details = [{"content": doc.page_content, "metadata": doc.metadata} for doc in relevant_docs]

        # 步骤 5：连接所有检索到的文档作为上下文
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # 步骤 6：创建回答生成提示词
        prompt = ChatPromptTemplate.from_template(
            "给定以下上下文，请回答问题：\n\n"
            "上下文：{context}\n\n"
            "问题：{question}\n\n"
            "答案："
        )

        # 步骤 7：创建 LLM 链并生成答案
        chain = LLMChain(llm=self.llm, prompt=prompt)
        answer = chain.run(context=context, question=query)

        # 返回完整结果
        return {
            "query": query,
            "retrieved_documents": doc_details,
            "context_used": context,
            "answer": answer,
            "model_used": self.llm.model_name,
        }


# ==================== 命令行参数解析 ====================

def parse_args():
    """
    解析命令行参数。

    允许的命令行参数：
    --path: PDF 文件路径
    --query: 测试查询
    --max_levels: RAPTOR 树的最大层级数

    返回：
        解析后的参数对象
    """
    import argparse
    parser = argparse.ArgumentParser(description="运行 RAPTORMethod")
    parser.add_argument("--path", type=str, default="../data/Understanding_Climate_Change.pdf",
                        help="要处理的 PDF 文件路径。")
    parser.add_argument("--query", type=str, default="What is the greenhouse effect?",
                        help="用于测试检索器的查询（默认：'文档的主题是什么？'）。")
    parser.add_argument('--max_levels', type=int, default=3, help="RAPTOR 树的最大层级数")
    return parser.parse_args()


# ==================== 主程序入口 ====================

# 主执行入口
if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    # 加载 PDF 文档
    # PyPDFLoader 是 LangChain 提供的 PDF 加载器
    loader = PyPDFLoader(args.path)
    documents = loader.load()
    # 提取所有页面的内容
    texts = [doc.page_content for doc in documents]

    # 创建 RAPTOR 方法实例（会自动构建树结构）
    raptor_method = RAPTORMethod(texts, max_levels=args.max_levels)

    # 运行查询并获取结果
    result = raptor_method.run(args.query)

    # 打印结果
    print(f"查询：{result['query']}")
    print(f"使用的上下文：{result['context_used']}")
    print(f"答案：{result['answer']}")
    print(f"使用的模型：{result['model_used']}")
