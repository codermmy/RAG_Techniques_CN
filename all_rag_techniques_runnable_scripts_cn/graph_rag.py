# ==================== 导入必要的库 ====================
# 图论和网络分析库
import networkx as nx
# 可视化库
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# 科学计算库
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# 系统操作库
import os
import sys
# 环境变量加载
from dotenv import load_dotenv
# LangChain 组件
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.callbacks import get_openai_callback
# 自然语言处理库
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import spacy
# 堆队列算法（用于优先级队列）
import heapq
# 命令行参数解析
import argparse
# 类型提示
from typing import List, Tuple, Dict
# 并发处理
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
# spaCy 相关
from spacy.cli import download
from spacy.lang.en import English
# Pydantic 数据模型
from pydantic import BaseModel, Field

# 将父目录添加到系统路径，因为项目使用笔记本工作方式，需要引用上级目录的模块
sys.path.append(os.path.abspath(
    os.path.join(os.getcwd(), '..')))
from helper_functions import *
from evaluation.evalute_rag import *

# 从 .env 文件加载环境变量（主要加载 OPENAI_API_KEY）
load_dotenv()

# 设置 OpenAI API 密钥环境变量，后续所有调用 OpenAI 的操作都需要这个密钥
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
# 解决 matplotlib 在某些系统上的库冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 下载 NLTK 数据（用于文本处理）
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)


# ==================== 文档处理器类 ====================
# DocumentProcessor 类负责文档的预处理工作：分割、嵌入、相似度计算

# 定义 DocumentProcessor 类
class DocumentProcessor:
    def __init__(self):
        """
        使用文本分割器和 OpenAI 嵌入初始化 DocumentProcessor。

        DocumentProcessor 的作用：
        - 将长文档分割成适合处理的小块
        - 为每个文本块创建向量嵌入
        - 计算文本之间的相似度

        属性：
        - text_splitter：RecursiveCharacterTextSplitter 实例，按字符递归分割文本
        - embeddings：OpenAIEmbeddings 实例，用于将文本转换为向量
        """
        # 文本分割器配置：
        # chunk_size=1000: 每个文本块最多 1000 个字符
        # chunk_overlap=200: 相邻文本块之间重叠 200 个字符，保持上下文连贯性
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # OpenAI 提供的文本嵌入模型，用于将文本转换为向量表示
        self.embeddings = OpenAIEmbeddings()

    def process_documents(self, documents):
        """
        处理文档列表，将它们分割成较小的块并创建向量存储。

        工作流程：
        1. 使用文本分割器将文档分割成小块
        2. 为每个块创建向量嵌入
        3. 使用 FAISS 构建向量存储，用于后续的相似度搜索

        参数：
        - documents (str 列表)：要处理的文档列表。

        返回：
        - tuple：包含：
          - splits (str 列表)：分割后的文档块列表。
          - vector_store (FAISS)：从分割后的文档块及其嵌入创建的 FAISS 向量存储。
            FAISS 支持高效的相似度搜索，可以快速找到与查询最相似的文档块。
        """
        # split_documents 将文档列表分割成更小的块
        splits = self.text_splitter.split_documents(documents)
        # from_documents 为每个块创建嵌入并构建 FAISS 索引
        vector_store = FAISS.from_documents(splits, self.embeddings)
        return splits, vector_store

    def create_embeddings_batch(self, texts, batch_size=32):
        """
        批量创建文本列表的嵌入。

        为什么要批量处理？
        - API 调用通常有速率限制
        - 批量处理可以提高效率，减少网络往返次数
        - 避免内存溢出问题

        参数：
        - texts (str 列表)：要嵌入的文本列表。
        - batch_size (int, 可选)：每批处理的文本数量。默认为 32。

        返回：
        - numpy.ndarray：输入文本的嵌入数组，形状为 (文本数量，嵌入维度)。
        """
        embeddings = []
        # 分批处理文本
        for i in range(0, len(texts), batch_size):
            # 获取当前批次的文本
            batch = texts[i:i + batch_size]
            # 调用嵌入模型处理当前批次
            batch_embeddings = self.embeddings.embed_documents(batch)
            # 将结果添加到总列表中
            embeddings.extend(batch_embeddings)
        # 转换为 numpy 数组以便后续计算
        return np.array(embeddings)

    def compute_similarity_matrix(self, embeddings):
        """
        计算给定嵌入集的余弦相似度矩阵。

        余弦相似度：
        - 衡量两个向量方向的相似程度
        - 值域为 [-1, 1]，1 表示完全相同，0 表示无关，-1 表示完全相反
        - 在文本相似度计算中非常常用

        参数：
        - embeddings (numpy.ndarray)：嵌入数组，形状为 (n_samples, n_features)。

        返回：
        - numpy.ndarray：余弦相似度矩阵，形状为 (n_samples, n_samples)。
          矩阵中第 i 行第 j 列的值表示第 i 个和第 j 个嵌入的相似度。
        """
        return cosine_similarity(embeddings)


# ==================== 知识图谱类 ====================
# 这部分定义了构建和管理知识图谱的核心类

# 定义 Concepts 类，用于 LLM 结构化输出
class Concepts(BaseModel):
    """
    用于存储从文本中提取的概念列表的数据模型。
    这个类让 LLM 以结构化格式返回提取的概念。
    """
    concepts_list: List[str] = Field(description="概念列表")


# 定义 KnowledgeGraph 类
class KnowledgeGraph:
    def __init__(self):
        """
        使用图、词干提取器和 NLP 模型初始化 KnowledgeGraph。

        KnowledgeGraph 的作用：
        - 将文档块作为节点构建图结构
        - 从文本中提取概念和命名实体
        - 基于相似度和共享概念建立节点间的连接
        - 为后续遍历和检索提供数据结构支持

        属性：
        - graph：networkx Graph 实例，存储节点（文档块）和边（连接关系）
        - lemmatizer：WordNetLemmatizer 实例，用于词形还原（如 going -> go）
        - concept_cache：字典，缓存已提取的概念避免重复计算
        - nlp：spaCy NLP 模型实例，用于命名实体识别
        - edges_threshold：浮点数，基于相似度添加边的阈值，默认 0.8
        """
        # 创建一个无向图（边没有方向性）
        self.graph = nx.Graph()
        # 词形还原器，将单词还原到基本形式
        self.lemmatizer = WordNetLemmatizer()
        # 概念缓存，避免对相同内容重复提取概念
        self.concept_cache = {}
        # 加载 spaCy NLP 模型用于命名实体识别
        self.nlp = self._load_spacy_model()
        # 边的相似度阈值，只有相似度超过此值的节点对才会建立连接
        self.edges_threshold = 0.8

    def build_graph(self, splits, llm, embedding_model):
        """
        通过添加节点、创建嵌入、提取概念和添加边来构建知识图谱。

        构建流程：
        1. _add_nodes: 将每个文档块添加为图的节点
        2. _create_embeddings: 为所有节点创建向量嵌入
        3. _extract_concepts: 从每个节点的内容中提取概念
        4. _add_edges: 基于嵌入相似度和共享概念添加边

        参数：
        - splits (列表)：文档分割列表，每个元素包含 page_content 属性。
        - llm：大型语言模型实例，用于提取概念。
        - embedding_model：嵌入模型实例，用于创建向量表示。

        返回：
        - None（结果存储在 self.graph 中）
        """
        # 步骤 1：添加节点（每个文档块作为一个节点）
        self._add_nodes(splits)
        # 步骤 2：创建嵌入（为后续计算相似度做准备）
        embeddings = self._create_embeddings(splits, embedding_model)
        # 步骤 3：提取概念（从每个节点的内容中提取关键概念）
        self._extract_concepts(splits, llm)
        # 步骤 4：添加边（基于相似度和共享概念建立连接）
        self._add_edges(embeddings)

    def _add_nodes(self, splits):
        """
        从文档分割向图添加节点。

        每个节点包含：
        - 节点 ID：文档块的索引号
        - 节点属性：content（文档块的文本内容）

        参数：
        - splits (列表)：文档分割列表。

        返回：
        - None（直接修改 self.graph）
        """
        for i, split in enumerate(splits):
            # add_node 第一个参数是节点 ID，后面是节点属性
            # 每个节点存储其对应的文档内容
            self.graph.add_node(i, content=split.page_content)

    def _create_embeddings(self, splits, embedding_model):
        """
        使用嵌入模型为文档分割创建嵌入。

        嵌入（Embedding）：
        - 将文本转换为固定长度的向量
        - 语义相似的文本在向量空间中距离更近
        - 用于计算文档块之间的相似度

        参数：
        - splits (列表)：文档分割列表。
        - embedding_model：嵌入模型实例。

        返回：
        - numpy.ndarray：文档分割的嵌入数组。
        """
        # 提取所有文档块的文本内容
        texts = [split.page_content for split in splits]
        # 调用嵌入模型批量处理
        return embedding_model.embed_documents(texts)

    def _compute_similarities(self, embeddings):
        """
        计算嵌入的余弦相似度矩阵。

        参数：
        - embeddings (numpy.ndarray)：嵌入数组。

        返回：
        - numpy.ndarray：嵌入的余弦相似度矩阵。
        """
        return cosine_similarity(embeddings)

    def _load_spacy_model(self):
        """
        加载 spaCy NLP 模型，必要时下载。

        spaCy 是什么？
        - 一个工业级的自然语言处理库
        - 提供命名实体识别、词性标注、依存句法分析等功能
        - 这里用于识别文本中的人名、组织名、地名等命名实体

        参数：
        - None

        返回：
        - spacy.Language：spaCy NLP 模型的实例。
        """
        try:
            # 尝试加载预训练模型
            return spacy.load("en_core_web_sm")
        except OSError:
            # 如果模型未安装，则下载并加载
            print("正在下载 spaCy 模型...")
            download("en_core_web_sm")
            return spacy.load("en_core_web_sm")

    def _extract_concepts_and_entities(self, content, llm):
        """
        使用 spaCy 和大型语言模型从内容中提取概念和命名实体。

        提取策略：
        1. 使用 spaCy 提取命名实体（PERSON, ORG, GPE, WORK_OF_ART）
        2. 使用 LLM 提取一般概念（非命名实体的关键词）
        3. 合并两者结果，去重后返回

        参数：
        - content (str)：要从中提取概念和实体的内容。
        - llm：大型语言模型的实例。

        返回：
        - list：提取的概念和实体列表。
        """
        # 检查缓存，避免重复计算
        if content in self.concept_cache:
            return self.concept_cache[content]

        # 第一步：使用 spaCy 提取命名实体
        # doc.ents 包含所有识别出的命名实体
        doc = self.nlp(content)
        # 提取特定类型的命名实体：
        # PERSON: 人名
        # ORG: 组织名
        # GPE: 地名（国家、城市等）
        # WORK_OF_ART: 作品名
        named_entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "WORK_OF_ART"]]

        # 第二步：使用 LLM 提取一般概念
        # 定义概念提取提示词模板
        concept_extraction_prompt = PromptTemplate(
            input_variables=["text"],
            template="从以下文本中提取关键概念（不包括命名实体）：\n\n{text}\n\n关键概念："
        )
        # 构建处理链，使用结构化输出
        concept_chain = concept_extraction_prompt | llm.with_structured_output(Concepts)
        # 调用 LLM 获取一般概念列表
        general_concepts = concept_chain.invoke({"text": content}).concepts_list

        # 第三步：合并命名实体和一般概念
        # 使用 set 去重，然后转回列表
        all_concepts = list(set(named_entities + general_concepts))

        # 缓存结果，下次遇到相同内容时直接返回
        self.concept_cache[content] = all_concepts
        return all_concepts

    def _extract_concepts(self, splits, llm):
        """
        使用多线程为所有文档分割提取概念。

        为什么使用多线程？
        - 概念提取是 I/O 密集型任务（需要调用 LLM API）
        - 多线程可以并发处理多个文档块，显著加快速度
        - ThreadPoolExecutor 自动管理线程池

        参数：
        - splits (列表)：文档分割列表。
        - llm：大型语言模型的实例。

        返回：
        - None（结果存储在 graph 节点的 'concepts' 属性中）
        """
        # 创建线程池
        with ThreadPoolExecutor() as executor:
            # 为每个文档块提交一个概念提取任务
            # future_to_node 映射：Future 对象 -> 节点 ID
            future_to_node = {executor.submit(self._extract_concepts_and_entities, split.page_content, llm): i
                              for i, split in enumerate(splits)}

            # 使用 tqdm 显示进度条
            # as_completed 按完成顺序返回 Future 对象
            for future in tqdm(as_completed(future_to_node), total=len(splits),
                               desc="提取概念和实体"):
                # 获取对应的节点 ID
                node = future_to_node[future]
                # 获取概念提取结果
                concepts = future.result()
                # 将概念存储到节点的 'concepts' 属性中
                self.graph.nodes[node]['concepts'] = concepts

    def _add_edges(self, embeddings):
        """
        基于嵌入的相似度和共享概念向图添加边。

        边的创建条件（必须同时满足）：
        1. 两个节点的嵌入相似度超过阈值（默认 0.8）
        2. 边的权重由相似度和共享概念共同决定

        参数：
        - embeddings (numpy.ndarray)：文档分割的嵌入数组。

        返回：
        - None（结果存储在 self.graph 中）
        """
        # 计算所有节点对之间的相似度矩阵
        similarity_matrix = self._compute_similarities(embeddings)
        # 获取节点总数
        num_nodes = len(self.graph.nodes)

        # 遍历所有节点对
        # 使用 tqdm 显示进度
        for node1 in tqdm(range(num_nodes), desc="添加边"):
            # 只需遍历 node1 之后的节点（无向图，避免重复）
            for node2 in range(node1 + 1, num_nodes):
                # 获取两个节点之间的相似度
                similarity_score = similarity_matrix[node1][node2]

                # 只有相似度超过阈值才添加边
                if similarity_score > self.edges_threshold:
                    # 计算两个节点共享的概念（交集）
                    shared_concepts = set(self.graph.nodes[node1]['concepts']) & set(
                        self.graph.nodes[node2]['concepts'])
                    # 计算边的权重（综合考虑相似度和共享概念）
                    edge_weight = self._calculate_edge_weight(node1, node2, similarity_score, shared_concepts)
                    # 添加边及其属性
                    self.graph.add_edge(node1, node2, weight=edge_weight,
                                        similarity=similarity_score,
                                        shared_concepts=list(shared_concepts))

    def _calculate_edge_weight(self, node1, node2, similarity_score, shared_concepts, alpha=0.7, beta=0.3):
        """
        基于相似度评分和共享概念计算边的权重。

        权重计算公式：
        weight = alpha * similarity_score + beta * normalized_shared_concepts

        其中：
        - alpha (0.7): 相似度权重的系数
        - beta (0.3): 共享概念权重的系数
        - normalized_shared_concepts: 归一化的共享概念数量

        参数：
        - node1 (int)：第一个节点。
        - node2 (int)：第二个节点。
        - similarity_score (float)：节点之间的相似度评分。
        - shared_concepts (set)：节点之间的共享概念集合。
        - alpha (float, 可选)：相似度评分的权重。默认为 0.7。
        - beta (float, 可选)：共享概念的权重。默认为 0.3。

        返回：
        - float：计算出的边权重，值越大表示两个节点连接越强。
        """
        # 计算两个节点中概念数量的较小值（用于归一化）
        max_possible_shared = min(len(self.graph.nodes[node1]['concepts']), len(self.graph.nodes[node2]['concepts']))
        # 计算归一化的共享概念数量（0 到 1 之间）
        normalized_shared_concepts = len(shared_concepts) / max_possible_shared if max_possible_shared > 0 else 0
        # 加权计算最终边权重
        return alpha * similarity_score + beta * normalized_shared_concepts

    def _lemmatize_concept(self, concept):
        """
        对给定概念进行词形还原。

        什么是词形还原（Lemmatization）？
        - 将单词的不同形式还原到词典中的基本形式
        - 例如：going -> go, better -> good, mice -> mouse
        - 与词干提取（Stemming）不同，词形还原会考虑词性和上下文

        为什么要做词形还原？
        - 避免将同一概念的不同形式当作不同概念
        - 例如："climates" 和 "climate" 应该视为同一概念

        参数：
        - concept (str)：要进行词形还原的概念。

        返回：
        - str：词形还原后的概念（小写形式）。
        """
        # 将概念按空格分割成单词，对每个单词进行词形还原，然后重新组合
        return ' '.join([self.lemmatizer.lemmatize(word) for word in concept.lower().split()])


# ==================== 查询引擎类 ====================
# 这部分定义了用于处理查询的 QueryEngine 类

# 定义 AnswerCheck 类，用于 LLM 结构化输出
class AnswerCheck(BaseModel):
    """
    用于检查答案完整性的数据模型。
    让 LLM 判断当前上下文是否足以完整回答查询。
    """
    is_complete: bool = Field(description="当前上下文是否提供对查询的完整回答")
    answer: str = Field(description="基于当前上下文的回答，如果有")


# 定义 QueryEngine 类
class QueryEngine:
    def __init__(self, vector_store, knowledge_graph, llm):
        """
        初始化查询引擎。

        QueryEngine 的作用：
        - 接收用户查询
        - 从向量存储检索相关文档
        - 在知识图谱中遍历扩展上下文
        - 生成最终回答

        参数：
        - vector_store: FAISS 向量存储，用于相似度搜索
        - knowledge_graph: KnowledgeGraph 实例，包含文档图结构
        - llm: 大型语言模型，用于生成回答和判断答案完整性

        属性：
        - max_context_length: 最大上下文长度（4000 字符），防止超出 LLM 输入限制
        - answer_check_chain: 用于判断答案是否完整的处理链
        """
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.llm = llm
        self.max_context_length = 4000  # 限制上下文长度，避免超出 LLM 的 token 限制
        # 创建答案检查链
        self.answer_check_chain = self._create_answer_check_chain()

    def _create_answer_check_chain(self):
        """
        创建一个链来检查上下文是否提供对查询的完整回答。

        这个链的工作流程：
        1. 使用提示词模板组织查询和上下文
        2. 让 LLM 判断是否已有完整答案
        3. 如果有完整答案，让 LLM 提供回答

        参数：
        - None

        返回：
        - Chain：检查上下文是否提供完整回答的链。
        """
        # 定义答案检查提示词模板
        answer_check_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="给定查询：'{query}'\n\n和当前上下文：\n{context}\n\n此上下文是否提供对查询的完整回答？如果是，请提供答案。如果否，请说明答案不完整。\n\n是否完整回答（是/否）：\n答案（如果完整）："
        )
        # 构建处理链，使用结构化输出确保返回 AnswerCheck 格式
        return answer_check_prompt | self.llm.with_structured_output(AnswerCheck)

    def _check_answer(self, query: str, context: str) -> Tuple[bool, str]:
        """
        检查当前上下文是否提供对查询的完整回答。

        工作原理：
        1. 将查询和上下文输入 LLM
        2. LLM 判断是否能基于上下文完整回答查询
        3. 如果可以，LLM 提供具体答案

        参数：
        - query (str)：要回答的查询。
        - context (str)：当前上下文（已积累的文档内容）。

        返回：
        - tuple：包含：
          - is_complete (bool)：上下文是否提供完整回答。
          - answer (str)：基于上下文的回答（如果完整）。
        """
        # 调用答案检查链，获取 LLM 判断结果
        response = self.answer_check_chain.invoke({"query": query, "context": context})
        return response.is_complete, response.answer

    def _expand_context(self, query: str, relevant_docs) -> Tuple[str, List[int], Dict[int, str], str]:
        """
        通过遍历知识图谱来扩展上下文，使用改进的 Dijkstra 算法。

        算法核心思想：
        这是一个智能的图谱遍历过程，类似于地图导航中的"最短路径"算法，
        但这里找的是"最强连接路径"，优先探索与当前查询最相关、连接最强的信息。

        详细工作流程：

        1. 初始化阶段：
           - 从最相关的文档节点开始（通过向量检索找到）
           - 使用优先级队列（最小堆）管理遍历顺序
           - 距离 = 1/连接强度，连接越强距离越短，优先被访问

        2. 遍历循环：
           a. 从优先级队列取出"距离"最近的节点（连接最强的）
           b. 检查当前累积的上下文是否已足够回答查询
           c. 如果答案不完整，探索该节点的邻居
           d. 更新邻居节点的距离（如果找到更强连接）
           e. 将邻居加入优先级队列

        3. 概念引导：
           - 记录已访问过的概念
           - 只有当邻居引入新概念时才扩展
           - 避免在相同概念上打转

        4. 终止条件：
           - 找到完整答案（提前终止）
           - 优先级队列为空（所有可达节点已探索）

        参数：
        - query (str)：要回答的查询。
        - relevant_docs (List[Document])：初始相关文档列表，作为遍历起点。

        返回：
        - tuple：包含：
          - expanded_context (str)：从遍历节点累积的完整上下文。
          - traversal_path (List[int]): 访问的节点索引序列（遍历路径）。
          - filtered_content (Dict[int, str])：节点索引到其内容的映射。
          - final_answer (str)：找到的最终答案（如果有）。
        """
        # ==================== 初始化变量 ====================
        expanded_context = ""  # 累积的上下文内容
        traversal_path = []  # 记录遍历的节点路径
        visited_concepts = set()  # 记录已访问的概念，避免重复探索
        filtered_content = {}  # 节点 ID 到内容的映射
        final_answer = ""  # 如果找到完整答案，存储在这里

        # 优先级队列：(距离，节点 ID)
        # 使用最小堆，距离小的（连接强的）优先出队
        priority_queue = []
        # 距离字典：记录到每个节点的最佳已知距离
        distances = {}

        print("\n遍历知识图谱：")

        # ==================== 初始化优先级队列 ====================
        # 用相关文档中的最近节点初始化队列
        for doc in relevant_docs:
            # 在向量存储中找到与当前文档最相似的节点
            closest_nodes = self.vector_store.similarity_search_with_score(doc.page_content, k=1)
            closest_node_content, similarity_score = closest_nodes[0]

            # 在知识图谱中找到对应的节点 ID
            closest_node = next(n for n in self.knowledge_graph.graph.nodes if
                                self.knowledge_graph.graph.nodes[n]['content'] == closest_node_content.page_content)

            # 计算优先级（相似度的倒数，用于最小堆行为）
            # 相似度越高，优先级值越小，越先被处理
            priority = 1 / similarity_score
            # 将节点加入优先级队列
            heapq.heappush(priority_queue, (priority, closest_node))
            # 记录初始距离
            distances[closest_node] = priority

        step = 0
        # ==================== 主遍历循环 ====================
        while priority_queue:
            # 获取优先级最高的节点（距离值最低，即连接最强）
            current_priority, current_node = heapq.heappop(priority_queue)

            # 如果我们已经找到到该节点的更好路径，则跳过（避免重复处理）
            if current_priority > distances.get(current_node, float('inf')):
                continue

            # 如果该节点还未被访问过
            if current_node not in traversal_path:
                step += 1
                traversal_path.append(current_node)
                # 获取节点的内容和概念
                node_content = self.knowledge_graph.graph.nodes[current_node]['content']
                node_concepts = self.knowledge_graph.graph.nodes[current_node]['concepts']

                # 将节点内容添加到累积的上下文中
                filtered_content[current_node] = node_content
                expanded_context += "\n" + node_content if expanded_context else node_content

                # 记录当前步骤以进行调试和可视化
                print(f"\n步骤 {step} - 节点 {current_node}：")
                print(f"内容：{node_content[:100]}...")  # 只打印前 100 个字符
                print(f"概念：{', '.join(node_concepts)}")
                print("-" * 50)

                # 检查当前上下文是否有完整答案
                is_complete, answer = self._check_answer(query, expanded_context)
                if is_complete:
                    final_answer = answer
                    print(f"找到完整答案！")
                    break

                # 处理当前节点的概念（进行词形还原后加入已访问集合）
                node_concepts_set = set(self.knowledge_graph._lemmatize_concept(c) for c in node_concepts)
                if not node_concepts_set.issubset(visited_concepts):
                    visited_concepts.update(node_concepts_set)

                    # ==================== 探索邻居节点 ====================
                    for neighbor in self.knowledge_graph.graph.neighbors(current_node):
                        # 获取边的数据（包含权重、相似度、共享概念等）
                        edge_data = self.knowledge_graph.graph[current_node][neighbor]
                        edge_weight = edge_data['weight']

                        # 计算到邻居的新距离（优先级）
                        # 距离 = 当前距离 + 1/边权重
                        # 边权重越大（连接越强），新距离越小，优先级越高
                        distance = current_priority + (1 / edge_weight)

                        # 如果我们找到到邻居的更强连接，更新其距离
                        if distance < distances.get(neighbor, float('inf')):
                            distances[neighbor] = distance
                            heapq.heappush(priority_queue, (distance, neighbor))

                            # 处理邻居节点（如果还不在遍历路径中）
                            if neighbor not in traversal_path:
                                step += 1
                                traversal_path.append(neighbor)
                                neighbor_content = self.knowledge_graph.graph.nodes[neighbor]['content']
                                neighbor_concepts = self.knowledge_graph.graph.nodes[neighbor]['concepts']

                                filtered_content[neighbor] = neighbor_content
                                expanded_context += "\n" + neighbor_content if expanded_context else neighbor_content

                                # 记录邻居节点信息
                                print(f"\n步骤 {step} - 节点 {neighbor}（{current_node}的邻居）：")
                                print(f"内容：{neighbor_content[:100]}...")
                                print(f"概念：{', '.join(neighbor_concepts)}")
                                print("-" * 50)

                                # 检查添加邻居内容后是否有完整答案
                                is_complete, answer = self._check_answer(query, expanded_context)
                                if is_complete:
                                    final_answer = answer
                                    print(f"找到完整答案！")
                                    break

                                # 处理邻居的概念
                                neighbor_concepts_set = set(
                                    self.knowledge_graph._lemmatize_concept(c) for c in neighbor_concepts)
                                if not neighbor_concepts_set.issubset(visited_concepts):
                                    visited_concepts.update(neighbor_concepts_set)

                # 如果找到最终答案，跳出主循环
                if final_answer:
                    break

        # ==================== 生成最终答案 ====================
        # 如果没有找到完整答案，使用 LLM 基于收集的上下文生成一个
        if not final_answer:
            print("\n生成最终答案...")
            response_prompt = PromptTemplate(
                input_variables=["query", "context"],
                template="基于以下上下文，请回答查询。\n\n上下文：{context}\n\n查询：{query}\n\n答案："
            )
            response_chain = response_prompt | self.llm
            input_data = {"query": query, "context": expanded_context}
            final_answer = response_chain.invoke(input_data)

        return expanded_context, traversal_path, filtered_content, final_answer

    def query(self, query: str) -> Tuple[str, List[int], Dict[int, str]]:
        """
        通过检索相关文档、扩展上下文并生成最终答案来处理查询。

        完整工作流程：
        1. 使用向量存储检索相关文档
        2. 调用 _expand_context 在知识图谱中遍历扩展
        3. 如果遍历中找到完整答案，直接返回
        4. 否则使用 LLM 基于收集的上下文生成答案
        5. 记录并打印 token 使用情况和成本

        参数：
        - query (str)：要回答的查询。

        返回：
        - tuple：包含：
          - final_answer (str)：查询的最终答案。
          - traversal_path (list)：知识图谱中节点的遍历路径。
          - filtered_content (dict)：节点的过滤内容。
        """
        # 使用 OpenAI 回调跟踪 token 使用和成本
        with get_openai_callback() as cb:
            print(f"\n处理查询：{query}")
            # 检索相关文档作为遍历起点
            relevant_docs = self._retrieve_relevant_documents(query)
            # 扩展上下文并获取答案
            expanded_context, traversal_path, filtered_content, final_answer = self._expand_context(query,
                                                                                                    relevant_docs)

            # 如果遍历过程中没有找到完整答案，则使用 LLM 生成
            if not final_answer:
                print("\n生成最终答案...")
                response_prompt = PromptTemplate(
                    input_variables=["query", "context"],
                    template="基于以下上下文，请回答查询。\n\n上下文：{context}\n\n查询：{query}\n\n答案："
                )

                response_chain = response_prompt | self.llm
                input_data = {"query": query, "context": expanded_context}
                response = response_chain.invoke(input_data)
                final_answer = response
            else:
                print("\n遍历过程中找到完整答案。")

            # 打印最终答案和成本信息
            print(f"\n最终答案：{final_answer}")
            print(f"\n总令牌数：{cb.total_tokens}")
            print(f"提示令牌数：{cb.prompt_tokens}")
            print(f"完成令牌数：{cb.completion_tokens}")
            print(f"总成本（美元）：${cb.total_cost}")

        return final_answer, traversal_path, filtered_content

    def _retrieve_relevant_documents(self, query: str):
        """
        使用向量存储基于查询检索相关文档。

        检索流程：
        1. 创建基础检索器（从向量存储）
        2. 创建 LLM 压缩器（提取与查询最相关的部分）
        3. 组合成上下文压缩检索器
        4. 执行检索

        为什么需要上下文压缩？
        - 检索到的文档可能包含不相关的部分
        - LLM 压缩器可以提取最相关的句子或段落
        - 减少噪声，提高答案质量

        参数：
        - query (str)：要回答的查询。

        返回：
        - list：相关文档列表（经过压缩提取）。
        """
        print("\n检索相关文档...")
        # 创建基础检索器，使用相似度搜索，返回最相关的 5 个文档
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        # 创建 LLM 链提取器，用于从文档中提取相关内容
        compressor = LLMChainExtractor.from_llm(self.llm)
        # 创建上下文压缩检索器（组合基础检索器和压缩器）
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
        # 执行检索
        return compression_retriever.invoke(query)


# ==================== 可视化类 ====================
# 这部分定义了用于可视化知识图谱遍历的 Visualizer 类

# 定义可视化类
class Visualizer:
    @staticmethod
    def visualize_traversal(graph, traversal_path):
        """
        在知识图谱上可视化遍历路径，突出显示节点、边和遍历路径。

        可视化元素说明：
        - 节点：表示文档块，用浅蓝色圆圈表示
        - 边：节点之间的连接，颜色深浅表示权重（连接强度）
        - 遍历路径：用红色虚线箭头表示
        - 起始节点：用绿色突出显示
        - 结束节点：用红色突出显示

        参数：
        - graph (networkx.Graph)：包含节点和边的知识图谱。
        - traversal_path (int 列表)：表示遍历路径的节点索引列表。

        返回：
        - None（直接显示 matplotlib 图形）
        """
        # 创建一个有向图用于可视化遍历路径
        traversal_graph = nx.DiGraph()

        # 从原始图复制所有节点和边
        for node in graph.nodes():
            traversal_graph.add_node(node)
        for u, v, data in graph.edges(data=True):
            traversal_graph.add_edge(u, v, **data)

        # 创建图形和坐标轴
        fig, ax = plt.subplots(figsize=(16, 12))  # 16x12 英寸的画布

        # 使用 spring_layout 生成节点位置（模拟物理弹簧的布局算法）
        # k=1 控制节点间的理想距离，iterations=50 是优化迭代次数
        pos = nx.spring_layout(traversal_graph, k=1, iterations=50)

        # ==================== 绘制边 ====================
        # 获取所有边
        edges = traversal_graph.edges()
        # 获取每条边的权重
        edge_weights = [traversal_graph[u][v].get('weight', 0.5) for u, v in edges]
        # 绘制边，颜色根据权重变化（蓝色渐变）
        nx.draw_networkx_edges(traversal_graph, pos,
                               edgelist=edges,
                               edge_color=edge_weights,
                               edge_cmap=plt.cm.Blues,
                               width=2,
                               ax=ax)

        # ==================== 绘制节点 ====================
        # 用浅蓝色绘制所有节点
        nx.draw_networkx_nodes(traversal_graph, pos,
                               node_color='lightblue',
                               node_size=3000,  # 节点大小
                               ax=ax)

        # ==================== 绘制遍历路径（弯曲箭头） ====================
        edge_offset = 0.1  # 曲线偏移量，避免箭头与节点重叠
        for i in range(len(traversal_path) - 1):
            start = traversal_path[i]
            end = traversal_path[i + 1]
            start_pos = pos[start]
            end_pos = pos[end]

            # 计算曲线的控制点（用于创建弧形路径）
            mid_point = ((start_pos[0] + end_pos[0]) / 2, (start_pos[1] + end_pos[1]) / 2)
            control_point = (mid_point[0] + edge_offset, mid_point[1] + edge_offset)

            # 绘制弯曲箭头
            arrow = patches.FancyArrowPatch(start_pos, end_pos,
                                            connectionstyle=f"arc3,rad={0.3}",  # 弧度 0.3
                                            color='red',
                                            arrowstyle="->",
                                            mutation_scale=20,
                                            linestyle='--',  # 虚线
                                            linewidth=2,
                                            zorder=4)  # 确保箭头在其他元素上层
            ax.add_patch(arrow)

        # ==================== 准备和绘制标签 ====================
        labels = {}
        # 为遍历路径中的节点添加编号标签
        for i, node in enumerate(traversal_path):
            concepts = graph.nodes[node].get('concepts', [])
            # 标签格式："序号。第一个概念"
            label = f"{i + 1}. {concepts[0] if concepts else ''}"
            labels[node] = label

        # 为不在遍历路径中的节点添加标签
        for node in traversal_graph.nodes():
            if node not in labels:
                concepts = graph.nodes[node].get('concepts', [])
                labels[node] = concepts[0] if concepts else ''

        # 绘制标签（字体大小 8，加粗）
        nx.draw_networkx_labels(traversal_graph, pos, labels, font_size=8, font_weight="bold", ax=ax)

        # ==================== 突出显示起始和结束节点 ====================
        start_node = traversal_path[0]
        end_node = traversal_path[-1]

        # 起始节点用浅绿色突出显示
        nx.draw_networkx_nodes(traversal_graph, pos,
                               nodelist=[start_node],
                               node_color='lightgreen',
                               node_size=3000,
                               ax=ax)

        # 结束节点用浅红色突出显示
        nx.draw_networkx_nodes(traversal_graph, pos,
                               nodelist=[end_node],
                               node_color='lightcoral',
                               node_size=3000,
                               ax=ax)

        # 设置标题
        ax.set_title("图谱遍历流程")
        # 隐藏坐标轴
        ax.axis('off')

        # ==================== 添加颜色条（边权重图例） ====================
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues,
                                   norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('边权重', rotation=270, labelpad=15)

        # ==================== 添加图例 ====================
        regular_line = plt.Line2D([0], [0], color='blue', linewidth=2, label='常规边')
        traversal_line = plt.Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='遍历路径')
        start_point = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=15,
                                 label='起始节点')
        end_point = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', markersize=15,
                               label='结束节点')
        legend = plt.legend(handles=[regular_line, traversal_line, start_point, end_point], loc='upper left',
                            bbox_to_anchor=(0, 1), ncol=2)
        legend.get_frame().set_alpha(0.8)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def print_filtered_content(traversal_path, filtered_content):
        """
        按遍历顺序打印访问节点的过滤内容。

        用途：
        - 用于调试，查看遍历过程中访问了哪些节点
        - 帮助理解系统如何逐步收集信息来回答问题

        参数：
        - traversal_path (int 列表)：表示遍历路径的节点索引列表。
        - filtered_content (int: str 字典)：将节点索引映射到其过滤内容的字典。

        返回：
        - None（直接打印输出）
        """
        print("\n按遍历顺序访问节点的过滤内容：")
        for i, node in enumerate(traversal_path):
            print(f"\n步骤 {i + 1} - 节点 {node}：")
            # 打印前 200 个字符，避免输出过长
            print(
                f"过滤内容：{filtered_content.get(node, '无可用过滤内容')[:200]}...")
            print("-" * 50)


# ==================== GraphRAG 主类 ====================
# 这是整个系统的封装类，整合了所有组件

# 定义图谱 RAG 类
class GraphRAG:
    def __init__(self, documents):
        """
        初始化 GraphRAG 系统，包含用于文档处理、知识图谱构建、查询和可视化的组件。

        GraphRAG 系统架构：
        1. DocumentProcessor：文档预处理（分割、嵌入）
        2. KnowledgeGraph：知识图谱构建（节点、边、概念提取）
        3. QueryEngine：查询处理（检索、遍历、回答生成）
        4. Visualizer：可视化遍历路径

        参数：
        - documents (str 列表)：要处理的文档列表。

        属性：
        - llm：ChatOpenAI 实例，用于生成响应和提取概念。
        - embedding_model：OpenAIEmbeddings 实例，用于文档嵌入。
        - document_processor：DocumentProcessor 实例，处理文档。
        - knowledge_graph：KnowledgeGraph 实例，构建和管理图谱。
        - query_engine：QueryEngine 实例，处理查询（初始化为 None）。
        - visualizer：Visualizer 实例，可视化图谱遍历。
        """
        # 初始化 LLM，使用 GPT-4o-mini 模型
        # temperature=0 使输出最确定，max_tokens=4000 限制响应长度
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=4000)
        # OpenAI 提供的嵌入模型
        self.embedding_model = OpenAIEmbeddings()
        # 文档处理器
        self.document_processor = DocumentProcessor()
        # 知识图谱
        self.knowledge_graph = KnowledgeGraph()
        # 查询引擎（稍后初始化）
        self.query_engine = None
        # 可视化器
        self.visualizer = Visualizer()
        # 处理传入的文档
        self.process_documents(documents)

    def process_documents(self, documents):
        """
        处理文档列表，将它们分割成块，嵌入它们，并构建知识图谱。

        工作流程：
        1. 使用 DocumentProcessor 分割文档并创建向量存储
        2. 使用 KnowledgeGraph 构建知识图谱（添加节点、边、概念）
        3. 初始化 QueryEngine 用于后续查询

        参数：
        - documents (str 列表)：要处理的文档列表。

        返回：
        - None
        """
        # 处理文档：分割成块并创建向量存储
        splits, vector_store = self.document_processor.process_documents(documents)
        # 构建知识图谱：添加节点、创建嵌入、提取概念、添加边
        self.knowledge_graph.build_graph(splits, self.llm, self.embedding_model)
        # 创建查询引擎，传入向量存储、知识图谱和 LLM
        self.query_engine = QueryEngine(vector_store, self.knowledge_graph, self.llm)

    def query(self, query: str):
        """
        通过从知识图谱检索相关信息并可视化遍历路径来处理查询。

        完整流程：
        1. 调用 QueryEngine.query() 检索信息并生成答案
        2. 如果有遍历路径，可视化显示
        3. 返回最终答案

        参数：
        - query (str)：要回答的查询。

        返回：
        - str：查询的响应。
        """
        # 执行查询，获取答案、遍历路径和内容
        response, traversal_path, filtered_content = self.query_engine.query(query)

        # 如果有遍历路径，可视化显示
        if traversal_path:
            self.visualizer.visualize_traversal(self.knowledge_graph.graph, traversal_path)
        else:
            print("无遍历路径可可视化。")

        return response


# ==================== 命令行参数解析 ====================

def parse_args():
    """
    解析命令行参数。

    允许的命令行参数：
    --path: PDF 文件路径
    --query: 检索文档的查询

    返回：
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(description="GraphRAG 系统")
    parser.add_argument('--path', type=str, default="../data/Understanding_Climate_Change.pdf",
                        help='PDF 文件路径。')
    parser.add_argument('--query', type=str, default='what is the main cause of climate change?',
                        help='检索文档的查询。')
    return parser.parse_args()


if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()

    # 加载 PDF 文档
    # PyPDFLoader 专门用于加载 PDF 文件
    loader = PyPDFLoader(args.path)
    documents = loader.load()
    # 只取前 10 页用于演示（实际应用中可以处理全部）
    documents = documents[:10]

    # 创建 GraphRAG 实例
    graph_rag = GraphRAG(documents)

    # 处理文档并创建图谱（在初始化时已自动执行）
    graph_rag.process_documents(documents)

    # 执行查询并从图谱 RAG 获取检索信息
    response = graph_rag.query(args.query)
