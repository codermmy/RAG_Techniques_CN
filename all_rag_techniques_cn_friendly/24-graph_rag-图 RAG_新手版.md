# 🌟 新手入门：图 RAG（GraphRAG）

> **💡 给新手的说明**
> - **难度等级**：⭐⭐⭐⭐☆（较难）
> - **预计学习时间**：60-90 分钟
> - **前置知识**：了解基本的 RAG 概念，有 Python 编程经验
> - **学完你将掌握**：如何用知识图谱增强检索系统，让 AI 理解概念之间的关系
>
> **🤔 为什么要学这个？** 传统 RAG 像查字典，找到相关词但不知道关系；GraphRAG 像有一张知识地图，能顺藤摸瓜找到所有关联信息！

---

## 📖 核心概念理解

### 什么是 GraphRAG？

**GraphRAG** = Graph（图）+ RAG（检索增强生成）

它用**知识图谱**的方式来组织和检索信息，不仅能找到相关文档，还能理解文档之间的**关系**。

### 通俗理解：图书馆的两种检索方式

#### 传统 RAG 方式
想象你在图书馆找书：
> 你："我想看关于气候变化的书"
> 管理员：给你 5 本包含"气候变化"这个词的书

问题：这些书之间有什么关系？哪本更核心？不知道。

#### GraphRAG 方式
> 你："我想看关于气候变化的书"
> 管理员：拿出一张知识地图，说：
> - "气候变化"连接着这些概念：
>   - → 温室效应（强相关）
>   - → 海平面上升（后果）
>   - → 碳排放（原因）
>   - → 可再生能源（解决方案）
> 然后顺着这张地图，带你找到最核心的几本书

### 核心组件一览

| 组件 | 作用 | 生活化比喻 |
|------|------|-----------|
| DocumentProcessor | 处理文档，分割成块 | 图书管理员把厚书拆成章节 |
| KnowledgeGraph | 构建知识图谱 | 绘制概念之间的关系地图 |
| QueryEngine | 处理查询，遍历图谱 | 拿着地图找信息的向导 |
| Visualizer | 可视化图谱和检索路径 | 把地图画出来给你看 |

### 关键概念解释

**📌 知识图谱（Knowledge Graph）**
- 由**节点**（概念/实体）和**边**（关系）组成
- 比如："巴黎" --(是首都)--> "法国"

**📌 节点（Node）**
- 图中的一个点，代表一个文档块或概念
- 比如：一段关于"温室效应"的文字

**📌 边（Edge）**
- 连接两个节点的线，代表关系
- 边有权重，表示关系强弱

**📌 遍历（Traversal）**
- 沿着图谱中的边走，从一个节点到另一个节点
- 像顺着地图上的路走

---

## 🛠️ 第一步：环境准备

### 📖 这是什么？

安装运行 GraphRAG 所需的所有工具包。由于涉及图谱构建和可视化，需要的包比较多。

### 💻 完整代码

```python
# 安装所需的包
# 每个包的作用：
# - faiss-cpu: 向量相似度搜索
# - futures: 并行处理，加快速度
# - langchain: RAG 框架
# - langchain-openai: OpenAI 集成
# - matplotlib: 绘图可视化
# - networkx: 图论操作
# - nltk: 自然语言处理
# - numpy: 数值计算
# - python-dotenv: 环境变量管理
# - scikit-learn: 机器学习工具
# - spacy: NLP 实体提取
# - tqdm: 进度条显示
!pip install faiss-cpu futures langchain langchain-openai matplotlib networkx nltk numpy python-dotenv scikit-learn spacy tqdm

# 克隆仓库以访问辅助函数
!git clone https://github.com/NirDiamant/RAG_TECHNIQUES.git
import sys
sys.path.append('RAG_TECHNIQUES')
```

> **💡 代码解释**
> - 这一长串包名不用怕，每个都有特定用途
> - `faiss-cpu`：快速查找相似文档
> - `networkx`：创建和操作图结构
> - `spacy`：从文本中提取人名、地名等实体
> - `matplotlib`：画图，可视化图谱
>
> **⚠️ 新手注意**
> - 安装可能需要 5-10 分钟，耐心等待
> - 如果某个包安装失败，可以尝试单独安装
> - spacy 需要额外下载语言模型（代码后面会处理）
>
> **❓ 常见问题**
>
> **Q: 一定要安装这么多包吗？**
>
> A: GraphRAG 确实比较复杂，但每个包都有用：
> - 基础功能：langchain, openai
> - 图谱构建：networkx, spacy
> - 可视化：matplotlib
> - 性能优化：faiss, futures

### 导入库和设置环境

```python
import networkx as nx
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.callbacks import get_openai_callback

from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import List, Tuple, Dict
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import spacy
import heapq

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np

from spacy.cli import download
from spacy.lang.en import English

# 从.env 文件加载环境变量
load_dotenv()

# 设置 OpenAI API 密钥
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解决 matplotlib 可能的冲突

# 下载 NLTK 数据（用于词形还原）
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
```

> **💡 代码解释**
>
> **核心库说明：**
>
> 1. **networkx (nx)**：创建和操作复杂网络
> 2. **heapq**：优先队列，用于图遍历算法
> 3. **WordNetLemmatizer**：词形还原，把"running"变成"run"
> 4. **ThreadPoolExecutor**：多线程并行处理，加快速度
>
> **⚠️ 新手注意**
> - `os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"` 是解决某些系统上的库冲突
> - nltk 下载只需要运行一次，会缓存到本地
> - 如果 spacy 下载失败，代码中有自动处理

---

## 🛠️ 第二步：定义 DocumentProcessor 类

### 📖 这是什么？

DocumentProcessor 负责把原始文档处理成适合构建图谱的格式。就像厨师准备食材：洗菜、切菜、分类。

### 💻 完整代码

```python
class DocumentProcessor:
    def __init__(self):
        """
        初始化文档处理器

        属性：
        - text_splitter: 文本分割器，把长文档切成小块
          - chunk_size=1000: 每块最多 1000 个字符
          - chunk_overlap=200: 相邻块重叠 200 个字符（保持上下文连贯）
        - embeddings: 嵌入模型，把文字转成数字向量
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.embeddings = OpenAIEmbeddings()

    def process_documents(self, documents):
        """
        处理文档列表

        参数:
            documents: 文档列表（Document 对象）

        返回:
            tuple: (分割后的文本块列表，向量存储)
        """
        # split_documents 把每个文档分割成多个块
        splits = self.text_splitter.split_documents(documents)

        # from_documents 为所有块创建嵌入并存储到 FAISS
        vector_store = FAISS.from_documents(splits, self.embeddings)

        return splits, vector_store

    def create_embeddings_batch(self, texts, batch_size=32):
        """
        批量创建嵌入

        参数:
            texts: 文本列表
            batch_size: 每批处理的数量（默认 32）

        返回:
            numpy 数组：所有文本的嵌入

        为什么批量处理？
        - API 调用有速率限制
        - 批量处理效率更高
        """
        embeddings = []
        # 每次处理 batch_size 个文本
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.embeddings.embed_documents(batch)
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)

    def compute_similarity_matrix(self, embeddings):
        """
        计算余弦相似度矩阵

        参数:
            embeddings: 嵌入数组

        返回:
            相似度矩阵（n x n 的二维数组）

        余弦相似度是什么？
        - 衡量两个向量方向的相似程度
        - 值在 -1 到 1 之间，越接近 1 越相似
        """
        return cosine_similarity(embeddings)
```

> **💡 代码解释**
>
> **文本分割原理：**
> ```
> 原文：[AAAAAAAAAAAAAAAAAAAA] (2000 字符)
> 分割后：
> 块 1: [AAAAAAAAAA(1000)]
> 块 2:       [AAAAAAAAAA(1000)]  ← 重叠 200 字符
> 块 3:             [AAAAAAAAAA(1000)]
> ```
> 重叠部分确保上下文不被切断
>
> **⚠️ 新手注意**
> - `chunk_size` 太大：每块内容多，但可能包含多个主题
> - `chunk_size` 太小：主题单一，但可能丢失上下文
> - 1000 是个常用值，可根据实际情况调整
>
> **❓ 常见问题**
>
> **Q: 为什么需要重叠（overlap）？**
>
> A: 想象一句话被切成两半：
> - 块 1 结尾："全球变暖的主要原因是..."
> - 块 2 开头："...二氧化碳排放增加"
> 单独看都不完整。重叠确保关键信息不被切断。

---

## 🛠️ 第三步：定义 KnowledgeGraph 类

### 📖 这是什么？

KnowledgeGraph 是 GraphRAG 的核心，负责构建知识图谱。它把文档块变成节点，找出它们之间的关系并连线。

### 💻 完整代码

```python
from langchain.schema import BaseModel
from pydantic import Field

# 定义 Concepts 类，用于结构化 LLM 的输出
class Concepts(BaseModel):
    concepts_list: List[str] = Field(description="概念列表")

class KnowledgeGraph:
    def __init__(self):
        """
        初始化知识图谱

        属性:
        - graph: networkx 图对象
        - lemmatizer: 词形还原器（把"running"→"run"）
        - concept_cache: 缓存已提取的概念，避免重复处理
        - nlp: spaCy NLP 模型
        - edges_threshold: 添加边的相似度阈值（0.8 表示 80% 相似）
        """
        self.graph = nx.Graph()
        self.lemmatizer = WordNetLemmatizer()
        self.concept_cache = {}
        self.nlp = self._load_spacy_model()
        self.edges_threshold = 0.8

    def build_graph(self, splits, llm, embedding_model):
        """
        构建知识图谱的主流程

        参数:
            splits: 文档分割列表
            llm: 语言模型
            embedding_model: 嵌入模型

        流程:
        1. 添加节点（每个分割是一个节点）
        2. 创建嵌入向量
        3. 提取概念
        4. 添加边（基于相似度和共享概念）
        """
        self._add_nodes(splits)
        embeddings = self._create_embeddings(splits, embedding_model)
        self._extract_concepts(splits, llm)
        self._add_edges(embeddings)

    def _add_nodes(self, splits):
        """
        从文档分割添加节点

        每个节点包含:
        - 索引 i（唯一标识）
        - content: 文本内容
        """
        for i, split in enumerate(splits):
            # add_node 第一个参数是节点 ID，后面是属性
            self.graph.add_node(i, content=split.page_content)

    def _create_embeddings(self, splits, embedding_model):
        """
        为所有分割创建嵌入

        返回:
            numpy 数组：每个节点对应的嵌入向量
        """
        texts = [split.page_content for split in splits]
        return embedding_model.embed_documents(texts)

    def _compute_similarities(self, embeddings):
        """
        计算所有节点对的相似度

        返回:
            n x n 的相似度矩阵
            matrix[i][j] = 节点 i 和节点 j 的相似度
        """
        return cosine_similarity(embeddings)

    def _load_spacy_model(self):
        """
        加载 spaCy NLP 模型

        spaCy 是什么？
        - 工业级 NLP 库
        - 可以提取人名、地名、组织名等实体

        如果模型不存在，自动下载
        """
        try:
            return spacy.load("en_core_web_sm")  # 小型模型，速度快
        except OSError:
            print("正在下载 spaCy 模型...")
            download("en_core_web_sm")
            return spacy.load("en_core_web_sm")

    def _extract_concepts_and_entities(self, content, llm):
        """
        从内容中提取概念和命名实体

        参数:
            content: 文本内容
            llm: 语言模型

        返回:
            概念列表

        两种提取方式结合:
        1. spaCy: 提取命名实体（人名、地名等）
        2. LLM: 提取一般概念（主题、思想）
        """
        # 检查缓存，避免重复处理
        if content in self.concept_cache:
            return self.concept_cache[content]

        # 1. 使用 spaCy 提取命名实体
        doc = self.nlp(content)
        named_entities = [
            ent.text for ent in doc.ents
            if ent.label_ in ["PERSON", "ORG", "GPE", "WORK_OF_ART"]
        ]
        # PERSON: 人名, ORG: 组织，GPE: 地名，WORK_OF_ART: 作品名

        # 2. 使用 LLM 提取一般概念
        concept_extraction_prompt = PromptTemplate(
            input_variables=["text"],
            template="从以下文本中提取关键概念（不包括命名实体）：\n\n{text}\n\n关键概念："
        )
        # with_structured_output 确保输出是指定的格式
        concept_chain = concept_extraction_prompt | llm.with_structured_output(Concepts)
        general_concepts = concept_chain.invoke({"text": content}).concepts_list

        # 3. 合并两类概念，去重
        all_concepts = list(set(named_entities + general_concepts))

        # 存入缓存
        self.concept_cache[content] = all_concepts
        return all_concepts

    def _extract_concepts(self, splits, llm):
        """
        为所有分割提取概念（使用多线程加速）

        ThreadPoolExecutor 是什么？
        - 同时运行多个任务
        - 比一个一个处理快得多
        """
        with ThreadPoolExecutor() as executor:
            # 为每个分割提交一个任务
            future_to_node = {
                executor.submit(
                    self._extract_concepts_and_entities,
                    split.page_content,
                    llm
                ): i
                for i, split in enumerate(splits)
            }

            # 处理完成的任务（tqdm 显示进度条）
            for future in tqdm(
                as_completed(future_to_node),
                total=len(splits),
                desc="提取概念和实体"
            ):
                node = future_to_node[future]
                concepts = future.result()
                # 将概念存储到对应节点
                self.graph.nodes[node]['concepts'] = concepts

    def _add_edges(self, embeddings):
        """
        基于相似度和共享概念添加边

        两个节点之间有边，当且仅当:
        1. 相似度 > threshold (0.8)
        2. 计算边权重（结合相似度和共享概念数）
        """
        similarity_matrix = self._compute_similarities(embeddings)
        num_nodes = len(self.graph.nodes)

        # 遍历所有节点对
        for node1 in tqdm(range(num_nodes), desc="添加边"):
            for node2 in range(node1 + 1, num_nodes):
                similarity_score = similarity_matrix[node1][node2]

                # 只添加相似度超过阈值的边
                if similarity_score > self.edges_threshold:
                    # 计算共享概念
                    shared_concepts = (
                        set(self.graph.nodes[node1]['concepts'])
                        & set(self.graph.nodes[node2]['concepts'])
                    )
                    # & 表示集合交集

                    # 计算边权重
                    edge_weight = self._calculate_edge_weight(
                        node1, node2, similarity_score, shared_concepts
                    )

                    # 添加边和属性
                    self.graph.add_edge(
                        node1, node2,
                        weight=edge_weight,
                        similarity=similarity_score,
                        shared_concepts=list(shared_concepts)
                    )

    def _calculate_edge_weight(self, node1, node2, similarity_score,
                                shared_concepts, alpha=0.7, beta=0.3):
        """
        计算边的权重

        权重公式:
        weight = α × 相似度 + β × 共享概念比例

        参数:
            alpha: 相似度的权重（默认 0.7）
            beta: 共享概念的权重（默认 0.3）

        为什么相似度权重更高？
        - 向量相似度已经包含了很多语义信息
        - 共享概念作为补充
        """
        # 计算可能的最大共享概念数
        max_possible_shared = min(
            len(self.graph.nodes[node1]['concepts']),
            len(self.graph.nodes[node2]['concepts'])
        )

        # 归一化共享概念数（0 到 1 之间）
        normalized_shared_concepts = (
            len(shared_concepts) / max_possible_shared
            if max_possible_shared > 0 else 0
        )

        # 加权计算
        return alpha * similarity_score + beta * normalized_shared_concepts

    def _lemmatize_concept(self, concept):
        """
        对概念进行词形还原

        为什么要词形还原？
        - "climate change" 和 "climate changes" 应该是同一个概念
        - 统一成基本形式，方便匹配

        返回:
            词形还原后的概念（小写）
        """
        return ' '.join([
            self.lemmatizer.lemmatize(word)
            for word in concept.lower().split()
        ])
```

> **💡 代码解释**
>
> **图谱构建流程：**
> ```
> 1. 原文档 → 分割 → 节点
> 2. 每个节点 → 提取概念 → 概念列表
> 3. 所有节点对 → 计算相似度 → 添加边（如果相似度够高）
> 4. 边权重 = 0.7 × 相似度 + 0.3 × 共享概念比例
> ```
>
> **⚠️ 新手注意**
> - `edges_threshold=0.8` 较高，确保只有真正相关的节点才连接
> - 阈值太高：图太稀疏，可能丢失连接
> - 阈值太低：图太密集，噪音太多
>
> **❓ 常见问题**
>
> **Q: 为什么要用 spaCy 和 LLM 两种方式提取概念？**
>
> A: 各有所长：
> - spaCy：擅长识别标准实体（人名、地名），快速准确
> - LLM：擅长理解抽象概念（"全球变暖"、"可持续发展"）
> 两者结合更全面。

---

## 🛠️ 第四步：定义 QueryEngine 类

### 📖 这是什么？

QueryEngine 是 GraphRAG 的"大脑"，负责处理用户查询。它使用一种类似 Dijkstra 算法的图遍历方式，在图谱中"漫步"，找到最相关的信息路径。

### 💻 完整代码

```python
class AnswerCheck(BaseModel):
    """用于结构化答案检查的输出"""
    is_complete: bool = Field(description="当前上下文是否提供对查询的完整答案")
    answer: str = Field(description="基于当前上下文的当前答案，如果有")

class QueryEngine:
    def __init__(self, vector_store, knowledge_graph, llm):
        """
        初始化查询引擎

        参数:
            vector_store: FAISS 向量存储
            knowledge_graph: 知识图谱
            llm: 语言模型
        """
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.llm = llm
        self.max_context_length = 4000  # 最大上下文长度
        self.answer_check_chain = self._create_answer_check_chain()

    def _create_answer_check_chain(self):
        """
        创建检查答案是否完整的链

        作用:
        - 判断当前收集的信息是否足够回答问题
        - 如果够，直接生成答案
        - 如果不够，继续遍历图谱
        """
        answer_check_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""给定查询：'{query}'

当前上下文:
{context}

此上下文是否提供对查询的完整答案？如果是，请提供答案。
如果否，请说明答案不完整。

是否完整答案（是/否）:
答案（如果完整）:"""
        )
        return answer_check_prompt | self.llm.with_structured_output(AnswerCheck)

    def _check_answer(self, query: str, context: str) -> Tuple[bool, str]:
        """
        检查当前上下文是否提供完整答案

        返回:
            tuple: (是否完整，答案)
        """
        response = self.answer_check_chain.invoke({
            "query": query,
            "context": context
        })
        return response.is_complete, response.answer

    def _expand_context(self, query: str, relevant_docs) -> Tuple[str, List[int], Dict[int, str], str]:
        """
        使用图遍历扩展上下文

        算法思路（类似 Dijkstra）:
        1. 从最相关的节点开始
        2. 使用优先队列管理遍历顺序
        3. 优先探索连接最强的节点
        4. 检查是否已找到完整答案
        5. 如果没有，继续探索邻居节点
        6. 重复直到找到答案或遍历完所有节点

        参数:
            query: 用户查询
            relevant_docs: 初始检索到的文档

        返回:
            tuple: (扩展后的上下文，遍历路径，节点内容映射，最终答案)
        """
        # 初始化变量
        expanded_context = ""
        traversal_path = []
        visited_concepts = set()
        filtered_content = {}
        final_answer = ""

        # 优先队列（最小堆）
        # 存储 (优先级，节点 ID)
        priority_queue = []

        # 距离字典：存储到每个节点的最佳"距离"
        # 距离 = 1/连接强度，越小越好
        distances = {}

        print("\n遍历知识图谱：")

        # 从相关文档的最近节点初始化优先队列
        for doc in relevant_docs:
            # 在向量存储中找到最相似的节点
            closest_nodes = self.vector_store.similarity_search_with_score(
                doc.page_content, k=1
            )
            closest_node_content, similarity_score = closest_nodes[0]

            # 在知识图谱中找到对应节点
            closest_node = next(
                n for n in self.knowledge_graph.graph.nodes
                if self.knowledge_graph.graph.nodes[n]['content'] == closest_node_content.page_content
            )

            # 初始化优先级（相似度的倒数，用于最小堆）
            priority = 1 / similarity_score
            heapq.heappush(priority_queue, (priority, closest_node))
            distances[closest_node] = priority

        step = 0
        while priority_queue:
            # 获取优先级最高的节点（距离最小）
            current_priority, current_node = heapq.heappop(priority_queue)

            # 如果已经有更好的路径，跳过
            if current_priority > distances.get(current_node, float('inf')):
                continue

            if current_node not in traversal_path:
                step += 1
                traversal_path.append(current_node)

                # 获取节点内容和概念
                node_content = self.knowledge_graph.graph.nodes[current_node]['content']
                node_concepts = self.knowledge_graph.graph.nodes[current_node]['concepts']

                # 添加到上下文
                filtered_content[current_node] = node_content
                expanded_context += "\n" + node_content if expanded_context else node_content

                # 打印调试信息
                print(f"\n步骤 {step} - 节点 {current_node}:")
                print(f"内容：{node_content[:100]}...")  # 只显示前 100 字符
                print(f"概念：{', '.join(node_concepts)}")
                print("-" * 50)

                # 检查是否有完整答案
                is_complete, answer = self._check_answer(query, expanded_context)
                if is_complete:
                    final_answer = answer
                    print("✓ 找到完整答案！")
                    break

                # 处理当前节点的概念
                node_concepts_set = set(
                    self.knowledge_graph._lemmatize_concept(c) for c in node_concepts
                )
                if not node_concepts_set.issubset(visited_concepts):
                    visited_concepts.update(node_concepts_set)

                    # 探索邻居节点
                    for neighbor in self.knowledge_graph.graph.neighbors(current_node):
                        edge_data = self.knowledge_graph.graph[current_node][neighbor]
                        edge_weight = edge_data['weight']

                        # 计算新距离
                        # 权重越大，距离越小（优先探索）
                        distance = current_priority + (1 / edge_weight)

                        # 如果找到更好的路径，更新
                        if distance < distances.get(neighbor, float('inf')):
                            distances[neighbor] = distance
                            heapq.heappush(priority_queue, (distance, neighbor))

                            # 处理邻居节点
                            if neighbor not in traversal_path:
                                step += 1
                                traversal_path.append(neighbor)

                                neighbor_content = self.knowledge_graph.graph.nodes[neighbor]['content']
                                neighbor_concepts = self.knowledge_graph.graph.nodes[neighbor]['concepts']

                                filtered_content[neighbor] = neighbor_content
                                expanded_context += "\n" + neighbor_content if expanded_context else neighbor_content

                                print(f"\n步骤 {step} - 节点 {neighbor}（{current_node} 的邻居）:")
                                print(f"内容：{neighbor_content[:100]}...")
                                print(f"概念：{', '.join(neighbor_concepts)}")
                                print("-" * 50)

                                # 再次检查答案
                                is_complete, answer = self._check_answer(query, expanded_context)
                                if is_complete:
                                    final_answer = answer
                                    break

                                # 处理邻居概念
                                neighbor_concepts_set = set(
                                    self.knowledge_graph._lemmatize_concept(c)
                                    for c in neighbor_concepts
                                )
                                if not neighbor_concepts_set.issubset(visited_concepts):
                                    visited_concepts.update(neighbor_concepts_set)

                # 如果找到答案，跳出主循环
                if final_answer:
                    break

        # 如果还没找到完整答案，用 LLM 生成
        if not final_answer:
            print("\n生成最终答案...")
            response_prompt = PromptTemplate(
                input_variables=["query", "context"],
                template="基于以下上下文，请回答查询。\n\n上下文：{context}\n\n查询：{query}\n\n答案:"
            )
            response_chain = response_prompt | self.llm
            input_data = {"query": query, "context": expanded_context}
            final_answer = response_chain.invoke(input_data)

        return expanded_context, traversal_path, filtered_content, final_answer

    def query(self, query: str) -> Tuple[str, List[int], Dict[int, str]]:
        """
        处理查询的主方法

        流程:
        1. 检索相关文档
        2. 扩展上下文（图遍历）
        3. 生成最终答案
        4. 显示 Token 使用统计

        返回:
            tuple: (最终答案，遍历路径，过滤内容)
        """
        with get_openai_callback() as cb:
            print(f"\n处理查询：{query}")

            # 检索相关文档
            relevant_docs = self._retrieve_relevant_documents(query)

            # 扩展上下文并获取答案
            expanded_context, traversal_path, filtered_content, final_answer = \
                self._expand_context(query, relevant_docs)

            if not final_answer:
                print("\n生成最终答案...")
                response_prompt = PromptTemplate(
                    input_variables=["query", "context"],
                    template="基于以下上下文，请回答查询。\n\n上下文：{context}\n\n查询：{query}\n\n答案:"
                )
                response_chain = response_prompt | self.llm
                input_data = {"query": query, "context": expanded_context}
                response = response_chain.invoke(input_data)
                final_answer = response
            else:
                print("\n遍历过程中找到完整答案。")

            print(f"\n最终答案：{final_answer}")
            print(f"\n总 Token 数：{cb.total_tokens}")
            print(f"提示 Token 数：{cb.prompt_tokens}")
            print(f"完成 Token 数：{cb.completion_tokens}")
            print(f"总成本（USD）: ${cb.total_cost}")

        return final_answer, traversal_path, filtered_content

    def _retrieve_relevant_documents(self, query: str):
        """
        检索相关文档

        使用:
        1. 基础检索器（向量相似度）
        2. 上下文压缩（LLM 提取相关信息）

        返回:
            相关文档列表
        """
        print("\n检索相关文档...")

        # 创建基础检索器
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # 创建压缩器（用 LLM 提取相关信息）
        compressor = LLMChainExtractor.from_llm(self.llm)

        # 组合成压缩检索器
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever
        )

        return compression_retriever.invoke(query)
```

> **💡 代码解释**
>
> **Dijkstra 算法通俗理解：**
>
> 想象你在一个迷宫里找宝藏：
> 1. 从入口开始（最相关的节点）
> 2. 每次都走最近/最好的路（优先队列）
> 3. 记录到每个路口的最佳距离（distances 字典）
> 4. 找到宝藏就停止（答案完整）
>
> **⚠️ 新手注意**
> - `heapq` 是最小堆，值越小优先级越高
> - 所以用 `1/similarity` 作为优先级（相似度越高，值越小）
> - `float('inf')` 表示无穷大
>
> **❓ 常见问题**
>
> **Q: 为什么需要优先队列？**
>
> A: 图可能很大，逐个遍历太慢。优先队列确保：
> - 先探索最相关的路径
> - 尽快找到答案，可能不需要遍历全图
> - 节省时间和 API 调用成本

---

## 🛠️ 第五步：定义 Visualizer 类

### 📖 这是什么？

Visualizer 负责把知识图谱和检索路径画出来，让你"看见"AI 是如何思考的。

### 💻 完整代码

```python
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Visualizer:
    @staticmethod
    def visualize_traversal(graph, traversal_path):
        """
        可视化知识图谱遍历

        参数:
            graph: 知识图谱
            traversal_path: 遍历路径（节点索引列表）

        可视化元素:
        - 蓝色边：常规连接（颜色深浅表示权重）
        - 红色虚线箭头：遍历路径
        - 绿色节点：起始节点
        - 红色节点：结束节点
        """
        # 创建遍历图
        traversal_graph = nx.DiGraph()

        # 从原始图复制节点和边
        for node in graph.nodes():
            traversal_graph.add_node(node)
        for u, v, data in graph.edges(data=True):
            traversal_graph.add_edge(u, v, **data)

        fig, ax = plt.subplots(figsize=(16, 12))

        # 使用 spring_layout 生成节点位置
        # 原理：把图想象成弹簧网络，节点会自然分布
        pos = nx.spring_layout(traversal_graph, k=1, iterations=50)

        # 绘制常规边（蓝色，权重决定颜色深浅）
        edges = traversal_graph.edges()
        edge_weights = [
            traversal_graph[u][v].get('weight', 0.5)
            for u, v in edges
        ]
        nx.draw_networkx_edges(
            traversal_graph, pos,
            edgelist=edges,
            edge_color=edge_weights,
            edge_cmap=plt.cm.Blues,  # 蓝色渐变
            width=2,
            ax=ax
        )

        # 绘制节点（浅蓝色）
        nx.draw_networkx_nodes(
            traversal_graph, pos,
            node_color='lightblue',
            node_size=3000,
            ax=ax
        )

        # 用弯曲箭头绘制遍历路径（红色虚线）
        edge_offset = 0.1
        for i in range(len(traversal_path) - 1):
            start = traversal_path[i]
            end = traversal_path[i + 1]
            start_pos = pos[start]
            end_pos = pos[end]

            # 计算曲线控制点
            mid_point = (
                (start_pos[0] + end_pos[0]) / 2,
                (start_pos[1] + end_pos[1]) / 2
            )
            control_point = (mid_point[0] + edge_offset, mid_point[1] + edge_offset)

            # 绘制弯曲箭头
            arrow = patches.FancyArrowPatch(
                start_pos, end_pos,
                connectionstyle=f"arc3,rad={0.3}",  # 弧形，半径 0.3
                color='red',
                arrowstyle="->",
                mutation_scale=20,
                linestyle='--',  # 虚线
                linewidth=2,
                zorder=4  # 画在最上层
            )
            ax.add_patch(arrow)

        # 准备节点标签（显示主要概念）
        labels = {}
        for i, node in enumerate(traversal_path):
            concepts = graph.nodes[node].get('concepts', [])
            label = f"{i + 1}. {concepts[0] if concepts else ''}"
            labels[node] = label

        # 为其他节点也添加标签
        for node in traversal_graph.nodes():
            if node not in labels:
                concepts = graph.nodes[node].get('concepts', [])
                labels[node] = concepts[0] if concepts else ''

        # 绘制标签
        nx.draw_networkx_labels(
            traversal_graph, pos,
            labels, font_size=8, font_weight="bold", ax=ax
        )

        # 突出显示起始和结束节点
        start_node = traversal_path[0]
        end_node = traversal_path[-1]

        nx.draw_networkx_nodes(
            traversal_graph, pos,
            nodelist=[start_node],
            node_color='lightgreen',  # 起点绿色
            node_size=3000,
            ax=ax
        )

        nx.draw_networkx_nodes(
            traversal_graph, pos,
            nodelist=[end_node],
            node_color='lightcoral',  # 终点红色
            node_size=3000,
            ax=ax
        )

        ax.set_title("图遍历流程")
        ax.axis('off')  # 隐藏坐标轴

        # 添加颜色条（表示边权重）
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.Blues,
            norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical',
                           fraction=0.046, pad=0.04)
        cbar.set_label('边权重', rotation=270, labelpad=15)

        # 添加图例
        regular_line = plt.Line2D(
            [0], [0], color='blue', linewidth=2, label='常规边'
        )
        traversal_line = plt.Line2D(
            [0], [0], color='red', linewidth=2, linestyle='--',
            label='遍历路径'
        )
        start_point = plt.Line2D(
            [0], [0], marker='o', color='w', markerfacecolor='lightgreen',
            markersize=15, label='起始节点'
        )
        end_point = plt.Line2D(
            [0], [0], marker='o', color='w', markerfacecolor='lightcoral',
            markersize=15, label='结束节点'
        )
        legend = plt.legend(
            handles=[regular_line, traversal_line, start_point, end_point],
            loc='upper left',
            bbox_to_anchor=(0, 1),
            ncol=2
        )
        legend.get_frame().set_alpha(0.8)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def print_filtered_content(traversal_path, filtered_content):
        """
        按遍历顺序打印节点内容

        作用:
        - 查看 AI 访问了哪些信息
        - 理解推理过程
        """
        print("\n按遍历顺序访问节点的过滤内容:")
        for i, node in enumerate(traversal_path):
            print(f"\n步骤 {i + 1} - 节点 {node}:")
            content = filtered_content.get(node, 'No filtered content available')
            print(f"过滤内容：{content[:200]}...")  # 只显示前 200 字符
            print("-" * 50)
```

> **💡 代码解释**
>
> **可视化元素说明：**
> - **节点大小**：统一 3000，便于阅读
> - **边的颜色**：蓝色越深表示关系越强
> - **红色虚线**：AI 实际的思考路径
> - **绿色/红色节点**：起点和终点
>
> **⚠️ 新手注意**
> - 如果节点太多，图会很乱，可以只画遍历路径上的节点
> - `spring_layout` 每次运行结果不同（随机性）
> - 设置随机种子可以得到可重复的结果

---

## 🛠️ 第六步：定义 GraphRAG 主类

### 📖 这是什么？

GraphRAG 类把所有组件整合在一起，提供简洁的接口。用户只需要调用 `process_documents` 和 `query` 两个方法。

### 💻 完整代码

```python
class GraphRAG:
    def __init__(self):
        """
        初始化 GraphRAG 系统

        组件:
        - llm: 语言模型（GPT-4o-mini）
        - embedding_model: 嵌入模型
        - document_processor: 文档处理器
        - knowledge_graph: 知识图谱
        - query_engine: 查询引擎（处理文档后创建）
        - visualizer: 可视化器
        """
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-4o-mini",
            max_tokens=4000
        )
        self.embedding_model = OpenAIEmbeddings()
        self.document_processor = DocumentProcessor()
        self.knowledge_graph = KnowledgeGraph()
        self.query_engine = None  # 处理文档后创建
        self.visualizer = Visualizer()

    def process_documents(self, documents):
        """
        处理文档并构建知识图谱

        流程:
        1. 分割文档并创建向量存储
        2. 构建知识图谱（节点、概念、边）
        3. 创建查询引擎

        参数:
            documents: 文档列表
        """
        print("正在处理文档...")
        splits, vector_store = self.document_processor.process_documents(documents)

        print("正在构建知识图谱...")
        self.knowledge_graph.build_graph(splits, self.llm, self.embedding_model)

        # 创建查询引擎
        self.query_engine = QueryEngine(vector_store, self.knowledge_graph, self.llm)
        print("✓ 文档处理完成！")

    def query(self, query: str):
        """
        处理查询并返回答案

        流程:
        1. 使用查询引擎检索和推理
        2. 可视化遍历路径
        3. 返回答案

        参数:
            query: 用户查询

        返回:
            答案字符串
        """
        print(f"\n处理查询：{query}")
        response, traversal_path, filtered_content = \
            self.query_engine.query(query)

        # 如果有遍历路径，可视化展示
        if traversal_path:
            self.visualizer.visualize_traversal(
                self.knowledge_graph.graph, traversal_path
            )
        else:
            print("没有遍历路径可展示。")

        return response
```

> **💡 代码解释**
>
> **设计模式：外观模式（Facade Pattern）**
> - GraphRAG 类隐藏了内部复杂性
> - 用户只需要知道两个方法
> - 内部有多个组件协同工作
>
> **⚠️ 新手注意**
> - `query_engine` 初始为 None，必须调用 `process_documents` 后才能查询
> - 处理文档是一次性的，可以重复查询

---

## 🛠️ 第七步：完整使用示例

### 📖 这是什么？

现在我们用真实文档来测试整个系统。

### 💻 完整代码

```python
from langchain.document_loaders import PyPDFLoader

# 下载示例数据（如果还没有）
import os
os.makedirs('data', exist_ok=True)

# 下载气候变化的 PDF 文档
!wget -O data/Understanding_Climate_Change.pdf \
    https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf

# 指定文件路径
path = "data/Understanding_Climate_Change.pdf"

# 加载 PDF 文档
loader = PyPDFLoader(path)
documents = loader.load()

# 只处理前 10 页（节省时间和成本）
documents = documents[:10]
print(f"加载了 {len(documents)} 页文档")

# 创建 GraphRAG 实例
graph_rag = GraphRAG()

# 处理文档（这可能需要几分钟）
graph_rag.process_documents(documents)

# 提问
query = "what is the main cause of climate change?"
print(f"\n提问：{query}")

# 获取答案
response = graph_rag.query(query)

print("\n" + "="*50)
print("最终答案:")
print("="*50)
print(response)
```

> **💡 代码解释**
>
> **PDF 加载器：**
> - `PyPDFLoader` 是 LangChain 的文档加载器
> - 自动处理 PDF 格式，提取文本
> - 每页作为一个 Document 对象
>
> **⚠️ 新手注意**
> - 文档处理可能需要 5-15 分钟（取决于文档数量）
> - 概念提取会调用多次 LLM，产生 API 费用
> - 建议先用少量文档测试
>
> **📊 预期输出示例：**
>
> ```
> 正在处理文档...
> 正在构建知识图谱...
> 提取概念和实体：100%|██████████| 25/25 [00:45<00:00]
> 添加边：100%|██████████| 300/300 [00:10<00:00]
> ✓ 文档处理完成！
>
> 处理查询：what is the main cause of climate change?
>
> 检索相关文档...
>
> 遍历知识图谱：
>
> 步骤 1 - 节点 3:
> 内容：The primary driver of climate change is the increase in greenhouse gases...
> 概念：greenhouse gases, climate change, emissions
> --------------------------------------------------
> ✓ 找到完整答案！
>
> 最终答案：The main cause of climate change is the increase in greenhouse gases...
> ```

---

## 📚 GraphRAG 的优势总结

| 优势 | 说明 | 实际价值 |
|------|------|---------|
| **改进上下文感知** | 图谱结构保留概念间关系 | 理解"温室效应"导致"全球变暖" |
| **增强检索** | 智能遍历，超越关键词匹配 | 找到语义相关但用词不同的内容 |
| **可解释的结果** | 可视化展示推理路径 | 理解 AI 如何得出答案 |
| **灵活的知识表示** | 容易添加新信息和关系 | 增量更新，无需重建 |
| **高效遍历** | 优先探索最强连接 | 快速找到核心信息 |

---

## 🎓 术语解释表

| 术语 | 英文 | 解释 |
|------|------|------|
| 知识图谱 | Knowledge Graph | 用图结构表示知识的系统 |
| 节点 | Node | 图中的点，代表概念或实体 |
| 边 | Edge | 图中的线，代表关系 |
| 词形还原 | Lemmatization | 把单词变成基本形式 |
| 命名实体识别 | NER | 识别人名、地名等 |
| 余弦相似度 | Cosine Similarity | 衡量向量相似程度 |
| 优先队列 | Priority Queue | 按优先级排序的队列 |
| 图遍历 | Graph Traversal | 沿着图的边访问节点 |

---

## ❓ 常见问题 FAQ

### Q1: GraphRAG 和普通 RAG 有什么区别？

**A:** 主要区别在于信息组织方式：

| 特性 | 普通 RAG | GraphRAG |
|------|---------|---------|
| 组织方式 | 扁平的文档块 | 图结构（节点 + 边） |
| 检索方式 | 向量相似度 | 图遍历 + 相似度 |
| 关系理解 | 无 | 理解概念间关系 |
| 可解释性 | 低 | 高（可可视化路径） |

### Q2: 处理大量文档需要多长时间？

**A:** 取决于：
- 文档数量和长度
- 概念提取的复杂度
- 网络速度（API 调用）

经验法则：每页文档约需 1-2 分钟处理时间。

### Q3: 费用高吗？

**A:** 主要成本来自：
1. 概念提取（每个文档块调用一次 LLM）
2. 嵌入生成（批量处理，相对便宜）
3. 查询时的 LLM 调用

建议：先用小样本测试，估算成本。

### Q4: 可以用中文文档吗？

**A:** 可以，但需要：
- 修改 spaCy 模型为中文模型
- 可能需要调整概念提取的提示词
- 中文嵌入模型（OpenAI 支持多语言）

---

## ✅ 学习检查清单

- [ ] 我理解了知识图谱的基本概念
- [ ] 我知道节点和边的含义
- [ ] 我理解了图遍历的工作原理
- [ ] 我能解释优先队列的作用
- [ ] 我知道如何创建 GraphRAG 实例
- [ ] 我理解了可视化的意义

---

## 🚀 下一步学习建议

1. **尝试自己的文档**：用 PDF、TXT 等格式的文档测试
2. **调整参数**：尝试改变 `edges_threshold`，观察图的变化
3. **学习微软 GraphRAG**：继续学习下一个教程，了解工业级实现
4. **可视化优化**：尝试不同的可视化样式

---

> **💪 恭喜！** 你已经完成了 GraphRAG 的新手教程！现在你理解了如何用知识图谱增强检索系统，这是构建更智能 RAG 应用的关键一步！
