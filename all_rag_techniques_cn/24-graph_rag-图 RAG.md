# GraphRAG：图增强检索增强生成

## 概述

GraphRAG 是一个先进的问答系统，结合了基于图的知识表示与检索增强生成的力量。它处理输入文档以创建丰富的知识图谱，然后用于增强用户查询答案的检索和生成。该系统利用自然语言处理、机器学习和图论来提供更准确和上下文相关的响应。

## 动机

传统的检索增强生成系统经常在长文档上维护上下文以及在相关信息片段之间建立连接方面存在困难。GraphRAG 通过以下方式解决这些限制：

1. 将知识表示为相互连接的图，更好地保留概念之间的关系。
2. 在查询过程中实现更智能的信息遍历。
3. 提供信息在回答过程中如何连接和访问的可视化表示。

## 关键组件

1. **DocumentProcessor**：处理输入文档的初始处理，创建文本块和嵌入。

2. **KnowledgeGraph**：构建处理文档的图表示，其中节点表示文本块，边表示它们之间的关系。

3. **QueryEngine**：通过利用知识图谱和向量存储来管理回答用户查询的过程。

4. **Visualizer**：创建图的可视化表示以及为回答查询而采用的遍历路径。

## 方法详情

1. **文档处理**：
   - 输入文档被分割成可管理的块。
   - 每个块使用语言模型进行嵌入。
   - 从这些嵌入创建向量存储以进行高效的相似度搜索。

2. **知识图谱构建**：
   - 为每个文本块创建图节点。
   - 使用 NLP 技术和语言模型的组合从每个块中提取概念。
   - 提取的概念被词形还原以改进匹配。
   - 基于语义相似度和共享概念在节点之间添加边。
   - 计算边权重以表示关系的强度。

3. **查询处理**：
   - 用户查询被嵌入并用于从向量存储中检索相关文档。
   - 用对应于最相关文档的节点初始化优先队列。
   - 系统采用类似 Dijkstra 的算法遍历知识图谱：
     * 节点按其优先级（与查询的连接强度）顺序进行探索。
     * 对于每个探索的节点：
       - 其内容被添加到上下文中。
       - 系统检查当前上下文是否提供完整答案。
       - 如果答案不完整：
         * 节点的概念被处理并添加到已访问概念集。
         * 探索相邻节点，根据边权重更新它们的优先级。
         * 如果发现更强的连接，节点被添加到优先队列。
   - 此过程持续到找到完整答案或优先队列耗尽。
   - 如果遍历图后未找到完整答案，系统使用累积的上下文和大型语言模型生成最终答案。

4. **可视化**：
   - 知识图谱被可视化，节点表示文本块，边表示关系。
   - 边颜色表示关系的强度（权重）。
   - 为回答查询而采用的遍历路径用弯曲的虚线箭头突出显示。
   - 遍历的起始和结束节点用不同的颜色突出显示以便于识别。

## 这种方法的好处

1. **改进上下文感知**：通过将知识表示为图，系统可以更好地维护上下文并在输入文档的不同部分之间建立连接。

2. **增强检索**：图结构允许更智能的信息检索，超越简单的关键词匹配。

3. **可解释的结果**：图和遍历路径的可视化提供了系统如何得出答案的洞察，提高了透明度和信任度。

4. **灵活的知识表示**：图结构可以轻松合并新信息和关系。

5. **高效的信息遍历**：图中的加权边允许系统在回答查询时优先考虑最相关的信息路径。

## 结论

GraphRAG 代表了检索增强生成系统的重大进步。通过结合基于图的知识表示和智能遍历机制，它提供了改进的上下文感知、更准确的检索和增强的可解释性。系统可视化其决策过程的能力为其操作提供了宝贵的见解，使其成为最终用户和开发人员的强大工具。随着自然语言处理和基于图的 AI 不断发展，像 GraphRAG 这样的系统为更复杂和强大的问答技术铺平了道路。

<div style="text-align: center;">

<img src="../images/graph_rag.svg" alt="graph RAG" style="width:100%; height:auto;">
</div>

# 包安装和导入

下面的单元格安装运行此笔记本所需的所有包。


```python
# 安装所需的包
!pip install faiss-cpu futures langchain langchain-openai matplotlib networkx nltk numpy python-dotenv scikit-learn spacy tqdm
```

```python
# 克隆仓库以访问辅助函数和评估模块
!git clone https://github.com/NirDiamant/RAG_TECHNIQUES.git
import sys
sys.path.append('RAG_TECHNIQUES')
# 如果需要使用最新数据运行
# !cp -r RAG_TECHNIQUES/data .
```

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


# 原始路径追加已替换为 Colab 兼容性
from helper_functions import *
from evaluation.evalute_rag import *

# 从.env 文件加载环境变量
load_dotenv()

# 设置 OpenAI API 密钥环境变量
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
```

### 定义文档处理器类

```python
# 定义 DocumentProcessor 类
class DocumentProcessor:
    def __init__(self):
        """
        使用文本分割器和 OpenAI embeddings 初始化 DocumentProcessor。

        属性：
        - text_splitter: RecursiveCharacterTextSplitter 的实例，具有指定的块大小和重叠。
        - embeddings: OpenAIEmbeddings 的实例，用于嵌入文档。
        """
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embeddings = OpenAIEmbeddings()

    def process_documents(self, documents):
        """
        处理文档列表，将其分割成较小的块并创建向量存储。

        Args:
        - documents (list of str): 要处理的文档列表。

        Returns:
        - tuple: 包含以下内容的元组：
          - splits (list of str): 分割后的文档块列表。
          - vector_store (FAISS): 从分割后的文档块及其嵌入创建的 FAISS 向量存储。
        """
        splits = self.text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(splits, self.embeddings)
        return splits, vector_store

    def create_embeddings_batch(self, texts, batch_size=32):
        """
        批量为文本列表创建嵌入。

        Args:
        - texts (list of str): 要嵌入的文本列表。
        - batch_size (int, optional): 每批处理的文本数量。默认值为 32。

        Returns:
        - numpy.ndarray: 输入文本的嵌入数组。
        """
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.embeddings.embed_documents(batch)
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)

    def compute_similarity_matrix(self, embeddings):
        """
        计算嵌入的余弦相似度矩阵。

        Args:
        - embeddings (numpy.ndarray): 嵌入数组。

        Returns:
        - numpy.ndarray: 输入嵌入的余弦相似度矩阵。
        """
        return cosine_similarity(embeddings)

```

### 定义知识图谱类

```python
# 定义 Concepts 类
class Concepts(BaseModel):
    concepts_list: List[str] = Field(description="概念列表")

# 定义 KnowledgeGraph 类
class KnowledgeGraph:
    def __init__(self):
        """
        使用图、词形还原器和 NLP 模型初始化 KnowledgeGraph。

        属性：
        - graph: networkx Graph 的实例。
        - lemmatizer: WordNetLemmatizer 的实例。
        - concept_cache: 缓存提取概念的字典。
        - nlp: spaCy NLP 模型的实例。
        - edges_threshold: 设置基于相似度添加边的阈值的浮点值。
        """
        self.graph = nx.Graph()
        self.lemmatizer = WordNetLemmatizer()
        self.concept_cache = {}
        self.nlp = self._load_spacy_model()
        self.edges_threshold = 0.8

    def build_graph(self, splits, llm, embedding_model):
        """
        通过添加节点、创建嵌入、提取概念和添加边来构建知识图谱。

        Args:
        - splits (list): 文档分割列表。
        - llm: 大型语言模型的实例。
        - embedding_model: 嵌入模型的实例。

        Returns:
        - None
        """
        self._add_nodes(splits)
        embeddings = self._create_embeddings(splits, embedding_model)
        self._extract_concepts(splits, llm)
        self._add_edges(embeddings)

    def _add_nodes(self, splits):
        """
        从文档分割向图添加节点。

        Args:
        - splits (list): 文档分割列表。

        Returns:
        - None
        """
        for i, split in enumerate(splits):
            self.graph.add_node(i, content=split.page_content)

    def _create_embeddings(self, splits, embedding_model):
        """
        使用嵌入模型为文档分割创建嵌入。

        Args:
        - splits (list): 文档分割列表。
        - embedding_model: 嵌入模型的实例。

        Returns:
        - numpy.ndarray: 文档分割的嵌入数组。
        """
        texts = [split.page_content for split in splits]
        return embedding_model.embed_documents(texts)

    def _compute_similarities(self, embeddings):
        """
        计算嵌入的余弦相似度矩阵。

        Args:
        - embeddings (numpy.ndarray): 嵌入数组。

        Returns:
        - numpy.ndarray: 嵌入的余弦相似度矩阵。
        """
        return cosine_similarity(embeddings)

    def _load_spacy_model(self):
        """
        加载 spaCy NLP 模型，必要时下载。

        Args:
        - None

        Returns:
        - spacy.Language: spaCy NLP 模型的实例。
        """
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model...")
            download("en_core_web_sm")
            return spacy.load("en_core_web_sm")

    def _extract_concepts_and_entities(self, content, llm):
        """
        使用 spaCy 和大型语言模型从内容中提取概念和命名实体。

        Args:
        - content (str): 要提取概念和实体的内容。
        - llm: 大型语言模型的实例。

        Returns:
        - list: 提取的概念和实体列表。
        """
        if content in self.concept_cache:
            return self.concept_cache[content]

        # 使用 spaCy 提取命名实体
        doc = self.nlp(content)
        named_entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "WORK_OF_ART"]]

        # 使用 LLM 提取一般概念
        concept_extraction_prompt = PromptTemplate(
            input_variables=["text"],
            template="从以下文本中提取关键概念（不包括命名实体）：\n\n{text}\n\n关键概念："
        )
        concept_chain = concept_extraction_prompt | llm.with_structured_output(Concepts)
        general_concepts = concept_chain.invoke({"text": content}).concepts_list

        # 组合命名实体和一般概念
        all_concepts = list(set(named_entities + general_concepts))

        self.concept_cache[content] = all_concepts
        return all_concepts

    def _extract_concepts(self, splits, llm):
        """
        使用多线程为所有文档分割提取概念。

        Args:
        - splits (list): 文档分割列表。
        - llm: 大型语言模型的实例。

        Returns:
        - None
        """
        with ThreadPoolExecutor() as executor:
            future_to_node = {executor.submit(self._extract_concepts_and_entities, split.page_content, llm): i
                              for i, split in enumerate(splits)}

            for future in tqdm(as_completed(future_to_node), total=len(splits), desc="提取概念和实体"):
                node = future_to_node[future]
                concepts = future.result()
                self.graph.nodes[node]['concepts'] = concepts

    def _add_edges(self, embeddings):
        """
        基于嵌入的相似性和共享概念向图添加边。

        Args:
        - embeddings (numpy.ndarray): 文档分割的嵌入数组。

        Returns:
        - None
        """
        similarity_matrix = self._compute_similarities(embeddings)
        num_nodes = len(self.graph.nodes)

        for node1 in tqdm(range(num_nodes), desc="添加边"):
            for node2 in range(node1 + 1, num_nodes):
                similarity_score = similarity_matrix[node1][node2]
                if similarity_score > self.edges_threshold:
                    shared_concepts = set(self.graph.nodes[node1]['concepts']) & set(self.graph.nodes[node2]['concepts'])
                    edge_weight = self._calculate_edge_weight(node1, node2, similarity_score, shared_concepts)
                    self.graph.add_edge(node1, node2, weight=edge_weight,
                                        similarity=similarity_score,
                                        shared_concepts=list(shared_concepts))

    def _calculate_edge_weight(self, node1, node2, similarity_score, shared_concepts, alpha=0.7, beta=0.3):
        """
        基于相似性分数和共享概念计算边的权重。

        Args:
        - node1 (int): 第一个节点。
        - node2 (int): 第二个节点。
        - similarity_score (float): 节点之间的相似性分数。
        - shared_concepts (set): 节点之间的共享概念集。
        - alpha (float, optional): 相似性分数的权重。默认值为 0.7。
        - beta (float, optional): 共享概念的权重。默认值为 0.3。

        Returns:
        - float: 计算出的边权重。
        """
        max_possible_shared = min(len(self.graph.nodes[node1]['concepts']), len(self.graph.nodes[node2]['concepts']))
        normalized_shared_concepts = len(shared_concepts) / max_possible_shared if max_possible_shared > 0 else 0
        return alpha * similarity_score + beta * normalized_shared_concepts

    def _lemmatize_concept(self, concept):
        """
        对给定概念进行词形还原。

        Args:
        - concept (str): 要进行词形还原的概念。

        Returns:
        - str: 词形还原后的概念。
        """
        return ' '.join([self.lemmatizer.lemmatize(word) for word in concept.lower().split()])

```

### 定义查询引擎类

```python
# 下载所需的数据文件
import os
os.makedirs('data', exist_ok=True)

# 下载此笔记本中使用的 PDF 文档
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf

```

```python

# 定义 AnswerCheck 类
class AnswerCheck(BaseModel):
    is_complete: bool = Field(description="当前上下文是否提供对查询的完整答案")
    answer: str = Field(description="基于上下文的当前答案，如果有")

# 定义 QueryEngine 类
class QueryEngine:
    def __init__(self, vector_store, knowledge_graph, llm):
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.llm = llm
        self.max_context_length = 4000
        self.answer_check_chain = self._create_answer_check_chain()

    def _create_answer_check_chain(self):
        """
        创建一个链来检查上下文是否提供对查询的完整答案。

        Args:
        - None

        Returns:
        - Chain: 用于检查上下文是否提供完整答案的链。
        """
        answer_check_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="给定查询：'{query}'\n\n和当前上下文:\n{context}\n\n此上下文是否提供对查询的完整答案？如果是，请提供答案。如果否，请说明答案不完整。\n\n是否完整答案（是/否）:\n答案（如果完整）:"
        )
        return answer_check_prompt | self.llm.with_structured_output(AnswerCheck)

    def _check_answer(self, query: str, context: str) -> Tuple[bool, str]:
        """
        检查当前上下文是否提供对查询的完整答案。

        Args:
        - query (str): 要回答的查询。
        - context (str): 当前上下文。

        Returns:
        - tuple: 包含以下内容的元组：
          - is_complete (bool): 上下文是否提供完整答案。
          - answer (str): 基于上下文的答案，如果完整。
        """
        response = self.answer_check_chain.invoke({"query": query, "context": context})
        return response.is_complete, response.answer



    def _expand_context(self, query: str, relevant_docs) -> Tuple[str, List[int], Dict[int, str], str]:
        """
        使用类似 Dijkstra 的方法遍历知识图谱来扩展上下文。

        此方法实现了 Dijkstra 算法的修改版本来探索知识图谱，
        优先考虑最相关和强连接的信息。算法工作原理如下：

        1. 初始化：
           - 从对应于最相关文档的节点开始。
           - 使用优先队列管理遍历顺序，其中优先级基于连接强度。
           - 维护到每个节点的最佳已知"距离"（连接强度的倒数）字典。

        2. 遍历：
           - 总是优先探索具有最高优先级（最强连接）的节点。
           - 对于每个节点，检查是否找到完整答案。
           - 探索节点的邻居，如果发现更强的连接则更新它们的优先级。

        3. 概念处理：
           - 跟踪已访问的概念以引导探索朝向新的、相关的信息。
           - 仅当邻居引入新概念时才扩展到邻居。

        4. 终止：
           - 如果找到完整答案则停止。
           - 继续直到优先队列为空（所有可达节点已探索）。

        这种方法确保：
        - 我们优先考虑最相关和强连接的信息。
        - 我们系统地探索新概念。
        - 我们通过遵循知识图谱中最强的连接找到最相关的答案。

        Args:
        - query (str): 要回答的查询。
        - relevant_docs (List[Document]): 开始遍历的相关文档列表。

        Returns:
        - tuple: 包含以下内容的元组：
          - expanded_context (str): 从遍历节点累积的上下文。
          - traversal_path (List[int]): 访问的节点索引序列。
          - filtered_content (Dict[int, str]): 节点索引到其内容的映射。
          - final_answer (str): 找到的最终答案，如果有。
        """
        # 初始化变量
        expanded_context = ""
        traversal_path = []
        visited_concepts = set()
        filtered_content = {}
        final_answer = ""

        priority_queue = []
        distances = {}  # 存储到每个节点的最佳已知"距离"（连接强度的倒数）

        print("\n遍历知识图谱：")

        # 从相关文档的最近节点初始化优先队列
        for doc in relevant_docs:
            # 为每个相关文档在知识图谱中找到最相似的节点
            closest_nodes = self.vector_store.similarity_search_with_score(doc.page_content, k=1)
            closest_node_content, similarity_score = closest_nodes[0]

            # 在我们的知识图谱中获取对应的节点
            closest_node = next(n for n in self.knowledge_graph.graph.nodes if self.knowledge_graph.graph.nodes[n]['content'] == closest_node_content.page_content)

            # 初始化优先级（相似性分数的倒数用于最小堆行为）
            priority = 1 / similarity_score
            heapq.heappush(priority_queue, (priority, closest_node))
            distances[closest_node] = priority

        step = 0
        while priority_queue:
            # 获取具有最高优先级（最低距离值）的节点
            current_priority, current_node = heapq.heappop(priority_queue)

            # 如果已经找到到此节点的更好路径，则跳过
            if current_priority > distances.get(current_node, float('inf')):
                continue

            if current_node not in traversal_path:
                step += 1
                traversal_path.append(current_node)
                node_content = self.knowledge_graph.graph.nodes[current_node]['content']
                node_concepts = self.knowledge_graph.graph.nodes[current_node]['concepts']

                # 将节点内容添加到累积的上下文
                filtered_content[current_node] = node_content
                expanded_context += "\n" + node_content if expanded_context else node_content

                # 记录当前步骤用于调试和可视化
                print(f"\n步骤 {step} - 节点 {current_node}:")
                print(f"内容：{node_content[:100]}...")
                print(f"概念：{', '.join(node_concepts)}")
                print("-" * 50)

                # 检查当前上下文是否有完整答案
                is_complete, answer = self._check_answer(query, expanded_context)
                if is_complete:
                    final_answer = answer
                    break

                # 处理当前节点的概念
                node_concepts_set = set(self.knowledge_graph._lemmatize_concept(c) for c in node_concepts)
                if not node_concepts_set.issubset(visited_concepts):
                    visited_concepts.update(node_concepts_set)

                    # 探索邻居
                    for neighbor in self.knowledge_graph.graph.neighbors(current_node):
                        edge_data = self.knowledge_graph.graph[current_node][neighbor]
                        edge_weight = edge_data['weight']

                        # 计算到邻居的新距离（优先级）
                        # 注意：我们使用 1 / edge_weight 因为更高的权重意味着更强的连接
                        distance = current_priority + (1 / edge_weight)

                        # 如果找到到邻居的更强连接，更新其距离
                        if distance < distances.get(neighbor, float('inf')):
                            distances[neighbor] = distance
                            heapq.heappush(priority_queue, (distance, neighbor))

                            # 处理邻居节点，如果它还不在遍历路径中
                            if neighbor not in traversal_path:
                                step += 1
                                traversal_path.append(neighbor)
                                neighbor_content = self.knowledge_graph.graph.nodes[neighbor]['content']
                                neighbor_concepts = self.knowledge_graph.graph.nodes[neighbor]['concepts']

                                filtered_content[neighbor] = neighbor_content
                                expanded_context += "\n" + neighbor_content if expanded_context else neighbor_content

                                # 记录邻居节点信息
                                print(f"\n步骤 {step} - 节点 {neighbor}（{current_node} 的邻居）:")
                                print(f"内容：{neighbor_content[:100]}...")
                                print(f"概念：{', '.join(neighbor_concepts)}")
                                print("-" * 50)

                                # 检查添加邻居内容后是否有完整答案
                                is_complete, answer = self._check_answer(query, expanded_context)
                                if is_complete:
                                    final_answer = answer
                                    break

                                # 处理邻居的概念
                                neighbor_concepts_set = set(self.knowledge_graph._lemmatize_concept(c) for c in neighbor_concepts)
                                if not neighbor_concepts_set.issubset(visited_concepts):
                                    visited_concepts.update(neighbor_concepts_set)

                # 如果找到最终答案，跳出主循环
                if final_answer:
                    break

        # 如果还没找到完整答案，使用 LLM 生成一个
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
        通过检索相关文档、扩展上下文和生成最终答案来处理查询。

        Args:
        - query (str): 要回答的查询。

        Returns:
        - tuple: 包含以下内容的元组：
          - final_answer (str): 查询的最终答案。
          - traversal_path (list): 知识图谱中节点的遍历路径。
          - filtered_content (dict): 节点的过滤内容。
        """
        with get_openai_callback() as cb:
            print(f"\n处理查询：{query}")
            relevant_docs = self._retrieve_relevant_documents(query)
            expanded_context, traversal_path, filtered_content, final_answer = self._expand_context(query, relevant_docs)

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
        使用向量存储基于查询检索相关文档。

        Args:
        - query (str): 要回答的查询。

        Returns:
        - list: 相关文档列表。
        """
        print("\n检索相关文档...")
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        compressor = LLMChainExtractor.from_llm(self.llm)
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
        return compression_retriever.invoke(query)

```

### 定义可视化类

```python
# 导入必要的库
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 定义 Visualizer 类
class Visualizer:
    @staticmethod
    def visualize_traversal(graph, traversal_path):
        """
        在知识图谱上可视化遍历路径，突出显示节点、边和遍历路径。

        Args:
        - graph (networkx.Graph): 包含节点和边的知识图谱。
        - traversal_path (list of int): 表示遍历路径的节点索引列表。

        Returns:
        - None
        """
        traversal_graph = nx.DiGraph()

        # 从原始图添加节点和边
        for node in graph.nodes():
            traversal_graph.add_node(node)
        for u, v, data in graph.edges(data=True):
            traversal_graph.add_edge(u, v, **data)

        fig, ax = plt.subplots(figsize=(16, 12))

        # 为所有节点生成位置
        pos = nx.spring_layout(traversal_graph, k=1, iterations=50)

        # 根据权重绘制常规边
        edges = traversal_graph.edges()
        edge_weights = [traversal_graph[u][v].get('weight', 0.5) for u, v in edges]
        nx.draw_networkx_edges(traversal_graph, pos,
                               edgelist=edges,
                               edge_color=edge_weights,
                               edge_cmap=plt.cm.Blues,
                               width=2,
                               ax=ax)

        # 绘制节点
        nx.draw_networkx_nodes(traversal_graph, pos,
                               node_color='lightblue',
                               node_size=3000,
                               ax=ax)

        # 用弯曲箭头绘制遍历路径
        edge_offset = 0.1
        for i in range(len(traversal_path) - 1):
            start = traversal_path[i]
            end = traversal_path[i + 1]
            start_pos = pos[start]
            end_pos = pos[end]

            # 计算曲线的控制点
            mid_point = ((start_pos[0] + end_pos[0]) / 2, (start_pos[1] + end_pos[1]) / 2)
            control_point = (mid_point[0] + edge_offset, mid_point[1] + edge_offset)

            # 绘制弯曲箭头
            arrow = patches.FancyArrowPatch(start_pos, end_pos,
                                            connectionstyle=f"arc3,rad={0.3}",
                                            color='red',
                                            arrowstyle="->",
                                            mutation_scale=20,
                                            linestyle='--',
                                            linewidth=2,
                                            zorder=4)
            ax.add_patch(arrow)

        # 准备节点标签
        labels = {}
        for i, node in enumerate(traversal_path):
            concepts = graph.nodes[node].get('concepts', [])
            label = f"{i + 1}. {concepts[0] if concepts else ''}"
            labels[node] = label

        for node in traversal_graph.nodes():
            if node not in labels:
                concepts = graph.nodes[node].get('concepts', [])
                labels[node] = concepts[0] if concepts else ''

        # 绘制标签
        nx.draw_networkx_labels(traversal_graph, pos, labels, font_size=8, font_weight="bold", ax=ax)

        # 突出显示起始和结束节点
        start_node = traversal_path[0]
        end_node = traversal_path[-1]

        nx.draw_networkx_nodes(traversal_graph, pos,
                               nodelist=[start_node],
                               node_color='lightgreen',
                               node_size=3000,
                               ax=ax)

        nx.draw_networkx_nodes(traversal_graph, pos,
                               nodelist=[end_node],
                               node_color='lightcoral',
                               node_size=3000,
                               ax=ax)

        ax.set_title("图遍历流程")
        ax.axis('off')

        # 为边权重添加颜色条
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('边权重', rotation=270, labelpad=15)

        # 添加图例
        regular_line = plt.Line2D([0], [0], color='blue', linewidth=2, label='常规边')
        traversal_line = plt.Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='遍历路径')
        start_point = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=15, label='起始节点')
        end_point = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', markersize=15, label='结束节点')
        legend = plt.legend(handles=[regular_line, traversal_line, start_point, end_point], loc='upper left', bbox_to_anchor=(0, 1), ncol=2)
        legend.get_frame().set_alpha(0.8)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def print_filtered_content(traversal_path, filtered_content):
        """
        按遍历顺序打印访问节点的过滤内容。

        Args:
        - traversal_path (list of int): 表示遍历路径的节点索引列表。
        - filtered_content (dict of int: str): 将节点索引映射到其过滤内容的字典。

        Returns:
        - None
        """
        print("\n按遍历顺序访问节点的过滤内容:")
        for i, node in enumerate(traversal_path):
            print(f"\n步骤 {i + 1} - 节点 {node}:")
            print(f"过滤内容：{filtered_content.get(node, 'No filtered content available')[:200]}...")  # 打印前 200 个字符
            print("-" * 50)

```

### 定义 graph RAG 类

```python
class GraphRAG:
    def __init__(self):
        """
        初始化 GraphRAG 系统，包含文档处理、知识图谱构建、
        查询和可视化的组件。

        属性：
        - llm: 用于生成响应的大型语言模型（LLM）实例。
        - embedding_model: 用于文档嵌入的嵌入模型实例。
        - document_processor: 用于处理文档的 DocumentProcessor 类实例。
        - knowledge_graph: 用于构建和管理知识图谱的 KnowledgeGraph 类实例。
        - query_engine: 用于处理查询的 QueryEngine 类实例（初始化为 None）。
        - visualizer: 用于可视化知识图谱遍历的 Visualizer 类实例。
        """
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=4000)
        self.embedding_model = OpenAIEmbeddings()
        self.document_processor = DocumentProcessor()
        self.knowledge_graph = KnowledgeGraph()
        self.query_engine = None
        self.visualizer = Visualizer()

    def process_documents(self, documents):
        """
        通过将文档分割成块、嵌入它们和构建知识图谱来处理文档列表。

        Args:
        - documents (list of str): 要处理的文档列表。

        Returns:
        - None
        """
        splits, vector_store = self.document_processor.process_documents(documents)
        self.knowledge_graph.build_graph(splits, self.llm, self.embedding_model)
        self.query_engine = QueryEngine(vector_store, self.knowledge_graph, self.llm)

    def query(self, query: str):
        """
        通过从知识图谱检索相关信息和可视化遍历路径来处理查询。

        Args:
        - query (str): 要回答的查询。

        Returns:
        - str: 查询的响应。
        """
        response, traversal_path, filtered_content = self.query_engine.query(query)

        if traversal_path:
            self.visualizer.visualize_traversal(self.knowledge_graph.graph, traversal_path)
        else:
            print("No traversal path to visualize.")

        return response

```

### 定义文档路径

```python
path = "data/Understanding_Climate_Change.pdf"
```

### 加载文档

```python
loader = PyPDFLoader(path)
documents = loader.load()
documents = documents[:10]
```

### 创建 graph RAG 实例

```python
graph_rag = GraphRAG()
```

### 处理文档并创建图

```python
graph_rag.process_documents(documents)
```

### 输入查询并从 graph RAG 获取检索信息

```python
query = "what is the main cause of climate change?"
response = graph_rag.query(query)
```

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--graph-rag)