# 使用Milvus向量数据库的Graph RAG

## 概述

### 你将学到什么
本notebook演示了一种创新的**Graph RAG（检索增强生成）**方法，它将知识图谱的力量与向量数据库相结合，显著提高了问答性能，特别是对于复杂的多跳查询。在本教程结束时，你将了解如何构建一个Graph RAG系统，该系统可以回答需要多个逻辑步骤和关系遍历的问题。

### 问题：传统RAG的局限性
传统RAG系统虽然强大，但在以下方面存在困难：
- **多跳问题**：需要多个逻辑步骤的查询（例如，"Euler老师的儿子做出了什么贡献？"）
- **复杂的实体关系**：理解不同实体如何连接和关联
- **上下文碎片化**：重要的关系可能分散在不同的文本段落中
- **语义差距**：简单的相似性搜索可能会错过逻辑相关但语义距离较远的信息

### 解决方案：使用向量数据库的Graph RAG
本notebook提出了一种**统一方法**，仅使用**向量数据库**（Milvus）实现Graph RAG能力，消除了对单独图数据库的需求，同时保持卓越的性能。以下是这种方法的特别之处：

**关键创新**：我们不存储显式的图结构，而是将实体和关系作为向量嵌入，并使用智能检索和扩展技术来重建类似图的推理路径。

### 主要优势
1. **简化架构**：单一向量数据库，而非向量数据库+图数据库的组合
2. **卓越的多跳性能**：处理需要多个关系遍历的复杂查询
3. **可扩展**：利用Milvus的分布式架构进行十亿级部署
4. **成本效益**：降低基础设施复杂性和运营开销
5. **灵活**：适用于任何文本语料库 - 只需提取实体和关系

### 方法概述
我们的方法包括四个主要阶段：

1. **离线数据准备**
   - 从文本语料库中提取实体和关系（三元组）
   - 创建三个向量集合：实体、关系和段落
   - 构建实体和关系之间的邻接映射

2. **查询时检索**
   - 使用向量相似性搜索检索相似的实体和关系
   - 使用命名实体识别（NER）识别查询实体

3. **子图扩展**
   - 使用邻接矩阵将检索到的实体/关系扩展到其邻域
   - 支持多度扩展（1跳、2跳邻居）
   - 合并实体和关系扩展路径的结果

4. **LLM重排序**
   - 使用大型语言模型智能过滤和排序候选关系
   - 应用思维链推理选择最相关的关系
   - 返回最终段落用于答案生成

### 架构图
下图说明了完整的工作流程：

![](../images/graph_rag_with_milvus_1.png)

**为什么这有效**：通过将实体和关系都表示为向量，我们可以利用语义相似性进行初始检索，然后使用图论扩展来发现间接关系，最后应用LLM推理进行相关性过滤。这创建了一个"两全其美"的系统，结合了语义搜索、图遍历和智能推理。

---

## 技术实现

在本节中，我们将实现方法概述中描述的Graph RAG系统。实现遵循我们的四阶段方法：数据准备、向量存储、查询处理和智能重排序。

## 先决条件

要完成此演示，你需要一个向量数据库。你可以通过[注册Zilliz Cloud](https://cloud.zilliz.com/signup?utm_source=github&utm_medium=referral&utm_campaign=Nir-250512)获得完全托管的Milvus向量数据库。Milvus是一个开源向量数据库，提供高性能的向量相似性搜索。

安装以下依赖项：

```python
! pip install --upgrade --quiet pymilvus numpy scipy langchain langchain-core langchain-openai tqdm
```

> 如果你使用Google Colab，要启用刚安装的依赖项，你可能需要**重启运行时**（点击屏幕顶部的"运行时"菜单，从下拉菜单中选择"重启 session"）。

我们将使用OpenAI的模型。你需要准备[`OPENAI_API_KEY`](https://platform.openai.com/docs/quickstart)作为环境变量。

```python
import os

os.environ["OPENAI_API_KEY"] = "sk-***********"
```

导入必要的库和依赖项。

```python
import numpy as np

from collections import defaultdict
from scipy.sparse import csr_matrix
from pymilvus import MilvusClient
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from tqdm import tqdm
```

在Zilliz Cloud页面上找到你的公共端点和Token（即API密钥）。

![](../images/zilliz_interface.png)

```python
# `uri` 和 `token` 对应于您的 Zill 云集群的公共端点和令牌iz Cloud (fully-managed Milvus) cluster.milvus_client = MilvusClient(    uri="YOUR_ZILLIZ_PUBLIC_ENDPOINT",     token="YOUR_ZILLIZ_TOKEN")llm = ChatOpenAI(    model="gpt-4o",    temperature=0,)embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
```

## 离线数据加载

### 理解数据模型

在深入实现之前，理解我们如何构建数据以实现使用向量的类图推理至关重要。我们的方法将传统文本文档转换为三个相互关联的组件：

1. **实体**：我们概念图的"节点" - 人物、地点、概念等。
2. **关系**：连接实体的"边" - 这些是完整的三元组（主语-谓语-宾语）
3. **段落**：提供上下文和详细信息的原始文本文档

**为什么这种结构有效**：通过将实体和关系分离到不同的向量集合中，我们可以对查询的不同方面进行针对性搜索。当用户问"Euler老师的儿子做出了什么贡献？"时，我们可以：
- 找到与"Euler"相关的实体
- 找到连接师生和父子概念的关系
- 扩展图以发现间接连接
- 检索最相关的段落用于最终答案生成

### 数据准备

我们将使用一个介绍Bernoulli家族和Euler之间关系的微型数据集作为示例。该微型数据集包含4个段落和一组相应的三元组，每个三元组包含一个主语、一个谓语和一个宾语。

**三元组结构**：每个关系表示为一个三元组[主语, 谓语, 宾语]。例如：
- `["Jakob Bernoulli", "was the older brother of", "Johann Bernoulli"]` 捕获家庭关系
- `["Johann Bernoulli", "was a student of", "Leonhard Euler"]` 捕获教育关系

在实践中，你可以使用任何方法从自己的自定义语料库中提取三元组。常见方法包括：
- **命名实体识别（NER）** + **关系抽取**模型
- **开放信息抽取**系统如OpenIE
- **大型语言模型**配合结构化提示
- **人工标注**用于高精度领域

```python
nano_dataset = [
    {
        "passage": "Jakob Bernoulli (1654–1705): Jakob was one of the earliest members of the Bernoulli family to gain prominence in mathematics. He made significant contributions to calculus, particularly in the development of the theory of probability. He is known for the Bernoulli numbers and the Bernoulli theorem, a precursor to the law of large numbers. He was the older brother of Johann Bernoulli, another influential mathematician, and the two had a complex relationship that involved both collaboration and rivalry.",
        "triplets": [
            ["Jakob Bernoulli", "made significant contributions to", "calculus"],
            [
                "Jakob Bernoulli",
                "made significant contributions to",
                "the theory of probability",
            ],
            ["Jakob Bernoulli", "is known for", "the Bernoulli numbers"],
            ["Jakob Bernoulli", "is known for", "the Bernoulli theorem"],
            ["The Bernoulli theorem", "is a precursor to", "the law of large numbers"],
            ["Jakob Bernoulli", "was the older brother of", "Johann Bernoulli"],
        ],
    },
    {
        "passage": "Johann Bernoulli (1667–1748): Johann, Jakob's younger brother, was also a major figure in the development of calculus. He worked on infinitesimal calculus and was instrumental in spreading the ideas of Leibniz across Europe. Johann also contributed to the calculus of variations and was known for his work on the brachistochrone problem, which is the curve of fastest descent between two points.",
        "triplets": [
            [
                "Johann Bernoulli",
                "was a major figure of",
                "the development of calculus",
            ],
            ["Johann Bernoulli", "was", "Jakob's younger brother"],
            ["Johann Bernoulli", "worked on", "infinitesimal calculus"],
            ["Johann Bernoulli", "was instrumental in spreading", "Leibniz's ideas"],
            ["Johann Bernoulli", "contributed to", "the calculus of variations"],
            ["Johann Bernoulli", "was known for", "the brachistochrone problem"],
        ],
    },
    {
        "passage": "Daniel Bernoulli (1700–1782): The son of Johann Bernoulli, Daniel made major contributions to fluid dynamics, probability, and statistics. He is most famous for Bernoulli's principle, which describes the behavior of fluid flow and is fundamental to the understanding of aerodynamics.",
        "triplets": [
            ["Daniel Bernoulli", "was the son of", "Johann Bernoulli"],
            ["Daniel Bernoulli", "made major contributions to", "fluid dynamics"],
            ["Daniel Bernoulli", "made major contributions to", "probability"],
            ["Daniel Bernoulli", "made major contributions to", "statistics"],
            ["Daniel Bernoulli", "is most famous for", "Bernoulli's principle"],
            [
                "Bernoulli's principle",
                "is fundamental to",
                "the understanding of aerodynamics",
            ],
        ],
    },
    {
        "passage": "Leonhard Euler (1707–1783) was one of the greatest mathematicians of all time, and his relationship with the Bernoulli family was significant. Euler was born in Basel and was a student of Johann Bernoulli, who recognized his exceptional talent and mentored him in mathematics. Johann Bernoulli's influence on Euler was profound, and Euler later expanded upon many of the ideas and methods he learned from the Bernoullis.",
        "triplets": [
            [
                "Leonhard Euler",
                "had a significant relationship with",
                "the Bernoulli family",
            ],
            ["leonhard Euler", "was born in", "Basel"],
            ["Leonhard Euler", "was a student of", "Johann Bernoulli"],
            ["Johann Bernoulli's influence", "was profound on", "Euler"],
        ],
    },
]
```

我们按如下方式构建实体和关系：
- 实体是三元组中的主语或宾语，因此我们直接从三元组中提取它们。
- 在这里，我们通过直接将主语、谓语和宾语用空格连接来构建关系的概念。

我们还准备一个字典将实体ID映射到关系ID，以及另一个字典将关系ID映射到段落ID，供后续使用。

### 构建知识图谱结构

下一步将我们的三元组转换为可搜索的向量格式，同时保持图连接信息。此过程涉及几个关键决策：

**实体提取策略**：我们通过收集所有主语和宾语来提取唯一实体。这确保我们捕获任何关系中提到的每个实体，创建知识域的全面覆盖。

**关系表示**：我们不是将关系存储为单独的主语-谓语-宾语组件，而是将它们连接成自然语言句子。例如，`["Jakob Bernoulli", "was the older brother of", "Johann Bernoulli"]`变成`"Jakob Bernoulli was the older brother of Johann Bernoulli"`。这种方法提供了几个优势：
- **语义丰富性**：完整句子为向量嵌入提供更多上下文
- **自然语言兼容性**：LLM可以轻松理解和推理完整句子
- **降低复杂性**：无需管理单独的谓语词汇表

**邻接映射构建**：我们构建两个关键的映射结构：
1. **`entityid_2_relationids`**：将每个实体映射到它参与的所有关系（实现实体到关系的扩展）
2. **`relationid_2_passageids`**：将每个关系映射到它出现的段落（实现关系到段落的检索）

这些映射对于子图扩展过程至关重要，允许我们在查询时高效地遍历概念图。

```python
entityid_2_relationids = defaultdict(list)
relationid_2_passageids = defaultdict(list)

entities = []
relations = []
passages = []
for passage_id, dataset_info in enumerate(nano_dataset):
    passage, triplets = dataset_info["passage"], dataset_info["triplets"]
    passages.append(passage)
    for triplet in triplets:
        if triplet[0] not in entities:
            entities.append(triplet[0])
        if triplet[2] not in entities:
            entities.append(triplet[2])
        relation = " ".join(triplet)
        if relation not in relations:
            relations.append(relation)
            entityid_2_relationids[entities.index(triplet[0])].append(
                len(relations) - 1
            )
            entityid_2_relationids[entities.index(triplet[2])].append(
                len(relations) - 1
            )
        relationid_2_passageids[relations.index(relation)].append(passage_id)
```

### 数据插入

为实体、关系和段落创建Milvus集合。我们创建三个独立的Milvus集合，每个都针对不同类型的检索进行了优化：

1. **实体集合**：存储实体名称和描述的向量嵌入
   - **目的**：支持以实体为中心的查询，如"查找与'Euler'相似的实体"
   - **搜索模式**：与查询实体的直接语义相似性

2. **关系集合**：存储完整关系句子的向量嵌入
   - **目的**：捕获与查询意图匹配的关系中的语义模式
   - **搜索模式**：查找与整个查询语义相似的关系

3. **段落集合**：存储原始文本段落的向量嵌入
   - **目的**：提供比较基准和最终答案的详细上下文
   - **搜索模式**：传统RAG风格的文档检索

**为什么三个集合？** 这种分离允许**多模态检索**：
- 如果查询提到特定实体，我们通过实体集合检索
- 如果查询描述关系或动作，我们通过关系集合检索
- 我们可以合并两条路径的结果，并与传统段落检索进行比较

**嵌入一致性**：所有集合使用相同的嵌入模型，以确保相似性搜索和结果合并时的兼容性。

```python
embedding_dim = len(embedding_model.embed_query("foo"))


def create_milvus_collection(collection_name: str):
    """
    Create a new Milvus collection with specified configuration.
    
    This function creates a new Milvus collection for storing vector embeddings.
    If a collection with the same name already exists, it will be dropped first
    to ensure a clean state.
    
    Args:
        collection_name (str): The name of the collection to create.
    """
    if milvus_client.has_collection(collection_name=collection_name):
        milvus_client.drop_collection(collection_name=collection_name)
    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=embedding_dim,
        consistency_level="Strong",
    )


entity_col_name = "entity_collection"
relation_col_name = "relation_collection"
passage_col_name = "passage_collection"
create_milvus_collection(entity_col_name)
create_milvus_collection(relation_col_name)
create_milvus_collection(passage_col_name)
```

将数据及其元数据信息插入Milvus集合，包括实体、关系和段落集合。元数据信息包括段落ID和邻接实体或关系ID。

```python
def milvus_insert(
    collection_name: str,
    text_list: list[str],
):
    """
    Insert text data with embeddings into a Milvus collection in batches.
    
    This function processes a list of text strings, generates embeddings for them,
    and inserts the data into the specified Milvus collection in batches for
    efficient processing.
    
    Args:
        collection_name (str): The name of the Milvus collection to insert data into.
        text_list (list[str]): A list of text strings to be embedded and inserted.
    """
    batch_size = 512
    for row_id in tqdm(range(0, len(text_list), batch_size), desc="Inserting"):
        batch_texts = text_list[row_id : row_id + batch_size]
        batch_embeddings = embedding_model.embed_documents(batch_texts)

        batch_ids = [row_id + j for j in range(len(batch_texts))]
        batch_data = [
            {
                "id": id_,
                "text": text,
                "vector": vector,
            }
            for id_, text, vector in zip(batch_ids, batch_texts, batch_embeddings)
        ]
        milvus_client.insert(
            collection_name=collection_name,
            data=batch_data,
        )


milvus_insert(
    collection_name=relation_col_name,
    text_list=relations,
)

milvus_insert(
    collection_name=entity_col_name,
    text_list=entities,
)

milvus_insert(
    collection_name=passage_col_name,
    text_list=passages,
)
```

## 在线查询

### 理解查询处理管道

查询阶段实现了我们的核心创新：将语义向量搜索与图遍历逻辑相结合。这个多阶段过程将自然语言问题转换为相关知识，遵循以下步骤：

1. **实体识别**：使用NER提取查询中提到的实体
2. **双重检索**：同时搜索实体和关系集合
3. **图扩展**：使用邻接信息发现间接连接
4. **LLM重排序**：应用智能过滤选择最相关的关系
5. **答案生成**：检索最终段落并生成响应

### 相似性检索

我们根据输入查询从Milvus中检索topK相似的实体和关系。

在执行实体检索时，我们首先应该使用某些特定方法（如NER命名实体识别）从查询文本中提取查询实体。为简单起见，我们在这里准备了NER结果。如果你想将查询更改为自定义问题，你必须更改相应的查询NER列表。
在实践中，你可以使用任何其他模型或方法从查询中提取实体。

### 双路径检索策略

我们的方法执行两个并行相似性搜索：

**路径1：基于实体的检索**
- **输入**：从查询中提取的实体（使用NER）
- **过程**：在我们的知识库中找到与查询实体相似的实体
- **为什么使用NER？**：许多复杂查询引用特定实体（"Euler"、"Bernoulli家族"）。通过显式识别这些，我们可以找到直接匹配及其相关关系
- **示例**：对于"Euler老师的儿子做出了什么贡献？"，NER识别"Euler"为关键实体

**路径2：基于关系的检索**
- **输入**：完整的查询文本
- **过程**：找到与查询意图语义匹配的关系
- **目的**：捕获关系模式和问题结构
- **示例**：查询模式"X老师的儿子做出了什么贡献"匹配关于家庭连接和贡献的关系模式

**双重检索的优势**：
- **全面覆盖**：实体路径捕获直接提及，关系路径捕获语义模式
- **冗余提高健壮性**：如果一条路径遗漏相关信息，另一条可能捕获它
- **不同粒度**：实体提供特定锚点，关系提供结构模式

```python
query = "What contribution did the son of Euler's teacher make?"query_ner_list = ["Euler"]# query_ner_list = ner(query) # 在实践中，用您的自定义 NER 应用替换roachquery_ner_embeddings = [    embedding_model.embed_query(query_ner) for query_ner in query_ner_list]top_k = 3entity_search_res = milvus_client.search(    collection_name=entity_col_name,    data=query_ner_embeddings,    limit=top_k,    output_fields=["id"],)query_embedding = embedding_model.embed_query(query)relation_search_res = milvus_client.search(    collection_name=relation_col_name,    data=[query_embedding],    limit=top_k,    output_fields=["id"],)[0]
```

### 扩展子图

我们使用检索到的实体和关系来扩展子图并获得候选关系，然后从两种方式合并它们。这是子图扩展过程的流程图：
![](../images/graph_rag_with_milvus_2.png)

在这里，我们构建一个邻接矩阵，并使用矩阵乘法在几度内计算邻接映射信息。这样，我们可以快速获得任意度扩展的信息。

### 图扩展的数学原理

子图扩展步骤是我们方法真正闪光的地方。我们不存储显式的图数据库，而是使用**邻接矩阵**和**矩阵乘法**来高效计算多跳关系。这种数学方法提供了几个优势：

**邻接矩阵构建**：我们创建一个二元矩阵，如果实体`i`参与关系`j`，则`entity_relation_adj[i][j] = 1`，否则为0。这种稀疏表示捕获了整个图结构。

**通过矩阵幂实现多度扩展**：
- **1度扩展**：`entity_adj_1_degree = entity_relation_adj @ entity_relation_adj.T`
- **2度扩展**：`entity_adj_2_degree = entity_adj_1_degree @ entity_adj_1_degree`
- **n度扩展**：通过将1度矩阵提升到n次幂来计算

**为什么这有效**：矩阵乘法自然地实现了图遍历。当我们乘以邻接矩阵时，我们正在计算图中的路径：
- 1跳：直接连接的实体/关系
- 2跳：通过一个中间实体连接的实体
- n跳：通过(n-1)个中间步骤连接的实体

**计算效率**：使用稀疏矩阵和向量化操作，我们可以在毫秒内扩展包含数千个实体的子图，使这种方法高度可扩展。

**双重扩展策略**：我们从检索到的实体和检索到的关系两个方向扩展，然后合并结果。这确保我们捕获相关信息，无论初始检索在实体还是关系方面更成功。

```python
# 构建实体和关系的邻接矩阵，其中值为e adjacency matrix is 1 if an entity is related to a relation, otherwise 0.entity_relation_adj = np.zeros((len(entities), len(relations)))for entity_id, entity in enumerate(entities):    entity_relation_adj[entity_id, entityid_2_relationids[entity_id]] = 1# 将邻接矩阵转换为稀疏矩阵以提高计算效率。entity_relation_adj = csr_matrix(entity_relation_adj)# 使用实体 - 关系邻接矩阵构建 1 度实体 - 实体邻接nd relation-relation adjacency matrices.entity_adj_1_degree = entity_relation_adj @ entity_relation_adj.Trelation_adj_1_degree = entity_relation_adj.T @ entity_relation_adj# 指定要扩展的子图的目标度数。# 对于大多数情况，1 或 2 就足够了。target_degree = 1# 使用矩阵乘法计算目标度数邻接矩阵。entity_adj_target_degree = entity_adj_1_degreefor _ in range(target_degree - 1):    entity_adj_target_degree = entity_adj_target_degree @ entity_adj_1_degree.Trelation_adj_target_degree = relation_adj_1_degreefor _ in range(target_degree - 1):    relation_adj_target_degree = relation_adj_target_degree @ relation_adj_1_degree.Tentity_relation_adj_target_degree = entity_adj_target_degree @ entity_relation_adj
```

通过从目标度扩展矩阵中取值，我们可以轻松地从检索到的实体和关系扩展相应度数，以获得子图的所有关系。

```python
expanded_relations_from_relation = set()expanded_relations_from_entity = set()filtered_hit_relation_ids = [    relation_res["entity"]["id"]    for relation_res in relation_search_res]for hit_relation_id in filtered_hit_relation_ids:    expanded_relations_from_relation.update(        relation_adj_target_degree[hit_relation_id].nonzero()[1].tolist()    )filtered_hit_entity_ids = [    one_entity_res["entity"]["id"]    for one_entity_search_res in entity_search_res    for one_entity_res in one_entity_search_res]for filtered_hit_entity_id in filtered_hit_entity_ids:    expanded_relations_from_entity.update(        entity_relation_adj_target_degree[filtered_hit_entity_id].nonzero()[1].tolist()    )# 合并来自关系和实体检索方式的扩展关系。relation_candidate_ids = list(    expanded_relations_from_relation | expanded_relations_from_entity)relation_candidate_texts = [    relations[relation_id] for relation_id in relation_candidate_ids]
```

我们已经通过扩展子图获得了候选关系，下一步将由LLM对其进行重排序。

### LLM重排序

在这个阶段，我们部署LLM强大的自注意力机制来进一步过滤和细化候选关系集。子图扩展步骤为我们提供了许多潜在相关的关系，但并非所有关系对回答我们的特定查询都同样有用。这就是**大型语言模型**擅长的地方 - 它们可以理解查询和候选关系的语义含义，然后智能地选择最相关的关系。

**为什么LLM重排序是必要的**：
- **语义理解**：LLM可以理解纯相似性搜索可能遗漏的复杂查询意图
- **多跳推理**：LLM可以追踪跨多个关系的逻辑连接
- **上下文感知**：LLM考虑关系如何协同工作来回答查询
- **质量过滤**：LLM可以识别并优先考虑最具信息量的关系

**思维链提示策略**：
我们使用结构化方法鼓励LLM：
1. **分析查询**：分解回答问题需要什么信息
2. **识别关键连接**：确定哪些类型的关系最有帮助
3. **推理相关性**：解释为什么选择特定关系
4. **按重要性排序**：根据对最终答案的效用对关系进行排序

**单样本学习模式**：我们提供一个具体的推理过程示例来指导LLM的行为。这个示例演示了如何识别核心实体、追踪多跳连接以及优先考虑最直接的关系。

**JSON输出格式**：通过要求结构化JSON输出，我们确保可靠解析和一致结果，使系统适用于生产环境。

#### 定义单样本学习示例

首先，我们准备单样本学习示例来指导LLM的推理过程：

```python
query_prompt_one_shot_input = """I will provide you with a list of relationship descriptions. Your task is to select 3 relationships that may be useful to answer the given question. Please return a JSON object containing your thought process and a list of the selected relationships in order of their relevance.

Question:
When was the mother of the leader of the Third Crusade born?

Relationship descriptions:
[1] Eleanor was born in 1122.
[2] Eleanor married King Louis VII of France.
[3] Eleanor was the Duchess of Aquitaine.
[4] Eleanor participated in the Second Crusade.
[5] Eleanor had eight children.
[6] Eleanor was married to Henry II of England.
[7] Eleanor was the mother of Richard the Lionheart.
[8] Richard the Lionheart was the King of England.
[9] Henry II was the father of Richard the Lionheart.
[10] Henry II was the King of England.
[11] Richard the Lionheart led the Third Crusade.

"""
query_prompt_one_shot_output = """{"thought_process": "To answer the question about the birth of the mother of the leader of the Third Crusade, I first need to identify who led the Third Crusade and then determine who his mother was. After identifying his mother, I can look for the relationship that mentions her birth.", "useful_relationships": ["[11] Richard the Lionheart led the Third Crusade", "[7] Eleanor was the mother of Richard the Lionheart", "[1] Eleanor was born in 1122"]}"""
```

#### 创建查询提示模板

接下来，我们定义格式化新查询的模板：

```python
query_prompt_template = """Question:
{question}

Relationship descriptions:
{relation_des_str}

"""
```

#### 实现重排序函数

现在我们实现处理候选关系的核心重排序函数：

```python
def rerank_relations(
    query: str, relation_candidate_texts: list[str], relation_candidate_ids: list[str]
) -> list[int]:
    """
    Rerank candidate relations using LLM to select the most relevant ones for answering a query.
    
    This function uses a large language model with Chain-of-Thought prompting to analyze
    candidate relationships and select the most useful ones for answering the given query.
    It employs a one-shot learning approach with a predefined example to guide the LLM's
    reasoning process.
    
    Args:
        query (str): The input question that needs to be answered.
        relation_candidate_texts (list[str]): List of candidate relationship descriptions.
        relation_candidate_ids (list[str]): List of IDs corresponding to the candidate relations.
        
    Returns:
        list[int]: A list of relation IDs ranked by their relevance to the query.
    """
    relation_des_str = "\n".join(
        map(
            lambda item: f"[{item[0]}] {item[1]}",
            zip(relation_candidate_ids, relation_candidate_texts),
        )
    ).strip()
    rerank_prompts = ChatPromptTemplate.from_messages(
        [
            HumanMessage(query_prompt_one_shot_input),
            AIMessage(query_prompt_one_shot_output),
            HumanMessagePromptTemplate.from_template(query_prompt_template),
        ]
    )
    rerank_chain = (
        rerank_prompts
        | llm.bind(response_format={"type": "json_object"})
        | JsonOutputParser()
    )
    rerank_res = rerank_chain.invoke(
        {"question": query, "relation_des_str": relation_des_str}
    )
    rerank_relation_ids = []
    rerank_relation_lines = rerank_res["useful_relationships"]
    id_2_lines = {}
    for line in rerank_relation_lines:
        id_ = int(line[line.find("[") + 1 : line.find("]")])
        id_2_lines[id_] = line.strip()
        rerank_relation_ids.append(id_)
    return rerank_relation_ids
```

#### 执行重排序过程

最后，我们对候选关系应用重排序函数：

```python
rerank_relation_ids = rerank_relations(
    query,
    relation_candidate_texts=relation_candidate_texts,
    relation_candidate_ids=relation_candidate_ids,
)
```

### 获取最终结果

我们可以从重排序的关系中获取最终检索的段落。最后一步通过与传统RAG方法直接比较来展示我们Graph RAG方法的优势。这种比较揭示了为什么基于图的推理对于复杂的多跳问题至关重要。

**我们的方法 - Graph RAG过程**：
1. 从LLM过滤的重排序关系开始
2. 使用`relationid_2_passageids`将关系映射回其源段落
3. 收集唯一段落同时保持相关性顺序
4. 返回前k个最相关的段落用于答案生成

**基线 - 朴素RAG过程**：
1. 使用查询嵌入直接搜索段落集合
2. 返回前k个语义最相似的段落
3. 不考虑实体关系或图结构

**关键差异**：
- **Graph RAG**：通过实体关系推理找到相关上下文
- **朴素RAG**：仅依赖查询和段落之间的表面语义相似性

**预期结果**：对于像"Euler老师的儿子做出了什么贡献？"这样的多跳问题，我们的Graph RAG方法应该：
- **识别推理链**：Euler -> Johann Bernoulli（老师）-> Daniel Bernoulli（儿子）-> 贡献
- **检索相关段落**：找到关于Daniel Bernoulli对流体力学贡献的段落
- **提供准确答案**：基于正确的上下文信息生成响应

相比之下，朴素RAG可能直接检索关于Euler的段落或完全遗漏多跳连接，导致不完整或不正确的答案。

```python
final_top_k = 2

final_passages = []
final_passage_ids = []
for relation_id in rerank_relation_ids:
    for passage_id in relationid_2_passageids[relation_id]:
        if passage_id not in final_passage_ids:
            final_passage_ids.append(passage_id)
            final_passages.append(passages[passage_id])
passages_from_our_method = final_passages[:final_top_k]
```

我们可以将结果与朴素RAG方法进行比较，该方法直接根据查询嵌入从段落集合中检索topK段落。

```python
naive_passage_res = milvus_client.search(
    collection_name=passage_col_name,
    data=[query_embedding],
    limit=final_top_k,
    output_fields=["text"],
)[0]
passages_from_naive_rag = [res["entity"]["text"] for res in naive_passage_res]

print(
    f"Passages retrieved from naive RAG: \n{passages_from_naive_rag}\n\n"
    f"Passages retrieved from our method: \n{passages_from_our_method}\n\n"
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """Use the following pieces of retrieved context to answer the question. If there is not enough information in the retrieved context to answer the question, just say that you don't know.
Question: {question}
Context: {context}
Answer:""",
        )
    ]
)

rag_chain = prompt | llm | StrOutputParser()

answer_from_naive_rag = rag_chain.invoke(
    {"question": query, "context": "\n".join(passages_from_naive_rag)}
)
answer_from_our_method = rag_chain.invoke(
    {"question": query, "context": "\n".join(passages_from_our_method)}
)

print(
    f"Answer from naive RAG: {answer_from_naive_rag}\n\nAnswer from our method: {answer_from_our_method}"
)
```

结果显示，朴素RAG检索的段落遗漏了一个真实段落，这导致了错误的答案。

我们的方法检索的段落是正确的，它有助于获得问题的准确答案。

### 关键洞察和学习成果

比较结果清楚地展示了我们的Graph RAG方法在多跳推理任务中的优越性。让我们分析我们取得的成果：

**性能分析**：
- **朴素RAG的局限性**：传统相似性搜索失败，因为查询"Euler老师的儿子做出了什么贡献？"与Daniel Bernoulli流体力学贡献的段落没有高语义相似性。表面级别的关键词匹配不佳。
- **Graph RAG的成功**：我们的方法成功追踪了逻辑链：查询提到"Euler" -> 实体检索找到"Leonhard Euler" -> 图扩展发现"Johann Bernoulli是Euler的老师" -> 进一步扩展发现"Daniel Bernoulli是Johann的儿子" -> 关系过滤识别Daniel的贡献 -> 检索到正确的段落。

**展示的方法创新**：
1. **仅向量的Graph RAG**：我们仅使用向量数据库实现了图级推理，消除了架构复杂性
2. **多模态检索**：结合基于实体和基于关系的搜索路径提供了冗余和改进的覆盖范围
3. **数学图扩展**：稀疏矩阵操作实现了规模化的高效多跳遍历
4. **LLM驱动的过滤**：思维链推理提供了超越简单相似性的智能关系选择

**实际应用**：
这种方法在需要复杂推理的领域中表现出色：
- **知识库**：科学文献、历史记录、技术文档
- **企业搜索**：在相互关联的业务实体和流程中查找信息
- **问答系统**：学术研究、法律文档分析、医学知识检索
- **内容推荐**：通过关系网络理解用户意图

**可扩展性考虑**：
- **向量数据库扩展**：Milvus可以通过分布式架构处理数十亿向量
- **矩阵计算效率**：稀疏矩阵操作随数据大小对数扩展
- **LLM推理优化**：重排序步骤可以并行化和缓存以处理重复模式

本教程表明，通过精心的系统设计，即使使用更简单的基础设施组件，也可以实现复杂的推理能力。这种力量和简单性的平衡使该方法非常适用于实际部署。

## 使用完全托管的Milvus扩展你的Graph RAG系统

虽然本教程中的示例适用于小型数据集，但在生产中实施大规模数据的Graph RAG需要健壮的基础设施。Milvus是一个可以扩展到数十亿的分布式向量数据库，使其成为管理大规模向量数据的可靠选择。在生产中管理自托管的Milvus集群可能会变得具有挑战性。如果你的优先事项是为RAG应用程序开发业务逻辑，Zilliz Cloud提供完全托管的Milvus服务，为你处理所有运营复杂性：

![Zilliz Cloud截图](../images/zilliz_screenshot.png)

- **生产就绪**：内置高可用性和安全功能，对关键任务AI应用程序至关重要
- **10倍更快的性能**：其专有的Cardinal向量索引引擎即使与高性能开源Milvus相比也提供10倍更快的性能。
- **AutoIndex**：AutoIndex功能节省了索引选择和参数调优的工作。
- **更低的总体拥有成本（TCO）**：专注于你的应用程序，我们处理扩展、更新和监控，只需为你使用的内容付费，具有灵活的定价层级，与管理自托管的Milvus集群相比，TCO更低

[**立即免费试用Zilliz Cloud ->**](https://cloud.zilliz.com/signup?utm_source=github&utm_medium=referral&utm_campaign=Nir-250512)

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--graphrag-with-milvus-vectordb)
