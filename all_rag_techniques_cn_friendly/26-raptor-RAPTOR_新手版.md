# 🌟 新手入门：RAPTOR（递归抽象处理与主题组织检索）

> **💡 给新手的说明**
> - **难度等级**：⭐⭐⭐☆☆（中等）
> - **预计学习时间**：45-60 分钟
> - **前置知识**：了解基本的 RAG 概念，有 Python 编程经验
> - **学完你将掌握**：如何用树形摘要结构实现多层次信息检索
>
> **🤔 为什么要学这个？** RAPTOR 像给文档建了一座"金字塔"：顶层是概括性摘要，底层是具体细节。无论是宏观问题还是细节问题，都能快速找到答案！

---

## 📖 核心概念理解

### 什么是 RAPTOR？

**RAPTOR** = **R**ecursive **A**bstractive **P**rocessing and **T**hematic **O**rganization with **R**etrieval

中文意思是：**递归抽象处理与主题组织检索**

它是一个通过创建**层次化文档摘要树**来实现高效检索的问答系统。

### 通俗理解：公司的组织架构

想象一个大型公司的信息结构：

```
                    CEO（顶层摘要）
                     ↓
            ┌────────┼────────┐
            ↓        ↓        ↓
        技术部    市场部    财务部  （中层摘要）
          ↓          ↓        ↓
      ┌───┴───┐  ┌──┴──┐  ┌─┴─┐
      ↓       ↓  ↓     ↓  ↓ ↓
    前端组 后端组 品牌 销售 会计 出纳（底层细节）
```

**RAPTOR 的工作方式：**

1. **自底向上构建**：
   - 底层：原始文档（员工）
   - 中层：部门摘要（部门职责）
   - 顶层：公司概览（CEO 总结）

2. **查询时的检索**：
   - 宏观问题 → 从 CEO 开始往下找
   - 具体问题 → 直接找对应部门

### 核心组件一览

| 组件 | 作用 | 生活化比喻 |
|------|------|-----------|
| **树构建 (Tree Building)** | 创建多层摘要 | 写报告：先写段落总结，再写章节总结，最后写全文总结 |
| **聚类 (Clustering)** | 将相似内容分组 | 把相关文件归类到同一个文件夹 |
| **向量存储 (Vector Store)** | 存储所有层级的内容 | 图书馆的索引卡片系统 |
| **上下文检索器** | 选择最相关的信息 | 图书管理员帮你找书 |
| **答案生成** | 基于检索结果生成答案 | 根据找到的资料写回答 |

### RAPTOR 树结构示意

```
Level 3:           [整本书的摘要]
                      ↑
Level 2:    [第 1 章摘要] [第 2 章摘要] [第 3 章摘要]
                 ↑          ↑          ↑
Level 1:   [1.1 节][1.2 节] [2.1 节][2.2 节] ...
              ↑      ↑       ↑      ↑
Level 0:  段落 1  段落 2   段落 3  段落 4  ... (原始文本)
```

---

## 🛠️ 第一步：环境准备

### 📖 这是什么？

安装 RAPTOR 所需的所有依赖包。

### 💻 完整代码

```python
# 安装所需的包
# 每个包的作用：
# - faiss-cpu: 向量相似度搜索
# - langchain: RAG 框架
# - langchain-openai: OpenAI 集成
# - matplotlib: 绘图可视化
# - numpy: 数值计算
# - pandas: 数据处理和分析
# - python-dotenv: 环境变量管理
# - scikit-learn: 机器学习（用于聚类）
!pip install faiss-cpu langchain langchain-openai matplotlib numpy pandas python-dotenv scikit-learn

# 克隆仓库以访问辅助函数
!git clone https://github.com/NirDiamant/RAG_TECHNIQUES.git
import sys
sys.path.append('RAG_TECHNIQUES')
```

> **💡 代码解释**
>
> **核心包说明：**
>
> | 包名 | 用途 | RAPTOR 中的作用 |
> |------|------|----------------|
> | `scikit-learn` | 机器学习库 | 提供高斯混合模型（GMM）用于聚类 |
> | `pandas` | 数据处理 | 存储和管理树结构数据 |
> | `faiss-cpu` | 向量搜索 | 快速检索相似内容 |
> | `matplotlib` | 绘图 | 可视化聚类结果 |
>
> **⚠️ 新手注意**
> - 安装可能需要 3-5 分钟
> - 如果安装失败，尝试先升级 pip
> - 某些系统可能需要额外安装系统库

### 导入库和设置环境

```python
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.mixture import GaussianMixture
from langchain.chains.llm import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import AIMessage
from langchain.docstore.document import Document
import matplotlib.pyplot as plt
import logging
import os
import sys
from dotenv import load_dotenv

# 从 .env 文件加载环境变量
load_dotenv()

# 设置 OpenAI API 密钥
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# 设置日志记录（方便调试）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

> **💡 代码解释**
>
> **GaussianMixture（高斯混合模型）：**
> - 一种聚类算法
> - 比 K-Means 更灵活
> - 可以处理不同大小和形状的簇
>
> **为什么用 pandas？**
> - DataFrame 方便存储树结构
> - 可以轻松添加元数据（层级、父子关系等）
> - 便于查询和筛选
>
> **⚠️ 新手注意**
> - 日志级别设为 INFO 可以看到详细处理过程
> - 生产环境可以设为 WARNING 减少输出

---

## 🛠️ 第二步：初始化核心组件

### 📖 这是什么？

初始化 LLM 和嵌入模型，这些是后续所有操作的基础。

### 💻 完整代码

```python
# 初始化嵌入模型和语言模型
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-4o-mini")

print("✓ 核心组件初始化完成")
```

> **💡 代码解释**
> - `OpenAIEmbeddings()`：创建嵌入模型，用于将文本转换为向量
> - `ChatOpenAI()`：创建聊天模型，用于生成摘要和答案
> - `gpt-4o-mini`：性价比高，适合此类任务

---

## 🛠️ 第三步：辅助函数定义

### 📖 这是什么？

定义一些通用的辅助函数，用于后续的核心逻辑。

### 💻 完整代码

```python
def extract_text(item):
    """
    从字符串或 AIMessage 对象中提取文本内容

    为什么要这个函数？
    LLM 可能返回不同类型的对象，需要统一处理

    参数:
        item: 字符串或 AIMessage 对象

    返回:
        文本字符串
    """
    if isinstance(item, AIMessage):
        return item.content  # AIMessage 对象需要取 content 属性
    return item  # 字符串直接返回

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    使用 OpenAI Embeddings 对文本进行嵌入

    参数:
        texts: 文本列表

    返回:
        嵌入向量列表（每个向量是一个浮点数列表）

    示例:
        输入：["hello", "world"]
        输出：[[0.1, 0.2, ...], [0.3, 0.4, ...]]
    """
    logging.info(f"正在为 {len(texts)} 个文本创建嵌入")
    return embeddings.embed_documents([extract_text(text) for text in texts])

def perform_clustering(embeddings: np.ndarray, n_clusters: int = 10) -> np.ndarray:
    """
    使用高斯混合模型对嵌入进行聚类

    参数:
        embeddings: 嵌入向量数组（numpy 数组）
        n_clusters: 聚类数量

    返回:
        每个样本的簇标签数组

    聚类的作用:
        将语义相似的文本分组到一起
        后续可以为每个组生成摘要
    """
    logging.info(f"正在执行聚类，目标簇数：{n_clusters}")

    # 创建高斯混合模型
    # random_state=42 确保结果可重复
    gm = GaussianMixture(n_components=n_clusters, random_state=42)

    # 拟合并预测簇标签
    return gm.fit_predict(embeddings)

def summarize_texts(texts: List[str]) -> str:
    """
    使用 OpenAI 总结文本列表

    参数:
        texts: 要总结的文本列表

    返回:
        摘要字符串

    为什么需要总结？
        将多个相关内容合并成一个概括性描述
        形成树的上一层级
    """
    logging.info(f"正在总结 {len(texts)} 个文本")

    # 创建提示模板
    prompt = ChatPromptTemplate.from_template(
        "简洁地总结以下文本：\n\n{text}"
    )

    # 创建链：提示模板 → LLM
    chain = prompt | llm

    # 调用链
    input_data = {"text": texts}
    return chain.invoke(input_data)

def visualize_clusters(embeddings: np.ndarray, labels: np.ndarray, level: int):
    """
    使用 PCA 可视化聚类结果

    参数:
        embeddings: 嵌入向量
        labels: 簇标签
        level: 树的层级（用于标题）

    原理:
        PCA(主成分分析) 将高维向量降到 2D 便于可视化
        不同簇用不同颜色表示
    """
    from sklearn.decomposition import PCA

    # 降维到 2D
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    # 创建图形
    plt.figure(figsize=(10, 8))

    # 散点图，颜色表示簇
    scatter = plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=labels,
        cmap='viridis'  # 颜色映射
    )

    plt.colorbar(scatter)  # 颜色条
    plt.title(f'Cluster Visualization - Level {level}')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()
```

> **💡 代码解释**
>
> **为什么需要聚类？**
>
> 想象你有 100 个文档段落：
> - 直接总结：信息太散乱
> - 先聚类：把相关内容分组，然后每组分别总结
> - 结果更有组织结构
>
> **PCA 可视化原理：**
> ```
> 原始嵌入：1536 维（OpenAI 嵌入维度）
>     ↓ PCA 降维
> 2 个主成分：保留最多信息的 2 个方向
>     ↓ 绘图
> 2D 散点图：每个点是一个文本，颜色表示所属簇
> ```
>
> **⚠️ 新手注意**
> - `random_state=42` 确保每次运行结果一致
> - 聚类数 `n_clusters` 应该小于文本数量的一半
> - PCA 只用于可视化，不影响实际处理

---

## 🛠️ 第四步：构建 RAPTOR 树

### 📖 这是什么？

这是 RAPTOR 的核心函数，负责创建层次化的摘要树结构。

### 💻 完整代码

```python
def build_raptor_tree(texts: List[str], max_levels: int = 3) -> Dict[int, pd.DataFrame]:
    """
    构建带有层级元数据和父子关系的 RAPTOR 树结构

    参数:
        texts: 原始文本列表（第 0 层）
        max_levels: 最大层数（不包括第 0 层）

    返回:
        字典：{层级：DataFrame}，每个 DataFrame 包含该层的所有节点

    树结构示意:
        Level 3:      [摘要]
                       ↑
        Level 2:   [摘要] [摘要]
                     ↑      ↑
        Level 1: [摘要] [摘要] [摘要]
                   ↑     ↑     ↑
        Level 0: 原文  原文   原文
    """
    results = {}  # 存储每一层的结果

    # 初始化第 0 层（原始文本）
    current_texts = [extract_text(text) for text in texts]
    current_metadata = [
        {"level": 0, "origin": "original", "parent_id": None}
        for _ in texts
    ]

    # 逐层向上构建
    for level in range(1, max_levels + 1):
        logging.info(f"正在处理第 {level} 层")

        # 步骤 1：为当前层的文本创建嵌入
        embeddings = embed_texts(current_texts)

        # 步骤 2：确定聚类数（不超过文本数的一半）
        n_clusters = min(10, len(current_texts) // 2)

        # 步骤 3：执行聚类
        cluster_labels = perform_clustering(np.array(embeddings), n_clusters)

        # 步骤 4：创建当前层的 DataFrame
        df = pd.DataFrame({
            'text': current_texts,
            'embedding': embeddings,
            'cluster': cluster_labels,
            'metadata': current_metadata
        })

        # 存储当前层结果
        results[level-1] = df

        # 步骤 5：为每个簇生成摘要
        summaries = []
        new_metadata = []

        for cluster in df['cluster'].unique():
            # 获取该簇的所有文档
            cluster_docs = df[df['cluster'] == cluster]
            cluster_texts = cluster_docs['text'].tolist()
            cluster_metadata = cluster_docs['metadata'].tolist()

            # 生成摘要
            summary = summarize_texts(cluster_texts)
            summaries.append(summary)

            # 创建新元数据
            new_metadata.append({
                "level": level,
                "origin": f"summary_of_cluster_{cluster}_level_{level-1}",
                "child_ids": [meta.get('id') for meta in cluster_metadata],
                "id": f"summary_{level}_{cluster}"
            })

        # 更新当前文本和元数据（用于下一层）
        current_texts = summaries
        current_metadata = new_metadata

        # 如果只剩一个摘要，树构建完成
        if len(current_texts) <= 1:
            results[level] = pd.DataFrame({
                'text': current_texts,
                'embedding': embed_texts(current_texts),
                'cluster': [0],
                'metadata': current_metadata
            })
            logging.info(f"在第 {level} 层停止，因为只剩一个摘要")
            break

    return results
```

> **💡 代码解释**
>
> **树构建流程详解：**
>
> ```
> 第 0 层（原始文本）:
> [段落 1] [段落 2] [段落 3] [段落 4] [段落 5] [段落 6]
>     ↓ 嵌入 + 聚类（假设分成 2 簇）
> 簇 0: [段落 1] [段落 2] [段落 3]
> 簇 1: [段落 4] [段落 5] [段落 6]
>     ↓ 为每个簇生成摘要
> 第 1 层:
> [摘要 A: 总结段落 1-3] [摘要 B: 总结段落 4-6]
>     ↓ 再次嵌入 + 聚类
> 簇 0: [摘要 A] [摘要 B]
>     ↓ 生成摘要
> 第 2 层:
> [总摘要：总结 A 和 B]
>     ↓ 只剩一个，停止
> ```
>
> **元数据说明：**
> - `level`: 所在层级
> - `origin`: 来源描述
> - `child_ids`: 子节点 ID 列表（用于追溯）
> - `id`: 唯一标识符
>
> **⚠️ 新手注意**
> - `n_clusters = min(10, len//2)` 确保聚类数合理
> - 如果某层只有 1-2 个文本，聚类可能没有意义
> - 树的高度取决于文档数量和内容

---

## 🛠️ 第五步：构建向量存储

### 📖 这是什么？

将树中所有层级的内容整合到一个向量存储中，方便后续检索。

### 💻 完整代码

```python
def build_vectorstore(tree_results: Dict[int, pd.DataFrame]) -> FAISS:
    """
    从 RAPTOR 树中的所有文本构建 FAISS 向量存储

    参数:
        tree_results: build_raptor_tree 返回的树结构

    返回:
        FAISS 向量存储

    为什么要整合所有层？
        查询时可以从任意层级检索相关内容
        高层级提供概览，低层级提供细节
    """
    all_texts = []
    all_embeddings = []
    all_metadatas = []

    # 遍历每一层
    for level, df in tree_results.items():
        # 收集文本
        all_texts.extend([str(text) for text in df['text'].tolist()])

        # 收集嵌入（可能是 numpy 数组，需要转换）
        all_embeddings.extend([
            embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            for embedding in df['embedding'].tolist()
        ])

        # 收集元数据
        all_metadatas.extend(df['metadata'].tolist())

    logging.info(f"正在构建包含 {len(all_texts)} 个文本的向量存储")

    # 创建 Document 对象列表
    documents = [
        Document(page_content=str(text), metadata=metadata)
        for text, metadata in zip(all_texts, all_metadatas)
    ]

    # 从文档创建 FAISS 向量存储
    return FAISS.from_documents(documents, embeddings)
```

> **💡 代码解释**
>
> **FAISS 向量存储的作用：**
>
> ```
> 查询："温室效应是什么？"
>     ↓ 嵌入查询
> 查询向量
>     ↓ 相似度搜索
> 找到最相似的文档（来自任意层级）
>     ↓
> 可能结果:
> - Level 2: 气候变化的总体摘要
> - Level 1: 温室气体的章节摘要
> - Level 0: 具体解释温室效应的段落
> ```
>
> **⚠️ 新手注意**
> - 所有层级的内容都在同一个向量空间中
> - 查询可以匹配到任意层级的内容
> - 元数据中的 `level` 字段用于区分来源

---

## 🛠️ 第六步：创建检索器

### 📖 这是什么？

创建一个带有上下文压缩功能的检索器，可以提取最相关的信息片段。

### 💻 完整代码

```python
def create_retriever(vectorstore: FAISS) -> ContextualCompressionRetriever:
    """
    创建带有上下文压缩的检索器

    参数:
        vectorstore: FAISS 向量存储

    返回:
        ContextualCompressionRetriever

    什么是上下文压缩？
        1. 先检索较多文档（可能包含无关信息）
        2. 用 LLM 提取与查询最相关的部分
        3. 输出精简的上下文
    """
    logging.info("正在创建上下文压缩检索器")

    # 创建基础检索器
    base_retriever = vectorstore.as_retriever()

    # 创建提取提示
    prompt = ChatPromptTemplate.from_template(
        "给定以下上下文和问题，仅提取与回答问题相关的信息：\n\n"
        "上下文：{context}\n"
        "问题：{question}\n\n"
        "相关信息："
    )

    # 创建 LLM 链提取器
    extractor = LLMChainExtractor.from_llm(llm, prompt=prompt)

    # 组合成压缩检索器
    return ContextualCompressionRetriever(
        base_compressor=extractor,
        base_retriever=base_retriever
    )
```

> **💡 代码解释**
>
> **上下文压缩流程：**
>
> ```
> 用户查询 → 基础检索 → 10 个文档（可能冗余）
>                          ↓
>                    LLM 提取器
>                          ↓
>              3 个精炼的相关片段
> ```
>
> **为什么需要压缩？**
> - 减少 token 使用（省钱）
> - 提高答案质量（减少噪音）
> - 加快处理速度

---

## 🛠️ 第七步：层次化检索

### 📖 这是什么？

从树的最高层开始，逐层向下检索，充分利用树的层次结构。

### 💻 完整代码

```python
def hierarchical_retrieval(query: str, retriever: ContextualCompressionRetriever, max_level: int) -> List[Document]:
    """
    从最高层级开始执行层次化检索

    参数:
        query: 用户查询
        retriever: 上下文压缩检索器
        max_level: 树的最大层级

    返回:
        检索到的文档列表

    检索策略:
        1. 从最高层（最概括）开始检索
        2. 如果找到相关文档，继续检索其子文档
        3. 逐层向下，收集所有相关信息
    """
    all_retrieved_docs = []

    # 从最高层向最低层遍历
    for level in range(max_level, -1, -1):
        # 从当前层级检索文档
        level_docs = retriever.get_relevant_documents(
            query,
            filter=lambda meta: meta['level'] == level  # 只检索该层的文档
        )
        all_retrieved_docs.extend(level_docs)

        # 如果找到文档且还有下层，检索子文档
        if level_docs and level > 0:
            # 收集子文档 ID
            child_ids = [
                doc.metadata.get('child_ids', [])
                for doc in level_docs
            ]
            # 展平列表并过滤 None
            child_ids = [
                item for sublist in child_ids
                for item in sublist if item is not None
            ]

            # 如果有有效的子文档 ID，修改查询以检索它们
            if child_ids:
                child_query = f" AND id:({' OR '.join(str(id) for id in child_ids)})"
                query += child_query

    return all_retrieved_docs
```

> **💡 代码解释**
>
> **层次化检索示例：**
>
> ```
> 查询："气候变化对农业的影响"
>
> Level 2（最高层）:
> 检索到：[气候变化的总体影响摘要]
> 子文档：[影响农业], [影响经济], [影响生态]
>     ↓
> Level 1:
> 检索到：[气候对农业的影响], [气候变化与粮食安全]
> 子文档：[作物减产], [灌溉问题], ...
>     ↓
> Level 0:
> 检索到：[某地区小麦减产 30%], [干旱导致...], ...
> ```
>
> **⚠️ 新手注意**
> - 从高层开始可以快速定位相关区域
> - 子文档 ID 用于精确检索特定内容
> - 如果某层没有结果，仍会继续检索下层

---

## 🛠️ 第八步：RAPTOR 查询流程

### 📖 这是什么？

整合所有组件，实现完整的查询处理流程。

### 💻 完整代码

```python
def raptor_query(query: str, retriever: ContextualCompressionRetriever, max_level: int) -> Dict[str, Any]:
    """
    使用 RAPTOR 系统和层次化检索处理查询

    参数:
        query: 用户查询
        retriever: 上下文压缩检索器
        max_level: 树的最大层级

    返回:
        包含查询结果详情的字典

    流程:
        1. 层次化检索相关文档
        2. 整理文档详情
        3. 构建上下文字
        4. 生成答案
    """
    logging.info(f"正在处理查询：{query}")

    # 步骤 1：执行层次化检索
    relevant_docs = hierarchical_retrieval(query, retriever, max_level)

    # 步骤 2：整理文档详情
    doc_details = []
    for i, doc in enumerate(relevant_docs, 1):
        doc_details.append({
            "index": i,
            "content": doc.page_content,
            "metadata": doc.metadata,
            "level": doc.metadata.get('level', 'Unknown'),
            "similarity_score": doc.metadata.get('score', 'N/A')
        })

    # 步骤 3：构建上下文字
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # 步骤 4：生成答案
    prompt = ChatPromptTemplate.from_template(
        "给定以下上下文，请回答问题：\n\n"
        "上下文：{context}\n\n"
        "问题：{question}\n\n"
        "答案："
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    answer = chain.run(context=context, question=query)

    logging.info("查询处理完成")

    # 返回完整结果
    result = {
        "query": query,
        "retrieved_documents": doc_details,
        "num_docs_retrieved": len(relevant_docs),
        "context_used": context,
        "answer": answer,
        "model_used": llm.model_name,
    }

    return result


def print_query_details(result: Dict[str, Any]):
    """
    打印查询过程的详细信息

    参数:
        result: raptor_query 返回的结果字典
    """
    print(f"查询：{result['query']}")
    print(f"\n检索到的文档数量：{result['num_docs_retrieved']}")

    print(f"\n检索到的文档:")
    for doc in result['retrieved_documents']:
        print(f"  文档 {doc['index']}:")
        print(f"    内容：{doc['content'][:100]}...")  # 显示前 100 字符
        print(f"    相似度分数：{doc['similarity_score']}")
        print(f"    树层级：{doc['metadata'].get('level', 'Unknown')}")
        print(f"    来源：{doc['metadata'].get('origin', 'Unknown')}")
        if 'child_docs' in doc['metadata']:
            print(f"    子文档数量：{len(doc['metadata']['child_docs'])}")
        print()

    print(f"\n用于生成答案的上下文:")
    print(result['context_used'])

    print(f"\n生成的答案:")
    print(result['answer'])

    print(f"\n使用的模型：{result['model_used']}")
```

> **💡 代码解释**
>
> **完整查询流程：**
>
> ```
> 用户提问
>    ↓
> hierarchical_retrieval (层次化检索)
>    ↓
> 检索到 N 个相关文档（来自不同层级）
>    ↓
> 构建上下文字
>    ↓
> LLM 生成答案
>    ↓
> 返回答案和元数据
> ```
>
> **⚠️ 新手注意**
> - `enumerate(relevant_docs, 1)` 从 1 开始计数
> - `doc['content'][:100]` 只显示前 100 字符避免输出太长
> - 完整内容在 `result['context_used']` 中

---

## 🛠️ 第九步：完整使用示例

### 📖 这是什么？

用真实的 PDF 文档测试整个 RAPTOR 系统。

### 💻 完整代码

```python
from langchain.document_loaders import PyPDFLoader

# 下载示例数据
os.makedirs('data', exist_ok=True)

# 下载气候变化的 PDF 文档
!wget -O data/Understanding_Climate_Change.pdf \
    https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf

# 指定文件路径
path = "data/Understanding_Climate_Change.pdf"

# 加载 PDF
loader = PyPDFLoader(path)
documents = loader.load()

# 提取文本内容
texts = [doc.page_content for doc in documents]
print(f"加载了 {len(texts)} 页文档")

# ============ 构建 RAPTOR 树 ============
print("\n开始构建 RAPTOR 树...")
tree_results = build_raptor_tree(texts, max_levels=3)
print(f"✓ 树构建完成，共 {len(tree_results)} 层")

# ============ 构建向量存储 ============
print("\n构建向量存储...")
vectorstore = build_vectorstore(tree_results)
print("✓ 向量存储构建完成")

# ============ 创建检索器 ============
print("\n创建检索器...")
retriever = create_retriever(vectorstore)
print("✓ 检索器创建完成")

# ============ 运行查询 ============
max_level = 3  # 根据实际树高度调整
query = "温室效应是什么？"
print(f"\n运行查询：{query}")

result = raptor_query(query, retriever, max_level)
print_query_details(result)
```

> **💡 代码解释**
>
> **执行流程：**
>
> ```
> 1. 加载 PDF → 提取文本
> 2. build_raptor_tree → 创建层次化摘要
> 3. build_vectorstore → 存储所有层级内容
> 4. create_retriever → 创建智能检索器
> 5. raptor_query → 处理用户查询
> ```
>
> **⚠️ 新手注意**
> - 树构建会调用多次 LLM，需要几分钟时间
> - 会产生一定的 API 费用
> - 建议先用少量文档测试
>
> **📊 预期输出示例：**
>
> ```
> 查询：温室效应是什么？
>
> 检索到的文档数量：5
>
> 检索到的文档:
>   文档 1:
>     内容：温室效应是指大气中的温室气体吸收和重新辐射红外辐射，导致地球表面...
>     相似度分数：0.85
>     树层级：1
>     来源：summary_of_cluster_0_level_0
>
>   文档 2:
>     内容：主要的温室气体包括二氧化碳、甲烷、水蒸气等...
>     相似度分数：0.78
>     树层级：0
>     来源：original
>
> ...
>
> 生成的答案:
> 温室效应是指大气中的温室气体（如二氧化碳、甲烷、水蒸气等）吸收地球表面
> 发出的红外辐射，并将部分辐射重新辐射回地表，从而使地球温度升高的自然
> 现象。这个过程类似于温室的玻璃效应，因此得名...
> ```

---

## 📚 RAPTOR 的优势总结

| 优势 | 说明 | 实际价值 |
|------|------|---------|
| **可扩展性** | 通过多层摘要处理大型文档集 | 能处理数百页的文档 |
| **灵活性** | 能提供高层概览和具体细节 | 适应不同类型的问题 |
| **上下文感知** | 从最合适的抽象层级检索 | 答案更准确、更有针对性 |
| **高效性** | 向量存储支持快速检索 | 响应时间短 |
| **可追溯性** | 维护摘要与原文的链接 | 可以验证信息来源 |

---

## 🎓 术语解释表

| 术语 | 英文 | 解释 |
|------|------|------|
| 递归 | Recursive | 重复应用相同过程 |
| 抽象 | Abstractive | 生成概括性描述（而非简单摘录） |
| 主题组织 | Thematic Organization | 按主题分组相关内容 |
| 高斯混合模型 | Gaussian Mixture Model | 一种概率聚类算法 |
| 主成分分析 | PCA | 降维技术，用于可视化 |
| 上下文压缩 | Contextual Compression | 用 LLM 提取最相关信息 |
| 层次化检索 | Hierarchical Retrieval | 从树的多层检索信息 |

---

## ❓ 常见问题 FAQ

### Q1: RAPTOR 和普通 RAG 有什么区别？

**A:** 主要区别：

| 特性 | 普通 RAG | RAPTOR |
|------|---------|--------|
| 组织结构 | 扁平的文档块 | 树形层次结构 |
| 检索方式 | 单一层级相似度 | 多层级层次化检索 |
| 摘要能力 | 无 | 有（每层都有摘要） |
| 适用问题 | 具体细节问题 | 宏观和细节问题 |

### Q2: 树应该建多少层？

**A:** 取决于文档数量：
- 少量文档（<10 页）：2 层足够
- 中等文档（10-50 页）：3 层
- 大量文档（>50 页）：4 层或更多

经验法则：树的高度应该是 `log₂(文档数)` 左右。

### Q3: 聚类效果不好怎么办？

**A:** 可以尝试：
1. 调整聚类数（增加或减少）
2. 更换聚类算法（如 K-Means）
3. 调整文本块大小
4. 使用不同的嵌入模型

### Q4: 可以用中文文档吗？

**A:** 可以！OpenAI 的嵌入模型支持多语言，包括中文。只需：
- 确保文档是 UTF-8 编码
- 提示词可以保持英文（GPT 支持多语言输入）

---

## ✅ 学习检查清单

- [ ] 我理解了 RAPTOR 树的基本结构
- [ ] 我知道聚类在树构建中的作用
- [ ] 我理解了层次化检索的原理
- [ ] 我能解释上下文压缩的好处
- [ ] 我能创建自己的 RAPTOR 系统
- [ ] 我知道如何调整树的层数

---

## 🚀 下一步学习建议

1. **尝试不同文档**：用 PDF、TXT 等多种格式测试
2. **可视化聚类**：调用 `visualize_clusters` 观察聚类效果
3. **比较不同层级的答案**：对比只从单层检索的结果差异
4. **学习 Self-RAG**：继续学习下一个教程，了解动态检索机制

---

> **💪 恭喜！** 你已经完成了 RAPTOR 的新手教程！现在你掌握了如何用树形结构组织文档，实现多层次智能检索。这是构建高级 RAG 系统的重要技能！
