# 🌟 新手入门：飞镖板检索（Dartboard RAG）

> **💡 给新手的说明**
> - **难度等级**：⭐⭐⭐⭐（中高级，需要线性代数和概率基础）
> - **预计学习时间**：60-90 分钟
> - **前置知识**：了解向量相似度、余弦相似度、基础 NumPy 操作
> - **本教程特色**：包含数学原理通俗解释、完整代码注释、可视化示例
>
> **📚 什么是飞镖板检索？** 想象你在玩飞镖游戏。传统方法只选离靶心最近的几支飞镖，但这些飞镖可能都扎在同一个位置。飞镖板方法则要求：既要离靶心近（相关性），又要分散在不同位置（多样性）。这样才能覆盖更大的得分区域！

---

## 📖 核心概念理解

### 通俗理解：相关性和多样性的平衡

**问题场景**：
假设你在写论文，需要搜索"气候变化的影响"。传统检索返回的前 5 篇文章可能都是同一作者、同一观点、同一数据——因为它们的内容太相似了！

```
❌ 传统检索的问题：
查询："气候变化的影响"
结果 1：全球变暖导致海平面上升（来自 A 论文）
结果 2：全球变暖导致海平面上升（来自 B 论文）
结果 3：全球变暖导致海平面上升（来自 C 论文）
结果 4：全球变暖导致海平面上升（来自 D 论文）
结果 5：全球变暖导致海平面上升（来自 E 论文）
→ 信息冗余！没有获得更全面的视角
```

**飞镖板检索的解决方案**：
```
✅ 飞镖板检索的结果：
查询："气候变化的影响"
结果 1：全球变暖导致海平面上升（相关性最高）
结果 2：极端天气事件增多（不同角度）
结果 3：生态系统受到威胁（另一个角度）
结果 4：农业生产受影响（又一个角度）
结果 5：人类健康风险增加（再一个角度）
→ 既相关又多样！全面覆盖主题
```

### 生活化比喻

| 场景 | 传统方法 | 飞镖板方法 |
|------|---------|-----------|
| 🍽️ **点菜** | 点了 5 道相似的菜（都是辣的） | 点 5 道不同口味、营养均衡的菜 |
| 📰 **看新闻** | 看 5 篇相同立场的报道 | 看来自不同媒体、不同视角的报道 |
| 🎵 **听歌** | 听 5 首同一歌手的歌 | 听 5 首不同歌手但都符合你口味的歌 |
| 🛍️ **购物** | 去 5 家卖同样商品的店 | 去 5 家有特色、商品不同的店 |

### 核心术语解释

| 术语 | 通俗解释 | 数学含义 |
|------|----------|----------|
| **相关性（Relevance）** | 文档与查询的匹配程度 | 查询向量与文档向量的相似度 |
| **多样性（Diversity）** | 文档之间的差异程度 | 文档向量之间的距离 |
| **余弦相似度** | 两个向量方向的接近程度 | cos(θ) = A·B / (‖A‖‖B‖) |
| **距离矩阵** | 所有文档两两之间的距离表格 | n×n 的矩阵，每个元素是距离 |
| **贪心算法** | 每一步都选当前最优解 | 逐步构建，不回溯 |

---

## 🛠️ 第一步：安装必要的包

### 📖 这是什么？
飞镖板检索需要的核心工具：
- `numpy`：数值计算库，用于矩阵运算
- `python-dotenv`：管理 API 密钥

### 💻 完整代码

```python
# 安装所需的包
# ⚠️ numpy 是科学计算的基础包，强烈建议安装
!pip install numpy python-dotenv
```

```python
# 克隆仓库以访问辅助函数
!git clone https://github.com/NirDiamant/RAG_TECHNIQUES.git
import sys
sys.path.append('RAG_TECHNIQUES')
```

```python
# 导入必要的库
import os
import sys
from dotenv import load_dotenv
from scipy.special import logsumexp  # 用于概率归一化
from typing import Tuple, List, Any  # 类型提示
import numpy as np  # 数值计算核心库

# 从.env 文件加载环境变量
load_dotenv()

# 设置 OpenAI API 密钥
if not os.getenv('OPENAI_API_KEY'):
    print("请输入你的 OpenAI API 密钥：")
    os.environ["OPENAI_API_KEY"] = input("请输入 OpenAI API 密钥：")
else:
    os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# 导入辅助函数
from helper_functions import *
from evaluation.evalute_rag import *
```

> **💡 代码解释**
> - `scipy.special.logsumexp`：用于数值稳定的概率计算
> - `typing` 模块：提供类型提示，让代码更易读
> - `numpy as np`：NumPy 是 Python 科学计算的标准库
>
> **⚠️ 新手注意**
> - 如果不想输入 API 密钥，确保 `.env` 文件中已配置
> - `scipy` 可能安装较慢，请耐心等待

### ❓ 常见问题

**Q1: 为什么要用 NumPy 而不是普通列表？**
```
NumPy 的优势：
1. 速度快：底层用 C 实现，比 Python 列表快几十倍
2. 功能多：矩阵运算、线性代数、统计等
3. 语法简洁：一行代码可以替代多层循环

例如计算余弦相似度：
# 普通 Python（慢且复杂）
def cosine_sim(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = sum(x*x for x in a) ** 0.5
    norm_b = sum(x*x for x in b) ** 0.5
    return dot / (norm_a * norm_b)

# NumPy（快且简洁）
np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

---

## 🛠️ 第二步：准备和编码文档

### 📖 这是什么？
我们需要一个文档集来演示。这里故意将文档重复 5 次，模拟真实场景中常见的"密集数据集"（很多相似文档）。

### 💻 完整代码

```python
# 创建 data 目录并下载示例 PDF
import os
os.makedirs('data', exist_ok=True)

# 下载示例 PDF 文档
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
```

```python
# 定义 PDF 路径
path = "data/Understanding_Climate_Change.pdf"
```

```python
def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """
    使用 OpenAI embeddings 将 PDF 书籍编码为向量存储。

    Args:
        path: PDF 文件的路径。
        chunk_size: 每个文本块的期望大小（默认 1000 字符）。
        chunk_overlap: 连续块之间的重叠量（默认 200 字符）。

    Returns:
        FAISS 向量存储，包含编码后的文档内容。
    """

    # ==================== 步骤 1: 加载 PDF 文档 ====================
    loader = PyPDFLoader(path)
    documents = loader.load()

    # ==================== 步骤 2: 模拟密集数据集 ====================
    # ⚠️ 关键技巧：将每个文档加载 5 次，模拟真实场景中的重复内容
    # 这是为了演示飞镖板检索的价值——在密集数据集中去重
    documents = documents * 5

    # ==================== 步骤 3: 分割文档 ====================
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)

    # 清理文本（替换制表符为空格）
    cleaned_texts = replace_t_with_space(texts)

    # ==================== 步骤 4: 创建 Embeddings ====================
    # 可以使用 OpenAI 或 Amazon Bedrock
    embeddings = get_langchain_embedding_provider(EmbeddingProvider.OPENAI)
    # embeddings = get_langchain_embedding_provider(EmbeddingProvider.AMAZON_BEDROCK)

    # ==================== 步骤 5: 创建向量存储 ====================
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore
```

```python
# 执行编码
chunks_vector_store = encode_pdf(path, chunk_size=1000, chunk_overlap=200)

print(f"向量存储创建完成！")
print(f"文档总数：{chunks_vector_store.index.ntotal}")
```

> **💡 代码解释**
> - `documents * 5`：这是故意制造冗余，让传统检索的缺陷更明显
> - `replace_t_with_space`：清理文本中的特殊字符
> - `EmbeddingProvider.OPENAI`：枚举类型，指定使用 OpenAI 的嵌入模型
>
> **⚠️ 新手注意**
> - 如果你有自己的文档，可以替换 PDF 路径
> - 密集数据集是故意设置的，实际使用不需要 `* 5`

---

## 🛠️ 第三步：理解传统检索的问题

### 📖 这是什么？
让我们先看看传统 top-k 检索在密集数据集中会遇到什么问题。

### 💻 完整代码

```python
# 辅助函数：将向量存储索引转换为文本
def idx_to_text(idx: int):
    """
    将向量存储索引转换为相应的文本内容。

    Args:
        idx: 向量存储中的索引。

    Returns:
        对应的文档文本内容。
    """
    # 获取文档存储 ID
    docstore_id = chunks_vector_store.index_to_docstore_id[idx]
    # 从文档存储中搜索
    document = chunks_vector_store.docstore.search(docstore_id)
    return document.page_content


def get_context(query: str, k: int = 5) -> List[str]:
    """
    使用传统 top-k 方法为查询检索前 k 个上下文项。

    Args:
        query: 搜索查询。
        k: 返回的结果数量（默认 5）。

    Returns:
        检索到的文本列表。
    """
    # 将查询转换为向量
    q_vec = chunks_vector_store.embedding_function.embed_documents([query])

    # 在向量存储中搜索最相似的 k 个结果
    _, indices = chunks_vector_store.index.search(np.array(q_vec), k=k)

    # 将索引转换为文本
    texts = [idx_to_text(i) for i in indices[0]]
    return texts
```

```python
# 测试查询
test_query = "What is the main cause of climate change?"

# 使用传统方法检索
print("🔍 使用传统 Top-K 检索:")
print("=" * 50)
texts = get_context(test_query, k=3)
show_context(texts)
```

> **💡 预期输出**
> ```
> 🔍 使用传统 Top-K 检索:
> ==================================================
> 结果 1：[某段关于气候变化的文本]
> 结果 2：[几乎相同的文本，只是来自不同副本]
> 结果 3：[还是差不多的文本...]
>
> ❌ 问题：3 个结果都是同一文档的重复！信息冗余严重。
> ```

### ⚠️ 新手注意

**这就是问题所在！** 当数据库中有大量相似文档时，传统检索返回的 top-k 结果可能都是"换汤不换药"的重复内容。飞镖板检索就是要解决这个问题。

---

## 🛠️ 第四步：理解距离和概率转换

### 📖 这是什么？
飞镖板算法需要将距离转换为概率。这里涉及一些数学知识，别担心，我会用通俗的方式解释！

### 💡 数学原理通俗解释

**距离转概率的意义**：
```
想象你在射击：
- 距离靶心越近 → 命中的概率越高
- 距离靶心越远 → 命中的概率越低

对数正态分布就是一种"距离→概率"的转换公式：
- 距离 = 0 → 概率最高
- 距离增大 → 概率快速下降
```

### 💻 完整代码

```python
def lognorm(dist: np.ndarray, sigma: float):
    """
    计算给定距离和 sigma 的对数正态概率。

    Args:
        dist: 距离数组（可以是标量、向量或矩阵）。
        sigma: 平滑参数，控制概率分布的"宽窄"。
               sigma 越小，概率随距离下降越快。

    Returns:
        对数概率值。值越大表示概率越高。

    📐 数学公式：
    log_prob = -log(sigma) - 0.5*log(2π) - dist²/(2*sigma²)

    这个公式保证：
    - 距离为 0 时，概率最高
    - 距离越大，概率越低
    - sigma 控制下降速度
    """
    # 避免除以零的错误
    if sigma < 1e-9:
        return -np.inf * dist

    # 计算对数正态概率
    return -np.log(sigma) - 0.5 * np.log(2 * np.pi) - dist**2 / (2 * sigma**2)
```

> **💡 代码解释**
>
> **参数 sigma 的作用**：
> ```python
> sigma = 0.1  # 小 sigma：概率随距离快速下降（"严格"）
> sigma = 1.0  # 大 sigma：概率随距离缓慢下降（"宽松"）
> ```
>
> **为什么用对数概率？**
> - 概率是小数（0 到 1），相乘会越来越小
> - 对数概率是负数，相加更稳定
> - 最后比较大小，不影响结果

### ❓ 常见问题

**Q1: 为什么要用对数正态分布？**
```
原因：
1. 距离为 0 时概率最高（符合直觉）
2. 距离增大时概率平滑下降（避免突变）
3. 数学性质好（便于计算和优化）

其他选择也可以，比如高斯分布、指数分布等。
```

**Q2: sigma 怎么选？**
```
经验值：
- sigma = 0.1：适合精细区分（距离差异很重要）
- sigma = 0.5：平衡选择（推荐默认值）
- sigma = 1.0：适合粗略区分（距离差异不太重要）

可以根据你的数据集调整，观察效果变化。
```

---

## 🛠️ 第五步：贪心飞镖板搜索算法

### 📖 这是什么？
**这是核心算法！** 飞镖板搜索通过贪心策略，逐步选择既相关又多样的文档。

### 💡 算法原理图解

```
假设我们要选 3 个文档：

初始状态：
查询 Q → [文档 A, 文档 B, 文档 C, 文档 D, 文档 E]
         (每个文档与 Q 有相关性分数)

第 1 步：选最相关的
✅ 选中：文档 A（与 Q 最相似）

第 2 步：综合考虑相关性和多样性
- 文档 B：与 Q 很相关，但与 A 太相似 → 扣分！
- 文档 C：与 Q 较相关，且与 A 很不同 → 加分！
✅ 选中：文档 C

第 3 步：继续综合考虑
- 文档 B：与 Q 相关，与 A 相似，与 C 相似 → 扣分！
- 文档 D：与 Q 较相关，与 A、C 都不同 → 加分！
- 文档 E：与 Q 不太相关 → 扣分！
✅ 选中：文档 D

最终结果：[A, C, D] —— 既相关又多样！
```

### 💻 完整代码

```python
# ==================== 配置参数 ====================
# 这些参数可以调整，观察不同效果
DIVERSITY_WEIGHT = 1.0    # 多样性权重：越大越重视多样性
RELEVANCE_WEIGHT = 1.0    # 相关性权重：越大越重视相关性
SIGMA = 0.1               # 平滑参数：控制概率分布的宽窄


def greedy_dartsearch(
    query_distances: np.ndarray,
    document_distances: np.ndarray,
    documents: List[str],
    num_results: int
) -> Tuple[List[str], List[float]]:
    """
    执行贪心飞镖板搜索，选择平衡相关性和多样性的文档。

    Args:
        query_distances: 查询与每个文档之间的距离（1 维数组）。
                        距离越小越相关。
        document_distances: 文档之间的成对距离（2 维矩阵）。
                           distances[i][j] = 文档 i 和文档 j 的距离。
        documents: 文档文本列表。
        num_results: 要返回的文档数量。

    Returns:
        元组 (selected_documents, selection_scores)：
        - selected_documents: 选中的文档文本列表
        - selection_scores: 每个文档的选择分数

    🎯 算法核心思想：
    1. 先选最相关的文档
    2. 后续选择时，综合考虑：
       - 与查询的相关性
       - 与已选文档的差异性（多样性）
    3. 重复直到选够数量
    """

    # ==================== 准备工作 ====================
    # 确保 sigma 不为零（避免除以零错误）
    sigma = max(SIGMA, 1e-5)

    # 将距离转换为概率
    # query_probabilities[i] = 查询与文档 i 的"相关概率"
    query_probabilities = lognorm(query_distances, sigma)

    # document_probabilities[i][j] = 文档 i 与文档 j 的"差异概率"
    document_probabilities = lognorm(document_distances, sigma)

    # ==================== 第 1 步：选择最相关的文档 ====================
    # 找到与查询最相关的文档索引
    most_relevant_idx = np.argmax(query_probabilities)

    # 初始化选中的文档列表
    selected_indices = np.array([most_relevant_idx])
    selection_scores = [1.0]  # 第一个文档的分数设为 1（虚拟值）

    # 记录从第一个文档到所有其他文档的距离
    max_distances = document_probabilities[most_relevant_idx]

    # ==================== 第 2 步：迭代选择剩余文档 ====================
    while len(selected_indices) < num_results:

        # --- 更新最大距离 ---
        # 对于每个未选文档，计算它到所有已选文档的最大距离
        # 这代表它与已选集合的"差异性"
        updated_distances = np.maximum(max_distances, document_probabilities)

        # --- 计算综合分数 ---
        # 综合分数 = 多样性分数 + 相关性分数
        combined_scores = (
            updated_distances * DIVERSITY_WEIGHT +    # 多样性部分
            query_probabilities * RELEVANCE_WEIGHT     # 相关性部分
        )

        # --- 归一化分数 ---
        # 使用 logsumexp 进行数值稳定的归一化
        normalized_scores = logsumexp(combined_scores, axis=1)

        # 屏蔽已选文档（设为负无穷，确保不会再被选中）
        normalized_scores[selected_indices] = -np.inf

        # --- 选择最佳文档 ---
        best_idx = np.argmax(normalized_scores)
        best_score = np.max(normalized_scores)

        # --- 更新状态 ---
        max_distances = updated_distances[best_idx]
        selected_indices = np.append(selected_indices, best_idx)
        selection_scores.append(best_score)

    # ==================== 返回结果 ====================
    # 将索引转换为实际文档文本
    selected_documents = [documents[i] for i in selected_indices]
    return selected_documents, selection_scores
```

> **💡 代码解释**
>
> **关键步骤详解**：
>
> 1. **距离转概率**（第 1 步）
>    ```python
>    # 距离 → 概率
>    # 距离 0 → 概率 1（最相关/最不同）
>    # 距离大 → 概率小
>    query_probabilities = lognorm(query_distances, sigma)
>    ```
>
> 2. **初始化**（第 2 步）
>    ```python
>    # 选最相关的作为起点
>    most_relevant_idx = np.argmax(query_probabilities)
>    ```
>
> 3. **迭代选择**（第 3 步）
>    ```python
>    # 综合分数 = 多样性 + 相关性
>    combined_scores = (
>        updated_distances * DIVERSITY_WEIGHT +
>        query_probabilities * RELEVANCE_WEIGHT
>    )
>    ```
>
> **⚠️ 新手注意**
> - `np.maximum`：逐元素取最大值，用于更新距离
> - `logsumexp`：数值稳定的归一化，避免溢出
> - `selected_indices`：记录已选文档，避免重复选择

### ❓ 常见问题

**Q1: 为什么第一个文档只看相关性？**
```
因为还没有"已选文档"，无法计算多样性。
第一个文档作为"锚点"，后续选择都参考它。
```

**Q2: DIVERSITY_WEIGHT 和 RELEVANCE_WEIGHT 怎么调？**
```
推荐配置：
- 重视相关性：RELEVANCE_WEIGHT=2.0, DIVERSITY_WEIGHT=0.5
- 重视多样性：RELEVANCE_WEIGHT=0.5, DIVERSITY_WEIGHT=2.0
- 平衡：两者都设为 1.0（默认）

根据应用场景调整！
```

**Q3: 贪心算法一定能找到最优解吗？**
```
不一定。贪心算法每一步都选当前最优，但整体可能不是最优。
不过优点是：
1. 速度快（不用穷举所有组合）
2. 效果通常不错（实际验证）
3. 易于理解和实现
```

---

## 🛠️ 第六步：整合飞镖板检索函数

### 📖 这是什么？
这是最终用户使用的接口函数，封装了所有底层复杂性。

### 💻 完整代码

```python
def get_context_with_dartboard(
    query: str,
    num_results: int = 5,
    oversampling_factor: int = 3
) -> Tuple[List[str], List[float]]:
    """
    使用飞镖板算法为查询检索最相关且多样的上下文。

    Args:
        query: 搜索查询字符串。
        num_results: 要返回的结果数量（默认 5）。
        oversampling_factor: 过采样因子（默认 3）。
                           初始检索会获取 num_results * oversampling_factor
                           个候选，以便有更多选择空间。

    Returns:
        元组 (selected_texts, selection_scores)：
        - selected_texts: 选中的上下文文本列表
        - selection_scores: 每个文本的选择分数

    🎯 工作流程：
    1. 获取候选文档（过采样）
    2. 计算距离矩阵
    3. 运行飞镖板算法
    4. 返回最终结果
    """

    # ==================== 步骤 1: 获取候选文档 ====================
    # 嵌入查询
    query_embedding = chunks_vector_store.embedding_function.embed_documents([query])

    # 检索候选（数量 = num_results * oversampling_factor）
    # 例如：要 5 个结果，先检索 15 个候选
    _, candidate_indices = chunks_vector_store.index.search(
        np.array(query_embedding),
        k=num_results * oversampling_factor
    )

    # ==================== 步骤 2: 准备数据 ====================
    # 获取候选文档的向量
    candidate_vectors = np.array(
        chunks_vector_store.index.reconstruct_batch(candidate_indices[0])
    )

    # 获取候选文档的文本
    candidate_texts = [idx_to_text(idx) for idx in candidate_indices[0]]

    # ==================== 步骤 3: 计算距离矩阵 ====================
    # 使用 1 - 余弦相似度作为距离
    # 余弦相似度 = 1 表示完全相同，0 表示完全无关
    # 距离 = 1 - 相似度，所以距离 0 表示相同，1 表示完全不同

    # 文档之间的距离（候选集合内部）
    document_distances = 1 - np.dot(candidate_vectors, candidate_vectors.T)

    # 查询与每个候选文档的距离
    query_distances = 1 - np.dot(query_embedding, candidate_vectors.T)

    # ==================== 步骤 4: 运行飞镖板算法 ====================
    selected_texts, selection_scores = greedy_dartsearch(
        query_distances,           # 查询到各文档的距离
        document_distances,        # 文档之间的距离
        candidate_texts,           # 候选文档文本
        num_results                # 要选多少个
    )

    return selected_texts, selection_scores
```

> **💡 代码解释**
>
> **为什么要过采样（Oversampling）？**
> ```
> 假设最终要 5 个文档：
>
> ❌ 不过采样：只检索 5 个候选，然后全都要
>    → 没有选择空间，飞镖板算法无用武之地
>
> ✅ 过采样：检索 15 个候选，从中选最好的 5 个
>    → 飞镖板算法可以充分权衡相关性和多样性
>
> oversampling_factor 越大：
> - 优点：选择空间大，结果质量可能更高
> - 缺点：计算量大，耗时长
>
> 推荐值：3（平衡速度和质量）
> ```
>
> **距离计算原理**：
> ```python
> # 余弦相似度
> cosine_sim(a, b) = (a · b) / (||a|| × ||b||)
>
> # 转换为距离
> distance = 1 - cosine_sim
>
> # 所以：
> # distance = 0 → 完全相同
> # distance = 1 → 完全不同
> ```

### ❓ 常见问题

**Q1: oversampling_factor 应该设多少？**
```
建议：
- 小数据集（<100 文档）：2-3
- 中等数据集：3-5
- 大数据集（>1000 文档）：5-10

可以根据效果调整，找到速度和质量的最佳平衡点。
```

**Q2: 返回的 scores 有什么用？**
```
scores 表示每个文档被选中的"信心度"：
- 高分：这个文档明显很好
- 低分：这个文档是"矮子里拔将军"

可以用来：
1. 过滤低质量结果（设置阈值）
2. 对结果排序展示
3. 诊断算法表现
```

---

## 🛠️ 第七步：测试飞镖板检索效果

### 📖 这是什么？
让我们用同样的查询，对比传统检索和飞镖板检索的效果！

### 💻 完整代码

```python
# 使用飞镖板检索
print("🎯 使用飞镖板检索:")
print("=" * 50)
texts, scores = get_context_with_dartboard(test_query, num_results=3)

# 打印结果和分数
for i, (text, score) in enumerate(zip(texts, scores), 1):
    print(f"\n【结果{i}】(分数：{score:.2f})")
    print(f"内容：{text[:150]}...")
    print("-" * 50)

# 对比传统方法
print("\n\n📊 对比分析:")
print("=" * 50)
print("传统 Top-K：3 个结果都是重复内容 ❌")
print("飞镖板检索：3 个结果各有侧重 ✅")
```

> **💡 预期输出**
> ```
> 🎯 使用飞镖板检索:
> ==================================================
>
> 【结果 1】(分数：1.00)
> 内容：气候变化的主要原因是人类活动，特别是化石燃料的燃烧...
> --------------------------------------------------
>
> 【结果 2】(分数：0.85)
> 内容：森林砍伐和土地利用变化也显著影响气候系统...
> --------------------------------------------------
>
> 【结果 3】(分数：0.72)
> 内容：工业过程释放的温室气体加剧了全球变暖...
> --------------------------------------------------
>
>
> 📊 对比分析:
> ==================================================
> 传统 Top-K：3 个结果都是重复内容 ❌
> 飞镖板检索：3 个结果各有侧重 ✅
> ```

---

## 📊 可视化理解

### 飞镖板算法工作流程图

```
┌─────────────────────────────────────────────────────────────┐
│                   飞镖板检索流程                             │
└─────────────────────────────────────────────────────────────┘

                    用户查询
                      │
                      ▼
        ┌─────────────────────────┐
        │  过采样检索候选文档      │
        │  (例如：15 个候选)       │
        └─────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────┐
        │  计算距离矩阵           │
        │  - 查询到各文档的距离    │
        │  - 文档之间的距离        │
        └─────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────┐
        │  飞镖板贪心搜索         │
        │                         │
        │  第 1 轮：选最相关的 A   │
        │  第 2 轮：A 相关 + 与 A 不同 → 选 B
        │  第 3 轮：A/B 相关 + 与 A/B 不同 → 选 C
        │  ...                    │
        └─────────────────────────┘
                      │
                      ▼
              返回最终结果
         [A, B, C, ...] —— 既相关又多样！
```

### 相关性与多样性的权衡

```
假设查询是"苹果"，已有文档 A（水果苹果）

候选文档对比：
┌──────────┬────────────┬────────────┬──────────┐
│  文档    │ 与查询相关 │ 与 A 多样  │ 综合得分 │
├──────────┼────────────┼────────────┼──────────┤
│ B(苹果)  │   ⭐⭐⭐⭐⭐  │    ⭐      │   ⭐⭐    │
│ C(香蕉)  │   ⭐⭐⭐    │    ⭐⭐⭐⭐⭐  │   ⭐⭐⭐⭐  │
│ D(果园)  │   ⭐⭐⭐⭐  │    ⭐⭐⭐    │   ⭐⭐⭐⭐  │
│ E(手机)  │   ⭐⭐     │    ⭐⭐⭐⭐⭐  │   ⭐⭐    │
└──────────┴────────────┴────────────┴──────────┘

结果：C 或 D 可能比 B 得分更高，因为它们提供了多样性！
```

---

## 🎯 参数调优指南

### 核心参数及其影响

| 参数 | 作用 | 调大效果 | 调小效果 | 推荐值 |
|------|------|---------|---------|--------|
| **DIVERSITY_WEIGHT** | 多样性权重 | 结果更多样，可能降低相关性 | 结果更相似，相关性可能更高 | 1.0 |
| **RELEVANCE_WEIGHT** | 相关性权重 | 结果更相关，可能降低多样性 | 结果更多样，相关性可能降低 | 1.0 |
| **SIGMA** | 概率平滑 | 概率分布更平缓 | 概率分布更尖锐 | 0.1 |
| **oversampling_factor** | 过采样倍数 | 选择空间大，计算慢 | 选择空间小，计算快 | 3 |

### 不同场景的配置建议

```python
# 场景 1: 学术文献检索（需要全面覆盖）
DIVERSITY_WEIGHT = 1.5
RELEVANCE_WEIGHT = 1.0
oversampling_factor = 5

# 场景 2: 事实核查（准确性优先）
DIVERSITY_WEIGHT = 0.5
RELEVANCE_WEIGHT = 2.0
oversampling_factor = 3

# 场景 3: 创意写作辅助（需要多样性）
DIVERSITY_WEIGHT = 2.0
RELEVANCE_WEIGHT = 0.5
oversampling_factor = 5

# 场景 4: 通用场景（平衡配置）
DIVERSITY_WEIGHT = 1.0
RELEVANCE_WEIGHT = 1.0
oversampling_factor = 3
```

---

## ⚠️ 避坑指南

### 常见错误及解决方法

**错误 1: 矩阵维度不匹配**
```
错误信息：ValueError: shapes (1,128) and (15,) not aligned
原因：query_embedding 和 candidate_vectors 维度不匹配
解决：
# 确保 query_embedding 是二维数组
query_embedding = np.array(query_embedding).reshape(1, -1)
```

**错误 2: 距离矩阵出现负值**
```
原因：余弦相似度可能略大于 1（浮点误差）
解决：
# 限制相似度在 [0, 1] 范围
cosine_sim = np.clip(cosine_sim, 0, 1)
document_distances = 1 - cosine_sim
```

**错误 3: 选中的文档太少**
```
原因：候选数量不足
解决：
# 增加 oversampling_factor
get_context_with_dartboard(query, num_results=5, oversampling_factor=5)
```

**错误 4: 结果仍然重复**
```
原因：多样性权重太低
解决：
# 增加 DIVERSITY_WEIGHT
DIVERSITY_WEIGHT = 2.0
```

---

## ❓ 新手常见问题

### Q1: 飞镖板检索一定比传统检索好吗？

**答**：不一定。适用场景：
- ✅ **适合**：文档集大、内容重叠多、需要全面信息
- ❌ **不适合**：文档集小、内容差异大、追求极致速度

### Q2: 可以只使用相关性或只使用多样性吗？

**答**：可以！通过调整权重：
```python
# 只看相关性（退化为传统检索）
DIVERSITY_WEIGHT = 0
RELEVANCE_WEIGHT = 1.0

# 只看多样性（不考虑相关性）
DIVERSITY_WEIGHT = 1.0
RELEVANCE_WEIGHT = 0
```

### Q3: 如何评估飞镖板检索的效果？

**答**：可以用以下指标：
- **NDCG**：衡量排序质量
- **多样性分数**：结果之间的平均距离
- **用户满意度**：最终用户评价

---

## 🎓 进阶：与混合检索结合

飞镖板算法可以和混合检索（稠密 + 稀疏）结合：

```python
# 概念示例
def hybrid_dartboard_search(query, dense_index, sparse_index, alpha=0.5):
    """
    结合稠密和稀疏检索的飞镖板搜索。

    Args:
        query: 查询文本
        dense_index: 稠密向量索引（语义相似度）
        sparse_index: 稀疏索引（BM25 等关键词匹配）
        alpha: 稠密权重（0-1）

    工作流程：
    1. 分别获取稠密和稀疏相似度
    2. 加权融合：final_sim = alpha * dense_sim + (1-alpha) * sparse_sim
    3. 转换为距离：distance = 1 - final_sim
    4. 运行飞镖板算法
    """
    pass  # 具体实现可根据需求扩展
```

---

## 📝 实战练习

### 练习 1: 调整参数观察效果

```python
# 尝试不同的权重配置
configs = [
    {"DIVERSITY_WEIGHT": 0.5, "RELEVANCE_WEIGHT": 2.0},
    {"DIVERSITY_WEIGHT": 1.0, "RELEVANCE_WEIGHT": 1.0},
    {"DIVERSITY_WEIGHT": 2.0, "RELEVANCE_WEIGHT": 0.5},
]

for config in configs:
    DIVERSITY_WEIGHT = config["DIVERSITY_WEIGHT"]
    RELEVANCE_WEIGHT = config["RELEVANCE_WEIGHT"]

    texts, scores = get_context_with_dartboard(test_query, num_results=3)
    print(f"\n配置：{config}")
    print(f"平均分数：{np.mean(scores):.2f}")
```

### 练习 2: 用自己的数据测试

```python
# 使用自己的文档集
my_texts = ["文档 1 内容", "文档 2 内容", ...]
my_vectorstore = encode_from_string(my_texts)

# 测试飞镖板检索
my_query = "你的问题"
results, scores = get_context_with_dartboard(my_query, num_results=5)
```

---

## 📚 总结

恭喜你完成了飞镖板检索的学习！现在你已经：

✅ **理解了**相关性和多样性的权衡
✅ **掌握了**飞镖板算法的核心原理
✅ **学会了**距离计算和概率转换
✅ **能够**在自己的项目中应用此技术

**下一步学习建议**：
1. 尝试不同的参数配置
2. 与其他 RAG 技术结合使用
3. 学习下一篇：多模态 RAG

---

> **💪 记住**：数学不可怕！每个公式背后都有直观的含义。多思考"为什么"，你一定能理解透彻。
>
> 如果本教程对你有帮助，欢迎分享给更多朋友！🌟
