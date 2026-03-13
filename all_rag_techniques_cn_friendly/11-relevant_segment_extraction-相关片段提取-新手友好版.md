# 🌟 新手入门：相关片段提取 (RSE)

> **💡 给新手的说明**
> - **难度等级**：⭐⭐⭐⭐ 中高级
> - **预计学习时间**：60-90 分钟
> - **前置知识**：了解 RAG 检索流程、基础的向量搜索概念
> - **本教程特色**：保留所有技术细节，增加通俗解释和代码注释

---

## 📖 核心概念理解

### 什么是相关片段提取？

想象你在准备考试，老师给了你一本被撕成单页的教科书。当你复习某个知识点时：

**传统方法（Top-k 检索）**：
- 老师只给你最相关的 5 页
- 但这 5 页可能分散在书的不同位置
- 你看不到完整的知识脉络

**相关片段提取（RSE）**：
- 老师不仅给你最相关的 5 页
- 还会把这些页在原书中**连续的部分**都给你
- 甚至会把**夹在相关页之间**的不太相关的页也给你
- 这样你能看到完整的知识片段

### 通俗理解

| 场景 | 传统检索 | RSE |
|-----|---------|-----|
| 看电影预告片 | 只给你看最精彩的 5 个镜头（可能来自不同电影） | 给你看完整的故事片段（包括过渡镜头） |
| 读小说 | 只返回包含关键词的句子 | 返回完整的章节片段 |
| 查地图 | 只显示 5 个最相关的地点 | 显示连续的区域，包括地点之间的道路 |

### 核心洞察

**相关块倾向于在其原始文档中聚集**

这是一个简单但强大的观察：
- 如果第 10 块与"财务报表"相关
- 那么第 11、12 块也很可能相关
- 即使第 11 块没有直接被检索到

### 为什么要用 RSE？

传统 RAG 面临的困境：

```
小块（200-500 字）vs 大块（2000+ 字）
      ↓                    ↓
  检索精准                上下文完整
  上下文缺失              检索困难
```

RSE 的解决方案：**动态片段长度**
- 简单问题 → 返回短片段
- 复杂问题 → 自动组合成大片段

---

## 🛠️ 第一步：环境准备与包安装

### 📖 这是什么？

准备运行 RSE 所需的环境。我们需要 Cohere 的重排序器来计算相关性分数。

### 💻 完整代码

```python
# 安装所需的包
# matplotlib: 绘图库，用于可视化相关性分布
# numpy: 数值计算库，用于数组操作
# python-dotenv: 管理环境变量
# scipy: 科学计算库，提供 beta 分布函数
!pip install matplotlib numpy python-dotenv
```

```python
import os
import numpy as np  # 数值计算
from typing import List  # 类型提示
from scipy.stats import beta  # beta 分布，用于转换相关性分数
import matplotlib.pyplot as plt  # 绘图
import cohere  # Cohere API
from dotenv import load_dotenv  # 环境变量管理

# 从 .env 文件加载环境变量
load_dotenv()

# 设置 Cohere API 密钥
os.environ["CO_API_KEY"] = os.getenv('CO_API_KEY')
```

> **💡 代码解释**
> - `scipy.stats.beta`：beta 分布用于将原始相关性分数转换到更均匀的分布
> - `numpy`：用于高效的数值计算，特别是数组操作
> - `matplotlib`：可视化块的相关性分布，帮助理解算法行为
>
> **⚠️ 新手注意**
> - 你需要 Cohere API 密钥来运行重排序功能
> - 在 `.env` 文件中设置：`CO_API_KEY=你的密钥`
> - 如果没有 API 密钥，可以用模拟数据学习算法逻辑

---

## 🛠️ 第二步：文档分块与相关性计算

### 📖 这是什么？

RSE 的第一步是把文档切成**不重叠**的块，然后计算每个块与查询的相关性分数。

### 重要概念：为什么不能重叠？

```
❌ 重叠分块：块 1[0-100] 块 2[80-180] 块 3[160-260]
   问题：拼接时会重复包含 80-100、160-180 的内容

✅ 不重叠分块：块 1[0-100] 块 2[100-200] 块 3[200-300]
   优点：可以干净利落地拼接成连续片段
```

### 💻 完整代码

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_into_chunks(text: str, chunk_size: int):
    """
    使用 RecursiveCharacterTextSplitter 将给定文本分割成指定大小的块。

    参数：
        text (str): 要分割成块的输入文本。
        chunk_size (int): 每个块的最大大小。

    返回：
        list[str]: 文本块列表。

    示例：
        >>> text = "This is a sample text to be split into chunks."
        >>> chunks = split_into_chunks(text, chunk_size=10)
        >>> print(chunks)
        ['This is a', 'sample', 'text to', 'be split', 'into', 'chunks.']
    """
    # 创建文本分割器
    # chunk_overlap=0: 关键！块与块之间不能重叠
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,  # ⚠️ RSE 要求无重叠
        length_function=len
    )

    # 执行分割
    texts = text_splitter.create_documents([text])

    # 提取纯文本内容
    chunks = [text.page_content for text in texts]
    return chunks
```

```python
def transform(x: float):
    """
    转换函数：将绝对相关性值映射到 0 和 1 之间更均匀分布的值。

    为什么需要转换？
    Cohere 重排序器返回的相关性分数往往集中在 0 或 1 附近：
    - 大部分分数要么接近 0（不相关）
    - 要么接近 1（非常相关）
    中间的分数很少

    beta 分布可以让分数分布更均匀，便于后续处理。

    参数：
        x (float): Cohere 重排序器返回的绝对相关性值

    返回：
        float: 转换后的相关性值
    """
    # beta 分布的参数
    # a=0.4, b=0.4 会产生 U 形分布，拉伸两端的值
    a, b = 0.4, 0.4

    # 使用累积分布函数（CDF）进行转换
    return beta.cdf(x, a, b)
```

```python
def rerank_chunks(query: str, chunks: List[str]):
    """
    使用 Cohere Rerank API 重新排序搜索结果。

    这个函数不仅返回相似度分数，还计算"块值"（chunk_values）。
    块值 = 相关性 × 排名衰减因子

    参数：
        query (str): 搜索查询
        chunks (list): 要重新排序的块列表

    返回：
        similarity_scores (list): 每个块的相似度分数列表
        chunk_values (list): 每个块的相关性值列表（排名和相似度的融合）
    """
    model = "rerank-english-v3.0"  # Cohere 重排序模型
    client = cohere.Client(api_key=os.environ["CO_API_KEY"])
    decay_rate = 30  # 排名衰减率

    # 调用重排序 API
    reranked_results = client.rerank(model=model, query=query, documents=chunks)
    results = reranked_results.results

    # 提取排序后的索引和分数
    reranked_indices = [result.index for result in results]
    reranked_similarity_scores = [result.relevance_score for result in results]

    # 转换回原始顺序，并计算块值
    similarity_scores = [0] * len(chunks)
    chunk_values = [0] * len(chunks)

    for i, index in enumerate(reranked_indices):
        # 第一步：用 beta 分布转换原始相关性分数
        absolute_relevance_value = transform(reranked_similarity_scores[i])

        # 第二步：存储转换后的相似度分数（按原始顺序）
        similarity_scores[index] = absolute_relevance_value

        # 第三步：计算块值 = 相关性 × 排名衰减
        # 排名越靠前（i 越小），衰减越小
        # exp(-i/30)：第 0 名≈1.0，第 30 名≈0.37，第 60 名≈0.14
        chunk_values[index] = np.exp(-i/decay_rate) * absolute_relevance_value

    return similarity_scores, chunk_values
```

> **💡 代码解释**
>
> **为什么要结合排名和相关性？**
> - 仅用相似度：可能把分散的高分块组合在一起
> - 仅用排名：忽略了实际相关性强度
> - 结合两者：既考虑相关性强度，又考虑排序位置
>
> **衰减函数的作用**：
> ```
> 排名 0: exp(0) = 1.00（无衰减）
> 排名 10: exp(-10/30) ≈ 0.72
> 排名 30: exp(-30/30) ≈ 0.37
> 排名 60: exp(-60/30) ≈ 0.14
> ```
>
> **⚠️ 新手注意**
> - `decay_rate` 控制衰减速度：越大衰减越慢
> - 可以根据你的文档长度调整这个参数

---

## 🛠️ 第三步：下载数据并分割文档

### 💻 完整代码

```python
# 下载所需的数据文件
import os
os.makedirs('data', exist_ok=True)

# 下载本笔记本使用的 PDF 文档
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
!wget -O data/nike_2023_annual_report.txt https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/nike_2023_annual_report.txt
```

```python
# 输入文档的文件路径
FILE_PATH = "data/nike_2023_annual_report.txt"

# 读取文档
with open(FILE_PATH, 'r') as file:
    text = file.read()

# 分割成块（每块 800 字符）
chunks = split_into_chunks(text, chunk_size=800)

print(f"将文档分割成 {len(chunks)} 个块")
# 输出示例：将文档分割成 450 个块
```

---

## 🛠️ 第四步：可视化块相关性分布

### 📖 这是什么？

在找最佳片段之前，我们先"看一眼"相关性在文档中是如何分布的。这能帮助我们理解 RSE 的工作原理。

### 💻 完整代码

```python
def plot_relevance_scores(chunk_values: List[float], start_index: int = None, end_index: int = None) -> None:
    """
    可视化文档中每个块与搜索查询的相关性分数。

    参数：
        chunk_values (list): 每个块的相关性值列表
        start_index (int): 要绘制的块的起始索引（可选）
        end_index (int): 要绘制的块的结束索引（可选）

    返回：
        None（直接绘制散点图）
    """
    plt.figure(figsize=(12, 5))  # 设置图表大小：宽 12，高 5
    plt.title(f"文档中每个块与搜索查询的相似度")
    plt.ylim(0, 1)  # y 轴范围：0 到 1
    plt.xlabel("块索引")  # x 轴：块的编号
    plt.ylabel("查询 - 块相似度")  # y 轴：相关性分数

    # 设置绘制范围
    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = len(chunk_values)

    # 绘制散点图
    plt.scatter(range(start_index, end_index), chunk_values[start_index:end_index])
    plt.show()
```

```python
# 示例查询：需要比单个块更长的结果
query = "Nike 合并财务报表"

# 计算相关性分数
similarity_scores, chunk_values = rerank_chunks(query, chunks)

# 绘制全图
plot_relevance_scores(chunk_values)
```

### 📊 如何解读相关性图

```
y 轴（相关性）
  1.0 |     ●           ●●●●
      |    ● ●         ●    ●      ●
  0.5 |   ●   ●       ●      ●    ● ●
      |  ●     ●     ●        ●  ●   ●
  0.0 |_●_______●___●__________●●_____●____ x 轴（块索引）
       0      100  200        300     400

观察：
● 相关块倾向于"聚集"（如 320-340 区域的密集点）
● 不相关的块夹在相关块之间
● 这正是 RSE 发挥作用的地方！
```

### 放大查看关键区域

```python
# 放大查看 320-340 区域
plot_relevance_scores(chunk_values, 320, 340)
```

```python
def print_document_segment(chunks: List[str], start_index: int, end_index: int):
    """
    打印文档片段的文本内容。

    参数：
        chunks (list): 文本块列表
        start_index (int): 片段的起始索引
        end_index (int): 片段的结束索引（不包含）

    返回：
        None（直接打印）
    """
    for i in range(start_index, end_index):
        print(f"\n=== 块 {i} ===")
        print(chunks[i])

# 查看 320-340 区域的实际内容
print_document_segment(chunks, 320, 340)
```

> **💡 关键发现**
>
> 查看后会发现：
> - 块 323-333：合并利润表内容（都相关）
> - 但重排序器只标记了约一半的块为"相关"
> - RSE 能自动把这些连续的块组合起来！

---

## 🛠️ 第五步：核心算法 - 寻找最佳片段

### 📖 这是什么？

这是 RSE 的核心算法。它要解决的问题是：**如何自动识别相关块的集群？**

### 算法思路

```
步骤 1：给每个块定义"值"
       相关块 = 正值（如 +0.8）
       不相关块 = 负值（如 -0.2）

步骤 2：片段的值 = 组成块的值之和

步骤 3：找到值最大的片段（最大和子数组问题）

步骤 4：重复找下一个，直到满足条件
```

### 通俗理解

想象你在找一片"好草地"（相关块集群）：
- 🌱 好草（相关块）：+1 分
- 🪨 石头（不相关块）：-0.2 分
- 你要找草最密集的区域（总分最高）
- 偶尔有几块石头没关系，只要整体草多

### 💻 完整代码

```python
def get_best_segments(relevance_values: list, max_length: int, overall_max_length: int, minimum_value: float):
    """
    获取块相关性值，然后运行优化算法以找到最佳片段。

    用更技术的术语来说：它解决了约束版本的最大和子数组问题。

    参数：
        relevance_values (list): 文档每个块的相关性值列表
        max_length (int): 单个片段的最大长度（以块数衡量）
        overall_max_length (int): 所有片段的最大长度（以块数衡量）
        minimum_value (float): 片段必须具有的最小值才能被视为最佳片段

    返回：
        best_segments (list): 元组 (start, end) 列表，表示最佳片段的索引
        scores (list): 每个最佳片段的分数列表

    示例：
        返回 [(323, 336), (50, 65), ...]
        表示找到了 2 个片段：块 323-335 和 块 50-64
    """
    best_segments = []  # 存储找到的最佳片段
    scores = []  # 存储每个片段的分数
    total_length = 0  # 已选片段的总长度

    # 循环直到达到总体最大长度
    while total_length < overall_max_length:
        # 在剩余块中找到最佳片段
        best_segment = None
        best_value = -1000  # 初始化为很小的值

        # 暴力搜索：尝试所有可能的片段
        for start in range(len(relevance_values)):
            # 剪枝 1：跳过负值起点（不可能产生最优解）
            if relevance_values[start] < 0:
                continue

            for end in range(start+1, min(start+max_length+1, len(relevance_values)+1)):
                # 剪枝 2：跳过负值终点
                if relevance_values[end-1] < 0:
                    continue

                # 剪枝 3：跳过与已选片段重叠的
                if any(start < seg_end and end > seg_start for seg_start, seg_end in best_segments):
                    continue

                # 剪枝 4：跳过会超过总体最大长度的
                if total_length + end - start > overall_max_length:
                    continue

                # 计算片段值 = 块值之和
                segment_value = sum(relevance_values[start:end])

                # 更新最佳片段
                if segment_value > best_value:
                    best_value = segment_value
                    best_segment = (start, end)

        # 如果没有找到有效片段，或最佳片段值太小，则停止
        if best_segment is None or best_value < minimum_value:
            break

        # 将最佳片段添加到结果列表
        best_segments.append(best_segment)
        scores.append(best_value)
        total_length += best_segment[1] - best_segment[0]

    return best_segments, scores
```

> **💡 算法解释**
>
> **为什么是"约束版本"？**
> - 经典的最大和子数组问题：找整个数组中和最大的连续子数组
> - 我们的约束：
>   1. 片段长度不能超过 `max_length`
>   2. 所有片段总长度不能超过 `overall_max_length`
>   3. 片段之间不能重叠
>   4. 片段值必须超过 `minimum_value`
>
> **时间复杂度**：O(n²)，但对于典型文档大小（几百个块）足够快（5-10 毫秒）
>
> **⚠️ 新手注意**
> - 这是简化实现，生产环境可用更高效的算法
> - 暴力搜索配合剪枝在实际场景中表现良好

---

## 🛠️ 第六步：运行优化算法

### 💻 完整代码

```python
# 定义优化参数
irrelevant_chunk_penalty = 0.2  # 不相关块的惩罚值
max_length = 20                  # 单个片段最大长度（块数）
overall_max_length = 30          # 所有片段最大总长度（块数）
minimum_value = 0.7              # 片段最小值阈值

# 从块相关性值中减去恒定阈值
# 相关块（值>0.2）变正，不相关块（值<0.2）变负
relevance_values = [v - irrelevant_chunk_penalty for v in chunk_values]

# 运行优化算法
best_segments, scores = get_best_segments(
    relevance_values,
    max_length,
    overall_max_length,
    minimum_value
)

# 打印结果
print("=== 最佳片段索引 ===")
print(best_segments)  # 示例输出：[(323, 336), (45, 52)]
print()
print("=== 最佳片段分数 ===")
print(scores)  # 示例输出：[8.5, 3.2]
```

> **💡 参数调优指南**
>
> | 参数 | 作用 | 调大效果 | 调小效果 |
> |-----|------|---------|---------|
> | `irrelevant_chunk_penalty` | 不相关块惩罚 | 片段更紧凑 | 片段更宽松 |
> | `max_length` | 单片段最大长度 | 允许更长片段 | 片段更短 |
> | `overall_max_length` | 总长度限制 | 返回更多内容 | 内容更精简 |
> | `minimum_value` | 最小值阈值 | 只返回高质量片段 | 返回更多片段 |
>
> **经验值**：
> - `irrelevant_chunk_penalty`: 0.15-0.25（推荐 0.2）
> - `max_length`: 10-30（根据 chunk_size 调整）
> - `overall_max_length`: 20-50（控制总输出长度）

### 结果解读

```
假设输出：
最佳片段索引：[(323, 336), (45, 52)]
最佳片段分数：[8.5, 3.2]

解读：
- 第一个片段：块 323-335（共 13 块），分数 8.5（高质量）
- 第二个片段：块 45-51（共 7 块），分数 3.2（中等质量）

理想情况下：
- 第一个片段包含完整的合并财务报表
- 第二个片段包含其他相关信息
```

---

## 🛠️ 第七步：特殊情况处理

### 📖 如果答案在单个块中怎么办？

不是所有查询都需要长片段。有些简单问题（如"CEO 是谁？"）单个块就够了。

### RSE 的自适应能力

```
情况 A：相关块聚集
●●●●●  → 组合成长片段

情况 B：相关块孤立
●   ●   ●  → 退化为标准 top-k 检索
```

### 练习：观察简单查询的行为

```python
# 简单查询示例
simple_query = "Nike 的 CEO 是谁？"

# 计算相关性
simple_scores, simple_chunk_values = rerank_chunks(simple_query, chunks)

# 可视化
plot_relevance_scores(simple_chunk_values)

# 运行 RSE
simple_relevance_values = [v - 0.2 for v in simple_chunk_values]
simple_segments, simple_scores = get_best_segments(
    simple_relevance_values,
    max_length=20,
    overall_max_length=30,
    minimum_value=0.7
)

print(f"找到的片段：{simple_segments}")
# 可能输出：[]（如果没有足够密集的聚集）
# 这时应退化为返回 top-k 个块
```

> **💡 设计智慧**
>
> RSE 的优雅之处在于：
> - 有聚集时 → 组合成长片段
> - 无聚集时 → 自动退化为标准检索
> - 不需要额外的判断逻辑！

---

## 📊 评估结果

### KITE 基准测试

#### 实验设置

| 项目 | 配置 |
|-----|------|
| 基准 | KITE（知识密集型任务评估） |
| 对比 | RSE vs Top-k 检索（k=20） |
| 重排序器 | Cohere 3 |
| 生成模型 | GPT-4o |
| 控制变量 | 其他参数保持一致 |

#### 测试结果

| 数据集 | Top-k | RSE | 提升幅度 |
|-------|-------|-----|---------|
| AI Papers | 4.5 | 7.9 | +75.6% |
| BVP Cloud | 2.6 | 4.4 | +69.2% |
| Sourcegraph | 5.7 | 6.6 | +15.8% |
| Supreme Court | 6.1 | 8.0 | +31.1% |
| **平均分** | **4.72** | **6.73** | **+42.6%** |

> **💡 结果解读**
>
> - **AI Papers** 提升最大（+75.6%）：学术论文需要完整段落理解
> - **BVP Cloud** 显著提升（+69.2%）：财务报告需要连续上下文
> - **Supreme Court** 提升明显（+31.1%）：法律意见需要完整论述
> - **Sourcegraph** 提升较小（+15.8%）：技术文档相对独立
> - **总体平均提升 42.6%**：效果非常显著！

#### FinanceBench 测试

| 配置 | 得分 |
|-----|------|
| 基准 | 19% |
| CCH + RSE 组合 | 83% |

> **⚠️ 注意**：此测试同时使用了 CCH 和 RSE，两者组合效果显著，但无法单独量化各自贡献。

---

## ❓ 常见问题 FAQ

### Q1: RSE 会增加多少延迟？
**A**: 优化算法本身只需 5-10 毫秒。主要延迟在重排序步骤（与标准流程相同）。整体延迟增加可忽略不计。

### Q2: 为什么块不能重叠？
**A**: 重叠会导致拼接时内容重复。例如：
- 块 1[0-100] 和 块 2[80-180] 重叠
- 拼接后 80-100 的内容出现两次
- 这会干扰 LLM 的理解

### Q3: 可以组合使用 CCH 和 RSE 吗？
**A**: 强烈推荐！CCH 增强单个块的上下文，RSE 组织块的连续片段，两者互补。

### Q4: 如何选择合适的 `chunk_size`？
**A**:
- 小文档（<10K 字）：400-800 字符
- 中等文档（10K-100K 字）：600-1000 字符
- 大文档（>100K 字）：800-1500 字符

### Q5: 我的文档很短，还需要 RSE 吗？
**A**: 如果文档本身就很短（如<2000 字），分块数量少，RSE 的收益有限。

---

## 🔑 关键要点总结

1. **核心洞察**：相关块倾向于在原文档中聚集
2. **算法思路**：将片段选择转化为最大和子数组问题
3. **关键参数**：`irrelevant_chunk_penalty` 控制片段松紧度
4. **自适应**：自动适应简单/复杂查询，无需额外判断
5. **效果显著**：平均提升 42.6%，某些场景提升 75%+
6. **最佳实践**：与 CCH 组合使用效果更佳

---

## 📚 进阶学习建议

### 实践练习
1. **基础**：用示例数据跑通整个流程
2. **进阶**：调整参数，观察片段变化
3. **高级**：实现更高效的优化算法

### 扩展阅读
- 最大和子数组问题（Kadane 算法）
- 文档检索中的位置编码
- 重排序（Rerank）技术详解

### 动手项目
```python
# 挑战：实现 RSE 的生产级版本
# 要求：
# 1. 支持增量更新（文档变化时只更新部分块）
# 2. 支持多文档联合优化
# 3. 添加缓存机制减少重排序调用
```

> **💪 动手练习**：找一个实际项目，对比"Top-k"和"RSE"的检索结果差异，记录你的观察！

---

*本教程保持与原文档一致的技术深度，同时增加了通俗解释和实用指导。*
