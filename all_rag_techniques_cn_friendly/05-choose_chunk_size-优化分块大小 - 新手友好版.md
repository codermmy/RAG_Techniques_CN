# 🌟 新手入门：优化分块大小以改进 RAG 检索

> **💡 给新手的说明**
> - **难度等级**：⭐⭐⭐☆☆（进阶级）
> - **预计时间**：40-60 分钟
> - **前置知识**：基础 RAG 系统知识，了解文本分块概念
> - **学习目标**：学会通过实验找到最佳的分块参数，提升 RAG 系统性能

---

## 📖 核心概念理解

### 什么是分块（Chunking）？

**分块**是将长文档切分成小片段的过程，是 RAG 系统的关键预处理步骤。

### 🍕 通俗理解：如何切蛋糕

想象你要把一个大蛋糕分给很多人：

1. **块太大（chunk_size 太大）**：
   - 🍰 一块蛋糕太大，客人吃不完
   - 对应：检索到的内容包含太多无关信息

2. **块太小（chunk_size 太小）**：
   - 🧁 块太小，客人吃不饱，需要拿很多块
   - 对应：丢失上下文，需要检索更多块才能获得完整信息

3. **块大小合适**：
   - 🎂 每块大小刚好，客人吃得满意
   - 对应：检索到的内容既完整又精确

### 📊 分块大小的影响

| 块大小 | 优点 | 缺点 |
|--------|------|------|
| **大块** (1000+ tokens) | 保留更多上下文 | 可能包含噪声，检索精度下降 |
| **中块** (256-512 tokens) | 平衡上下文和精度 | 需要根据具体场景调整 |
| **小块** (128 tokens) | 检索精确 | 可能丢失上下文信息 |

### 🔍 为什么要优化分块大小？

不同的应用场景需要不同的分块大小：

- **技术文档**：可能需要较大的块（保留代码上下文）
- **FAQ 文档**：可能需要较小的块（每个问题一个块）
- **书籍**：可能需要中等大小的块（按段落或章节分）

### 📈 评估指标解释

| 指标 | 英文 | 含义 | 生活比喻 |
|------|------|------|----------|
| **Faithfulness** | 忠实度 | 答案是否基于检索到的内容 | 是否"有凭有据" |
| **Relevancy** | 相关性 | 答案是否回答了问题 | 是否"对题" |
| **Response Time** | 响应时间 | 从提问到得到答案的时间 | 是否"快速" |

---

## 🛠️ 第一步：环境准备

### 📖 这是什么？

安装优化分块大小实验所需的 Python 库。

### 💻 完整代码

```python
# ============================================
# 安装所需的包
# ============================================
# llama-index: LlamaIndex 框架核心
# openai: OpenAI SDK
# python-dotenv: 环境变量管理

!pip install llama-index openai python-dotenv
```

> **💡 代码解释**
> - **LlamaIndex**：一个专门的数据框架，用于构建 LLM 应用
> - 它提供了方便的评估工具，适合做分块大小的对比实验
>
> **⚠️ 新手注意**
> - 安装可能需要几分钟
> - 如果遇到依赖冲突，可以尝试创建虚拟环境

---

## 🔑 第二步：导入库和配置

### 📖 这是什么？

导入必要的库并配置 API 密钥。

### 💻 完整代码

```python
# ============================================
# 导入必要的库
# ============================================
import nest_asyncio
import random

# 解决 Jupyter Notebook 中的异步问题
nest_asyncio.apply()

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.prompts import PromptTemplate
from llama_index.core.evaluation import (
    DatasetGenerator,
    FaithfulnessEvaluator,
    RelevancyEvaluator
)
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

import openai
import time
import os

# 加载环境变量
load_dotenv()

# 设置 OpenAI API 密钥
openai.api_key = os.getenv("OPENAI_API_KEY")
```

> **💡 代码解释**
>
> **nest_asyncio 是什么？**
> - 解决 Jupyter Notebook 中异步事件循环冲突的问题
> - 简单说就是让异步代码能在 Notebook 中正常运行
>
> **LlamaIndex 核心组件**：
> - `VectorStoreIndex`：向量索引，用于存储和检索文档
> - `SimpleDirectoryReader`：从目录读取文档
> - `FaithfulnessEvaluator`：忠实度评估器
> - `RelevancyEvaluator`：相关性评估器
>
> **Settings 是什么？**
> - LlamaIndex 的全局配置
> - 设置后所有组件都会使用这些配置
>
> **⚠️ 新手注意**
> - 确保 `.env` 文件中有 `OPENAI_API_KEY`
> - 这个实验会多次调用 API，可能产生一定费用

---

## 📄 第三步：读取文档

### 📖 这是什么？

加载用于实验的文档数据。

### 💻 完整代码

```python
# 指定数据目录
data_dir = "../data"

# 从目录加载所有文档
documents = SimpleDirectoryReader(data_dir).load_data()
```

> **💡 代码解释**
> - `SimpleDirectoryReader` 会自动读取目录中的所有支持的文件
> - 支持 PDF、TXT、DOCX 等格式
>
> **⚠️ 新手注意**
> - 确保 `../data` 目录存在且有文档
> - 可以替换成你自己的文档路径
>
> **📊 查看文档信息**
> ```python
> print(f"文档数量：{len(documents)}")
> print(f"第一个文档的前 200 字符：{documents[0].text[:200]}")
> ```

---

## ❓ 第四步：创建评估问题

### 📖 这是什么？

自动生成一些测试问题，用于评估不同分块大小的效果。

### 💻 完整代码

```python
# 设置评估问题数量
num_eval_questions = 25

# 使用前 20 个文档来生成评估问题
eval_documents = documents[0:20]

# 创建数据集生成器
data_generator = DatasetGenerator.from_documents(eval_documents)

# 从文档节点生成问题
eval_questions = data_generator.generate_questions_from_nodes()

# 随机选择指定数量的问题
k_eval_questions = random.sample(eval_questions, num_eval_questions)
```

> **💡 代码解释**
>
> **DatasetGenerator 是什么？**
> - LlamaIndex 提供的工具，自动从文档生成问题和答案
> - 基于文档内容生成，确保问题有答案
>
> **为什么只用 20 个文档？**
> - 生成问题需要调用 API，用部分文档可以节省费用
> - 20 个文档通常足够生成有代表性的问题
>
> **⚠️ 新手注意**
> - 生成问题可能需要一些时间
> - 可以调整 `num_eval_questions` 来改变测试规模
>
> **📊 查看生成的问题**
> ```python
> for i, q in enumerate(k_eval_questions[:5]):  # 显示前 5 个问题
>     print(f"{i+1}. {q}")
> ```

---

## 📊 第五步：配置评估器

### 📖 这是什么？

配置用于评估答案质量的评估器。

### 💻 完整代码

```python
# ============================================
# 配置评估器
# ============================================

# 使用 GPT-4 来评估响应（更准确的评估）
gpt4 = OpenAI(temperature=0, model="gpt-4o")

# 为 LLM 设置适当的配置
Settings.llm = gpt4

# ========== 定义基于 GPT-4 的 Faithfulness 评估器 ==========
faithfulness_gpt4 = FaithfulnessEvaluator()

# 修改 Faithfulness 评估器的提示模板
# 让它更严格地判断答案是否基于上下文
faithfulness_new_prompt_template = PromptTemplate("""
请判断给定信息是否直接得到上下文的支持。
你需要回答 YES 或 NO。
如果上下文的任何部分明确支持该信息，即使大多数上下文无关，也回答 YES。
如果上下文没有明确支持该信息，回答 NO。

下面提供了一些示例。

Information: Apple pie is generally double-crusted.
Context: An apple pie is a fruit pie in which the principal filling ingredient is apples.
Apple pie is often served with whipped cream, ice cream ('apple pie à la mode'), custard, or cheddar cheese.
It is generally double-crusted, with pastry both above and below the filling; the upper crust may be solid or latticed (woven of crosswise strips).
Answer: YES

Information: Apple pies taste bad.
Context: An apple pie is a fruit pie in which the principal filling ingredient is apples.
Apple pie is often served with whipped cream, ice cream ('apple pie à la mode'), custard, or cheddar cheese.
It is generally double-crusted, with pastry both above and below the filling; the upper crust may be solid or latticed (woven of crosswise strips).
Answer: NO

Information: Paris is the capital of France.
Context: This document describes a day trip in Paris. You will visit famous landmarks like the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral.
Answer: NO

Information: {query_str}
Context: {context_str}
Answer:
""")

# 用新的提示模板更新评估器
faithfulness_gpt4.update_prompts({"your_prompt_key": faithfulness_new_prompt_template})

# ========== 定义基于 GPT-4 的 Relevancy 评估器 ==========
relevancy_gpt4 = RelevancyEvaluator()
```

> **💡 代码解释**
>
> **为什么要自定义提示模板？**
> - 默认的评估标准可能不够严格
> - 自定义模板可以让评估更符合需求
>
> **Few-Shot Prompting**：
> - 给 AI 提供几个示例（苹果派的例子）
> - 帮助 AI 理解评分标准
>
> **评估标准说明**：
> - **Faithfulness（忠实度）**：答案是否基于检索到的内容，有无瞎编
> - **Relevancy（相关性）**：答案是否回答了问题，有无跑题
>
> **⚠️ 新手注意**
> - 使用 GPT-4 评估会更准确，但费用更高
> - 可以改用 GPT-3.5 来节省成本

---

## 🧪 第六步：定义评估函数

### 📖 这是什么？

创建一个函数来评估特定分块大小的效果。

### 💻 完整代码

```python
def evaluate_response_time_and_accuracy(chunk_size, eval_questions):
    """
    评估 GPT-3.5-turbo 为给定块大小生成的响应的平均响应时间、
    faithfulness（忠实度）和 relevancy（相关性）。

    参数：
    chunk_size (int): 正在处理的数据块大小。
    eval_questions (list): 评估问题列表。

    返回：
    tuple: 包含平均响应时间、faithfulness 和 relevancy 指标的元组。

    指标说明：
    - 响应时间：从提问到得到答案的平均时间（秒）
    - faithfulness: 答案基于检索内容的比例（0-1）
    - relevancy: 答案回答问题的比例（0-1）
    """

    total_response_time = 0      # 总响应时间
    total_faithfulness = 0       # 总忠实度得分
    total_relevancy = 0          # 总相关性得分

    # ========== 创建向量索引 ==========
    # 使用 GPT-3.5-Turbo 来生成响应（更快更便宜）
    llm = OpenAI(model="gpt-3.5-turbo")

    # 设置 LLM 配置
    Settings.llm = llm
    Settings.chunk_size = chunk_size          # 设置块大小
    Settings.chunk_overlap = chunk_size // 5  # 重叠设置为块大小的 20%

    # 从文档创建向量索引
    vector_index = VectorStoreIndex.from_documents(eval_documents)

    # ========== 构建查询引擎 ==========
    # similarity_top_k=5 表示每次检索返回 5 个最相似的块
    query_engine = vector_index.as_query_engine(similarity_top_k=5)

    num_questions = len(eval_questions)

    # ========== 遍历每个问题进行评估 ==========
    # 虽然 BatchEvalRunner 可用于更快的评估，
    # 但我们在这里使用循环来专门测量不同块大小的响应时间。
    for question in eval_questions:
        # 记录开始时间
        start_time = time.time()

        # 获取答案
        response_vector = query_engine.query(question)

        # 计算耗时
        elapsed_time = time.time() - start_time

        # 评估忠实度：答案是否基于检索内容
        faithfulness_result = faithfulness_gpt4.evaluate_response(
            response=response_vector
        ).passing

        # 评估相关性：答案是否回答问题
        relevancy_result = relevancy_gpt4.evaluate_response(
            query=question, response=response_vector
        ).passing

        # 累加结果
        total_response_time += elapsed_time
        total_faithfulness += faithfulness_result
        total_relevancy += relevancy_result

    # ========== 计算平均值 ==========
    average_response_time = total_response_time / num_questions
    average_faithfulness = total_faithfulness / num_questions
    average_relevancy = total_relevancy / num_questions

    return average_response_time, average_faithfulness, average_relevancy
```

> **💡 代码解释**
>
> **chunk_overlap 是什么？**
> - 相邻块之间的重叠部分
> - 设置为 `chunk_size // 5` 即 20% 的重叠
> - 作用：避免关键信息被切分到两个块中间
>
> **为什么要循环而不是批量？**
> - 为了准确测量每个问题的响应时间
> - 批量评估会混在一起，无法精确计时
>
> **⚠️ 新手注意**
> - 这个函数运行一次可能需要几分钟（取决于问题数量）
> - 每次都会创建新的索引，会产生 API 调用费用
>
> **📊 评估结果示例**
> ```
> 假设 25 个问题的结果：
> - 总时间：125 秒
> - 15 个答案忠实（faithfulness_result=1），10 个不忠实（=0）
> - 20 个答案相关（relevancy_result=1），5 个不相关（=0）
>
> 平均值：
> - 响应时间：125/25 = 5.0 秒
> - 忠实度：15/25 = 0.6
> - 相关性：20/25 = 0.8
> ```

---

## 🔬 第七步：测试不同的块大小

### 📖 这是什么？

运行实验，比较不同块大小的效果。

### 💻 完整代码

```python
# ============================================
# 测试不同的块大小
# ============================================

# 定义要测试的块大小列表
chunk_sizes = [128, 256]

# 对每个块大小进行评估
for chunk_size in chunk_sizes:
    print(f"\n{'='*50}")
    print(f"正在测试块大小：{chunk_size}")
    print(f"{'='*50}")

    avg_response_time, avg_faithfulness, avg_relevancy = evaluate_response_time_and_accuracy(
        chunk_size,
        k_eval_questions
    )

    print(f"块大小 {chunk_size} - ")
    print(f"  平均响应时间：{avg_response_time:.2f}秒")
    print(f"  平均 Faithfulness: {avg_faithfulness:.2f}")
    print(f"  平均 Relevancy: {avg_relevancy:.2f}")
```

> **💡 预期输出示例**
> ```
> ==================================================
> 正在测试块大小：128
> ==================================================
> 块大小 128 -
>   平均响应时间：4.25 秒
>   平均 Faithfulness: 0.72
>   平均 Relevancy: 0.84
>
> ==================================================
> 正在测试块大小：256
> ==================================================
> 块大小 256 -
>   平均响应时间：4.80 秒
>   平均 Faithfulness: 0.80
>   平均 Relevancy: 0.88
> ```
>
> **⚠️ 新手注意**
> - 运行时间较长，建议先用小块大小测试（如只测试 1-2 个问题）
> - 结果会有波动，可以多次运行取平均值
>
> **📊 结果解读**
> - **响应时间**：越小越好
> - **Faithfulness/Relevancy**：越接近 1 越好
> - 需要在速度和准确性之间权衡

---

## 📈 第八步：结果分析与可视化

### 📖 这是什么？

分析实验结果，找到最佳分块大小。

### 💻 完整代码

```python
# ============================================
# 扩展测试（可选）
# ============================================

# 测试更多块大小
chunk_sizes = [128, 256, 512, 1024]
results = []

for chunk_size in chunk_sizes:
    avg_response_time, avg_faithfulness, avg_relevancy = evaluate_response_time_and_accuracy(
        chunk_size,
        k_eval_questions
    )
    results.append({
        'chunk_size': chunk_size,
        'response_time': avg_response_time,
        'faithfulness': avg_faithfulness,
        'relevancy': avg_relevancy
    })
    print(f"块大小 {chunk_size}: 时间={avg_response_time:.2f}s, "
          f"忠实度={avg_faithfulness:.2f}, 相关性={avg_relevancy:.2f}")
```

### 可视化结果

```python
# 如果有 matplotlib，可以绘制图表
try:
    import matplotlib.pyplot as plt

    # 提取数据
    sizes = [r['chunk_size'] for r in results]
    times = [r['response_time'] for r in results]
    faithfulness = [r['faithfulness'] for r in results]
    relevancy = [r['relevancy'] for r in results]

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 左图：响应时间
    ax1.plot(sizes, times, 'o-', label='响应时间')
    ax1.set_xlabel('块大小')
    ax1.set_ylabel('平均响应时间（秒）')
    ax1.set_title('块大小 vs 响应时间')
    ax1.grid(True)

    # 右图：准确率
    ax2.plot(sizes, faithfulness, 'o-', label='Faithfulness')
    ax2.plot(sizes, relevancy, 's-', label='Relevancy')
    ax2.set_xlabel('块大小')
    ax2.set_ylabel('得分')
    ax2.set_title('块大小 vs 准确率')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

except ImportError:
    print("安装 matplotlib 以查看可视化结果：!pip install matplotlib")
```

> **💡 图表解读**
> - 响应时间图：通常块越大，时间越长
> - 准确率图：可能存在一个最佳点，不是越大越好
>
> **📊 典型趋势**
> ```
> 块大小   响应时间   Faithfulness   Relevancy
> 128      快         较低           较低
> 256      中等       中等           中等
> 512      较慢       较高           较高
> 1024     慢         可能下降       可能下降
> ```

---

## 🎯 完整代码总结

下面是一个简化的实验模板：

```python
# 1. 导入必要的库
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
import time

# 2. 加载文档
documents = SimpleDirectoryReader("data").load_data()

# 3. 定义评估函数
def test_chunk_size(chunk_size):
    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = chunk_size // 5

    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    start = time.time()
    response = query_engine.query("测试问题")
    elapsed = time.time() - start

    print(f"块大小 {chunk_size}: 时间={elapsed:.2f}s, 答案={response}")
    return elapsed

# 4. 测试不同块大小
for size in [128, 256, 512]:
    test_chunk_size(size)
```

---

## ❓ 常见问题 FAQ

### Q1: 最佳块大小是多少？
**A**:
- 没有统一答案，取决于：
  - 文档类型（技术文档、小说、新闻等）
  - 问题类型（事实性问题、理解性问题等）
  - 使用的模型
- 建议从 256-512 开始测试

### Q2: 为什么需要重叠（overlap）？
**A**:
- 避免关键信息被切分到两个块中间
- 例如："人工智能是一种..."如果被切成"人工"和"智能是一种..."就丢失了语义
- 20% 重叠是一个经验值

### Q3: 评估需要多少问题？
**A**:
- 快速测试：10-25 个问题
- 正式评估：50-100 个问题
- 越多越准确，但费用也越高

### Q4: 可以不用 OpenAI 吗？
**A**:
- 可以！LlamaIndex 支持多种模型
- 如 Anthropic、Cohere、本地模型等
- 但 OpenAI 的评估工具最成熟

### Q5: 除了块大小，还需要优化什么？
**A**:
- **重叠大小**：通常设为块大小的 10-20%
- **检索数量（top_k）**：返回多少块给 LLM
- **分块策略**：按句子、段落、还是固定字符数

---

## 🚀 进阶技巧

### 针对不同文档类型的推荐设置

```python
# 技术文档（代码、API 文档）
Settings.chunk_size = 512    # 较大，保留代码上下文
Settings.chunk_overlap = 100

# FAQ 文档
Settings.chunk_size = 256    # 较小，每个问题一个块
Settings.chunk_overlap = 50

# 书籍/长文
Settings.chunk_size = 1024   # 大，保留章节上下文
Settings.chunk_overlap = 200

# 新闻文章
Settings.chunk_size = 512    # 中等，一篇文章可能一个块就够了
Settings.chunk_overlap = 100
```

### 自动化网格搜索

```python
def grid_search_chunk_sizes(documents, eval_questions):
    """自动化搜索最佳块大小"""
    best_score = 0
    best_size = 0

    for size in [128, 256, 512, 1024]:
        Settings.chunk_size = size
        Settings.chunk_overlap = size // 5

        # 创建索引
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()

        # 评估
        correct = 0
        for q in eval_questions[:10]:  # 用 10 个问题快速测试
            response = query_engine.query(q)
            # 简单评估（实际应该用评估器）
            if response and len(str(response)) > 0:
                correct += 1

        score = correct / 10
        print(f"块大小 {size}: 得分 {score}")

        if score > best_score:
            best_score = score
            best_size = size

    print(f"最佳块大小：{best_size}")
    return best_size
```

---

## 📚 关键知识点回顾

| 概念 | 说明 | 推荐值 |
|------|------|--------|
| **Chunk Size** | 每个文本块的大小 | 256-512 tokens |
| **Chunk Overlap** | 相邻块的重叠部分 | 块大小的 10-20% |
| **Faithfulness** | 答案是否基于检索内容 | 越接近 1 越好 |
| **Relevancy** | 答案是否回答问题 | 越接近 1 越好 |
| **Top-k** | 检索返回的块数量 | 3-5 |
| **响应时间** | 从提问到回答的时间 | 越短越好 |

---

## 🎓 实验建议

### 第一次实验建议流程

1. **小规模测试**：用 5-10 个问题快速测试
2. **确定范围**：找到大致的最佳范围
3. **精细调整**：在最佳范围附近测试更多值
4. **验证结果**：用更多问题验证

### 记录实验结果

```
实验日期：2024-XX-XX
文档类型：技术文档
问题数量：25

| 块大小 | 响应时间 | Faithfulness | Relevancy |
|--------|----------|--------------|-----------|
| 128    | 4.25s    | 0.72         | 0.84      |
| 256    | 4.80s    | 0.80         | 0.88      |
| 512    | 5.50s    | 0.78         | 0.85      |

结论：256 是最佳选择
```

---

*本教程是 RAG 技术系列教程的优化专题，建议先学习基础 RAG 教程再学习本教程。*

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--choose-chunk-size)
