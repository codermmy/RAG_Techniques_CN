# 🌟 新手入门：命题分块 (Propositions Chunking)

> **💡 给新手的说明**
> - **难度等级**：⭐⭐⭐⭐（中高级）
> - **预计学习时间**：45-60 分钟
> - **前置知识**：了解基础的 Python 编程，对 RAG 系统有基本认识
> - **本教程你将学会**：如何将文档分解为原子性命题，实现更精确的信息检索

---

## 📖 核心概念理解

### 什么是命题分块？

想象你在整理一本厚厚的书。传统方法是一页一页地切分（这就是普通的文本分块）。但**命题分块**更像是在提取书中的"知识点卡片"——每张卡片只记录一个完整的事实。

### 通俗理解

**生活化比喻**：
- 📚 **传统分块**：把一本书撕成小册子，每本 10 页。你想找"谁第一个登上月球"，可能要在好几本小册子里翻。
- 📇 **命题分块**：把书变成卡片盒，每张卡片写一个完整事实：
  - "1969 年，Neil Armstrong 在 Apollo 11 任务期间成为第一个在月球上行走的人"
  - 这张卡片会被拆成多张小卡片：
    - "Neil Armstrong 是一名宇航员"
    - "Neil Armstrong 于 1969 年在月球上行走"
    - "Neil Armstrong 是第一个在月球上行走的人"
    - "Apollo 11 任务发生于 1969 年"

现在，无论问什么问题，都能精准定位到对应的卡片！

### 为什么要用命题分块？

| 场景 | 传统分块 | 命题分块 |
|------|----------|----------|
| 问："谁第一个登月？" | 返回一大段文字，需要再找答案 | 精准返回"Neil Armstrong 是第一个在月球上行走的人" |
| 问："Apollo 11 是什么时候的任务？" | 可能返回不相关的段落 | 精准返回"Apollo 11 任务发生于 1969 年" |

---

## 🛠️ 第一步：环境准备

### 📖 这是什么？

在开始之前，我们需要安装必要的 Python 包。这就像做饭前要准备好锅碗瓢盆一样。

### 💻 完整代码

```python
# 安装所需的包
# !pip install faiss-cpu langchain langchain-community python-dotenv
```

> **💡 代码解释**
> - `faiss-cpu`：Facebook 开发的向量搜索库，用于快速找到相似的文本
> - `langchain`：构建 LLM 应用的框架
> - `langchain-community`：LangChain 的社区扩展包
> - `python-dotenv`：用于加载环境变量（存放 API 密钥）
>
> **⚠️ 新手注意**
> - 如果你使用的是 Jupyter Notebook，去掉 `!` 前面的 `#` 注释
> - 如果你使用的是普通 Python 脚本，在终端运行：`pip install faiss-cpu langchain langchain-community python-dotenv`
> - 安装可能需要几分钟，请耐心等待

### 配置环境变量

```python
### LLMs
import os
from dotenv import load_dotenv

# 从 '.env' 文件加载环境变量
load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')  # 用于 LLM
```

> **💡 代码解释**
> - `load_dotenv()`：从 `.env` 文件中读取环境变量
> - `os.getenv('GROQ_API_KEY')`：获取 GROQ API 密钥
>
> **⚠️ 新手注意**
> - 你需要先创建一个 `.env` 文件，内容格式：`GROQ_API_KEY=你的密钥`
> - API 密钥可以从 [Groq 官网](https://console.groq.com/) 免费获取
> - **永远不要**把 API 密钥直接写在代码里！

---

## 📝 第二步：准备测试文档

### 📖 这是什么？

我们需要一篇示例文档来演示命题分块的效果。这里使用关于 Paul Graham "Founder Mode" 文章的内容。

### 💻 完整代码

```python
sample_content = """Paul Graham 的文章"Founder Mode"发表于 2024 年 9 月，挑战了关于初创企业规模扩展的传统智慧，认为创始人应该保持独特的管理风格，而不是随着公司成长采用传统的企业管理实践。
传统智慧 vs 创始人模式
文章认为，给成长型公司的传统建议——雇佣优秀人才并给予它们自主权——在应用于初创企业时往往失败。
这种方法虽然适合成熟公司，但对创始人愿景和直接参与至关重要的初创企业可能有害。"Founder Mode"被描述为一个尚未完全理解或记录的新兴范式，与商学院和职业经理人经常建议的传统"经理人模式"形成对比。
独特的创始人能力
创始人拥有职业经理人没有的独特见解和能力，主要是因为他们对公司的愿景和文化有深刻理解。
Graham 建议创始人应该利用这些优势，而不是 conform 传统管理实践。"Founder Mode"是一个尚未完全理解或记录的新兴范式，Graham 希望随着时间的推移，它能像传统经理人模式一样被充分理解，使创始人即使在公司规模扩展时也能保持独特的方法。
扩展初创企业的挑战
随着初创企业成长，人们普遍认为必须过渡到更结构化的管理方法。然而，许多创始人发现这种过渡有问题，因为它经常导致失去推动初创企业最初成功的创新和敏捷精神。
Airbnb 联合创始人 Brian Chesky 分享了他的经验，他被告知以传统管理风格运营公司，导致了糟糕的结果。他最终通过采用不同的方法取得了成功，这个方法受到 Steve Jobs 管理 Apple 方式的启发。
Steve Jobs 的管理风格
Steve Jobs 在 Apple 的管理方法成为 Brian Chesky 在 Airbnb 实施"Founder Mode"的灵感来源。一个值得注意的做法是 Jobs 每年为 Apple 最重要的 100 人举办 retreat，无论他们在组织结构图上的位置如何。这种非常规方法使 Jobs 即使在 Apple 成长时也能保持初创企业般的环境，培养跨层级的创新和直接沟通。这些实践强调了创始人深入参与公司运营的重要性，挑战了随着公司规模扩大将责任委托给职业经理人的传统观念。
"""
```

> **💡 代码解释**
> - `sample_content`：存储示例文本的变量
> - 这段文本包含了多个事实，非常适合演示命题分块

---

## ✂️ 第三步：文档分块 (Chunking)

### 📖 这是什么？

在生成命题之前，我们先把长文档切成小块。为什么？因为 LLM（大语言模型）一次只能处理有限的内容，就像人一口吃不了一个大面包一样。

### 💻 完整代码

```python
### 构建索引
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

# 设置 embeddings
embedding_model = OllamaEmbeddings(model='nomic-embed-text:v1.5', show_progress=True)

# 文档列表
docs_list = [Document(page_content=sample_content, metadata={"Title": "Paul Graham 的 Founder Mode 文章", "Source": "https://www.perplexity.ai/page/paul-graham-s-founder-mode-ess-t9TCyvkqRiyMQJWsHr0fnQ"})]

# 分割
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=200, chunk_overlap=50
)

doc_splits = text_splitter.split_documents(docs_list)
```

> **💡 代码解释**
> - `RecursiveCharacterTextSplitter`：递归字符文本分割器，会智能地在句子边界处切分
> - `chunk_size=200`：每块最多 200 个字符
> - `chunk_overlap=50`：相邻块之间重叠 50 个字符（避免切断上下文）
> - `OllamaEmbeddings`：使用 Ollama 的本地 embedding 模型
>
> **⚠️ 新手注意**
> - `chunk_size` 太小会丢失上下文，太大会超出 LLM 处理能力
> - `chunk_overlap` 确保关键信息不会被切到两块之间
> - 需要安装 Ollama：`brew install ollama` (Mac) 或从官网下载

### 添加块 ID

```python
for i, doc in enumerate(doc_splits):
    doc.metadata['chunk_id'] = i+1  # 添加块 id
```

> **💡 代码解释**
> - 给每个块添加一个编号，方便后续追踪
> - `i+1` 是因为 Python 索引从 0 开始，但人类习惯从 1 开始数

---

## 🎯 第四步：生成命题 (Generate Propositions)

### 📖 这是什么？

这是核心步骤！我们用 LLM 把每个文本块拆分成一个个"原子事实"——命题。每个命题都是独立的、完整的、不需要额外上下文就能理解。

### 💻 完整代码

```python
from typing import List
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq

# 数据模型
class GeneratePropositions(BaseModel):
    """给定文档中的所有命题列表"""

    propositions: List[str] = Field(
        description="命题列表（事实性、自包含且简洁的信息）"
    )


# 使用函数调用的 LLM
llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
structured_llm= llm.with_structured_output(GeneratePropositions)
```

> **💡 代码解释**
> - `GeneratePropositions`：定义输出格式，告诉 LLM 我们想要一个命题列表
> - `ChatGroq`：通过 Groq 服务调用 LLM
> - `model="llama-3.1-70b-versatile"`：使用 Meta 的 Llama 3.1 70B 模型
> - `temperature=0`：让输出更稳定、更可预测（适合事实性任务）
>
> **⚠️ 新手注意**
> - `temperature` 范围是 0-1，0 最稳定，1 最有创造力
> - 事实性任务（如提取信息）用 0，创意性任务（如写诗）用 0.7-1

### Few-Shot 示例提示

```python
# Few shot 提示 --- 我们可以添加更多示例使其更好
proposition_examples = [
    {"document":
        "1969 年，Neil Armstrong 在 Apollo 11 任务期间成为第一个在月球上行走的人。",
     "propositions":
        "['Neil Armstrong 是一名宇航员。', 'Neil Armstrong 于 1969 年在月球上行走。', 'Neil Armstrong 是第一个在月球上行走的人。', 'Neil Armstrong 在 Apollo 11 任务期间在月球上行走。', 'Apollo 11 任务发生于 1969 年。']"
    },
]

example_proposition_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{document}"),
        ("ai", "{propositions}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt = example_proposition_prompt,
    examples = proposition_examples,
)
```

> **💡 代码解释**
> - **Few-Shot Learning**：给模型几个例子，让它学会"照猫画虎"
> - 这里我们展示了一个例子：一句话如何拆成 5 个命题
> - 例子越多，模型理解得越好
>
> **📊 术语解释**
> - **Few-Shot**：少样本学习，给模型少量示例让它理解任务

### 系统提示词

```python
# 提示
system = """请将以下文本分解为简单、自包含的命题。确保每个命题符合以下标准：

    1. 表达单一事实：每个命题应陈述一个具体事实或主张。
    2. 无需上下文即可理解：命题应是自包含的，意味着无需额外上下文即可理解。
    3. 使用全名，不使用代词：避免代词或模糊引用；使用全实体名称。
    4. 包含相关日期/限定词：如果适用，包含必要的日期、时间和限定词以使事实精确。
    5. 包含一个主谓关系：专注于单一主体及其对应的动作或属性，不使用连词或多个从句。"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        few_shot_prompt,
        ("human", "{document}"),
    ]
)

proposition_generator = prompt | structured_llm
```

> **💡 代码解释**
> - 系统提示词定义了命题的 5 个标准
> - `prompt | structured_llm`：创建一个链，先构造提示再调用 LLM
>
> **📊 术语解释**
> - **自包含**：命题自己能说清楚，不需要看上下文
>   - ❌ 不好："他是第一个登月的人"（"他"是谁？）
>   - ✅ 好："Neil Armstrong 是第一个在月球上行走的人"

### 批量生成命题

```python
propositions = []  # 存储文档中的所有命题

for i in range(len(doc_splits)):
    response = proposition_generator.invoke({"document": doc_splits[i].page_content})  # 创建命题
    for proposition in response.propositions:
        propositions.append(Document(page_content=proposition, metadata={"Title": "Paul Graham 的 Founder Mode 文章", "Source": "https://www.perplexity.ai/page/paul-graham-s-founder-mode-ess-t9TCyvkqRiyMQJWsHr0fnQ", "chunk_id": i+1}))
```

> **💡 代码解释**
> - 遍历所有文本块
> - 对每个块调用 LLM 生成命题
> - 把生成的命题存成 `Document` 对象，保留元数据（标题、来源、块 ID）
>
> **⚠️ 新手注意**
> - 这个过程可能需要几十秒，因为要多次调用 LLM
> - 如果文档很长，可以考虑加进度条：`from tqdm import tqdm`

---

## ✅ 第五步：质量检查 (Quality Check)

### 📖 这是什么？

LLM 生成的命题可能有质量问题。我们用另一个 LLM 来当"质检员"，给每个命题打分。

### 💻 完整代码

```python
# 数据模型
class GradePropositions(BaseModel):
    """对给定命题的准确性、清晰度、完整性和简洁性进行评分"""

    accuracy: int = Field(
        description="根据命题反映原文的程度，评分 1-10。"
    )

    clarity: int = Field(
        description="根据命题在无额外上下文情况下的易理解程度，评分 1-10。"
    )

    completeness: int = Field(
        description="根据命题是否包含必要细节（如日期、限定词），评分 1-10。"
    )

    conciseness: int = Field(
        description="根据命题是否简洁且不丢失重要信息，评分 1-10。"
    )

# 使用函数调用的 LLM
llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
structured_llm= llm.with_structured_output(GradePropositions)
```

> **💡 代码解释**
> - 定义 4 个评分维度：
>   - **准确性**：命题是否忠实于原文
>   - **清晰度**：是否容易理解
>   - **完整性**：是否包含必要细节
>   - **简洁性**：是否简洁不冗余

### 评估提示词

```python
# 提示
evaluation_prompt_template = """
请根据以下标准评估以下命题：
- **准确性**：根据命题反映原文的程度，评分 1-10。
- **清晰度**：根据命题在无额外上下文情况下的易理解程度，评分 1-10。
- **完整性**：根据命题是否包含必要细节（如日期、限定词），评分 1-10。
- **简洁性**：根据命题是否简洁且不丢失重要信息，评分 1-10。

示例：
Docs: 1969 年，Neil Armstrong 在 Apollo 11 任务期间成为第一个在月球上行走的人。

Propositons_1: Neil Armstrong 是一名宇航员。
Evaluation_1: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

Propositons_2: Neil Armstrong 于 1969 年在月球上行走。
Evaluation_2: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

Propositons_3: Neil Armstrong 是第一个在月球上行走的人。
Evaluation_3: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

Propositons_4: Neil Armstrong 在 Apollo 11 任务期间在月球上行走。
Evaluation_4: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

Propositons_5: Apollo 11 任务发生于 1969 年。
Evaluation_5: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

格式：
Proposition: "{proposition}"
Original Text: "{original_text}"
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", evaluation_prompt_template),
        ("human", "{proposition}, {original_text}"),
    ]
)

proposition_evaluator = prompt | structured_llm
```

### 执行质量检查

```python
# 定义评估类别和阈值
evaluation_categories = ["accuracy", "clarity", "completeness", "conciseness"]
thresholds = {"accuracy": 7, "clarity": 7, "completeness": 7, "conciseness": 7}

# 评估命题的函数
def evaluate_proposition(proposition, original_text):
    response = proposition_evaluator.invoke({"proposition": proposition, "original_text": original_text})

    # 解析响应以提取分数
    scores = {"accuracy": response.accuracy, "clarity": response.clarity, "completeness": response.completeness, "conciseness": response.conciseness}
    return scores

# 检查命题是否通过质量检查
def passes_quality_check(scores):
    for category, score in scores.items():
        if score < thresholds[category]:
            return False
    return True

evaluated_propositions = []  # 存储文档中所有评估后的命题

# 遍历生成的命题并评估它们
for idx, proposition in enumerate(propositions):
    scores = evaluate_proposition(proposition.page_content, doc_splits[proposition.metadata['chunk_id'] - 1].page_content)
    if passes_quality_check(scores):
        # 命题通过质量检查，保留它
        evaluated_propositions.append(proposition)
    else:
        # 命题未通过，丢弃或标记以进一步审查
        print(f"{idx+1}) 命题：{proposition.page_content} \n 分数：{scores}")
        print("未通过")
```

> **💡 代码解释**
> - `thresholds`：设定及格线，所有维度都要≥7 分
> - `passes_quality_check`：检查命题是否所有维度都及格
> - 通过的命题保留，不通过的打印出来（可以选择丢弃或人工复核）
>
> **⚠️ 新手注意**
> - 阈值可以根据你的需求调整
> - 要求严格就调高阈值（如 8），要求宽松就调低（如 6）

---

## 🔍 第六步：将命题嵌入向量存储

### 📖 这是什么？

把通过质量检查的命题转换成向量（一串数字），存到 FAISS 向量数据库中。这样就能用数学方法快速找到相似的命题了。

### 💻 完整代码

```python
# 添加到向量存储
vectorstore_propositions = FAISS.from_documents(evaluated_propositions, embedding_model)
retriever_propositions = vectorstore_propositions.as_retriever(
                search_type="similarity",
                search_kwargs={'k': 4},  # 要检索的文档数量
            )
```

> **💡 代码解释**
> - `FAISS.from_documents`：把命题文档转成向量并存储
> - `as_retriever`：创建一个检索器对象
> - `search_type="similarity"`：使用相似性搜索
> - `k=4`：每次检索返回最相似的 4 个结果

### 测试检索效果

```python
query = "谁的管理方法成为 Brian Chesky 在 Airbnb 实施'Founder Mode'的灵感来源？"
res_proposition = retriever_propositions.invoke(query)
```

```python
for i, r in enumerate(res_proposition):
    print(f"{i+1}) 内容：{r.page_content} --- 块 ID: {r.metadata['chunk_id']}")
```

> **💡 预期输出示例**
> ```
> 1) 内容：Steve Jobs 在 Apple 的管理方法成为 Brian Chesky 在 Airbnb 实施"Founder Mode"的灵感来源。--- 块 ID: 3
> 2) 内容：Steve Jobs 每年为 Apple 最重要的 100 人举办 retreat。--- 块 ID: 3
> ...
> ```

---

## 📊 第七步：与较大块的性能比较

### 📖 这是什么？

我们创建一个使用原始大块（非命题）的检索系统，看看命题分块和普通分块哪个更好用。

### 💻 完整代码

```python
# 添加到向量存储_larger_
vectorstore_larger = FAISS.from_documents(doc_splits, embedding_model)
retriever_larger = vectorstore_larger.as_retriever(
                search_type="similarity",
                search_kwargs={'k': 4},  # 要检索的文档数量
            )
```

```python
res_larger = retriever_larger.invoke(query)
```

```python
for i, r in enumerate(res_larger):
    print(f"{i+1}) 内容：{r.page_content} --- 块 ID: {r.metadata['chunk_id']}")
```

> **💡 预期输出示例**
> ```
> 1) 内容：Steve Jobs 在 Apple 的管理方法成为 Brian Chesky 在 Airbnb 实施"Founder Mode"的灵感来源。一个值得注意的做法是 Jobs 每年为 Apple 最重要的 100 人举办 retreat...--- 块 ID: 3
> ```
>
> **注意**：大块检索返回的是整段文本，而命题检索返回的是精确的事实陈述

---

## 🧪 第八步：更多测试用例

### 测试 1

```python
test_query_1 = "文章'Founder Mode'是关于什么的？"
res_proposition = retriever_propositions.invoke(test_query_1)
res_larger = retriever_larger.invoke(test_query_1)
```

```python
for i, r in enumerate(res_proposition):
    print(f"{i+1}) 内容：{r.page_content} --- 块 ID: {r.metadata['chunk_id']}")
```

```python
for i, r in enumerate(res_larger):
    print(f"{i+1}) 内容：{r.page_content} --- 块 ID: {r.metadata['chunk_id']}")
```

### 测试 2

```python
test_query_2 = "谁是 Airbnb 的联合创始人？"
res_proposition = retriever_propositions.invoke(test_query_2)
res_larger = retriever_larger.invoke(test_query_2)
```

```python
for i, r in enumerate(res_proposition):
    print(f"{i+1}) 内容：{r.page_content} --- 块 ID: {r.metadata['chunk_id']}")
```

```python
for i, r in enumerate(res_larger):
    print(f"{i+1}) 内容：{r.page_content} --- 块 ID: {r.metadata['chunk_id']}")
```

### 测试 3

```python
test_query_3 = "文章'founder mode'是什么时候发表的？"
res_proposition = retriever_propositions.invoke(test_query_3)
res_larger = retriever_larger.invoke(test_query_3)
```

```python
for i, r in enumerate(res_proposition):
    print(f"{i+1}) 内容：{r.page_content} --- 块 ID: {r.metadata['chunk_id']}")
```

```python
for i, r in enumerate(res_larger):
    print(f"{i+1}) 内容：{r.page_content} --- 块 ID: {r.metadata['chunk_id']}")
```

---

## 📈 总结对比

| **方面** | **基于命题的检索** | **简单块检索** |
|-----------|-------------------|----------------|
| **响应精度** | 高：提供聚焦且直接的答案 | 中：提供更多上下文但可能包含无关信息 |
| **清晰度和简洁性** | 高：清晰简洁，避免不必要的细节 | 中：更全面但可能令人不知所措 |
| **上下文丰富度** | 低：可能缺乏上下文，专注于特定命题 | 高：提供额外的上下文和细节 |
| **全面性** | 低：可能省略更广泛的上下文或补充细节 | 高：提供更完整的视图和丰富的信息 |
| **叙述流畅性** | 中：可能碎片化或不连贯 | 高：保持原文的逻辑流畅性和连贯性 |
| **信息过载** | 低：不太可能因过多信息而令人不知所措 | 高：有过多的信息让用户不知所措的风险 |
| **用例适用性** | 最适合快速、事实性查询 | 最适合需要深入理解的复杂查询 |
| **效率** | 高：提供快速、有针对性的响应 | 中：可能需要更多精力筛选额外内容 |
| **特异性** | 高：精确且有针对性的响应 | 中：由于包含更广泛的上下文，答案可能针对性较低 |

---

## ❓ 常见问题 FAQ

### Q1：命题分块适合所有场景吗？
**A**：不是。命题分块特别适合：
- ✅ 事实性查询（谁、什么时候、什么）
- ✅ 需要精确答案的场景
- ❌ 不适合需要大量上下文的复杂分析

### Q2：为什么要做质量检查？
**A**：LLM 可能生成：
- 不准确的命题（歪曲原意）
- 不清晰的命题（有代词指代不明）
- 不完整的命题（缺少关键信息）
- 质量检查可以过滤掉这些"次品"

### Q3：可以用其他 embedding 模型吗？
**A**：可以！教程用的是 Ollama 的本地模型，你也可以用：
- OpenAI Embeddings
- Cohere Embeddings
- HuggingFace Embeddings

### Q4：命题分块会很慢吗？
**A**：相比普通分块，确实会慢一些，因为：
1. 要调用 LLM 生成命题
2. 要调用 LLM 做质量检查
3. 要对更多命题（而不是块）进行 embedding
但检索质量通常值得这点时间开销。

### Q5：我该怎么选择 chunk_size？
**A**：一般建议：
- 200-500 字符：适合命题生成
- 500-1000 字符：适合普通检索
具体要根据你的文档类型和查询模式调整。

---

## 🎉 恭喜你学完了！

现在你已经掌握了：
1. ✅ 命题分块的核心概念
2. ✅ 完整的实现流程
3. ✅ 与普通分块的对比

**下一步建议**：
- 用自己的文档尝试这个流程
- 调整参数看效果变化
- 结合其他 RAG 技术（如 HyDE）进一步提升

---

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--proposition-chunking)
