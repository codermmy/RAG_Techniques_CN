# 🌟 新手入门：查询转换技术 (Query Transformations)

> **💡 给新手的说明**
> - **难度等级**：⭐⭐⭐（中级）
> - **预计学习时间**：30-40 分钟
> - **前置知识**：了解基础的 Python 编程，对 RAG 系统有基本认识
> - **本教程你将学会**：如何通过转换用户查询来提升 RAG 系统的检索效果

---

## 📖 核心概念理解

### 什么是查询转换？

想象你在图书馆找书：
- **原始查询**：你对图书管理员说"我想了解气候变化的影响"
- **问题**：这个表达太模糊了，管理员不知道你想要什么

**查询转换**就像一个"翻译官"，把你的模糊问题"翻译"成更容易找到答案的形式：
1. **查询重写**："请告诉我气候变化对温度、生物多样性和极端天气的具体影响"
2. **后退提示**："气候变化的一般影响有哪些？"（先找大框架）
3. **子查询分解**：拆成 4 个小问题分别检索

### 通俗理解

**生活化比喻**：
- 🔍 **原始查询**：就像在 Google 里随便输入几个词
- ✨ **查询转换后**：就像专业的图书管理员帮你整理的精准搜索关键词

### 为什么需要查询转换？

| 问题类型 | 原始查询 | 转换后查询 | 效果提升 |
|---------|---------|-----------|---------|
| 太模糊 | "气候变化的影响" | "气候变化对生物多样性、海洋、农业的具体影响" | ✅ 更具体 |
| 太复杂 | "气候变化如何影响环境并且对人类有什么后果" | 拆成 2 个问题分别检索 | ✅ 更清晰 |
| 缺上下文 | "这个理论是谁提出的" | "气候变化的主要理论有哪些，谁提出的" | ✅ 更完整 |

---

## 🛠️ 第一步：环境准备

### 📖 这是什么？

安装必要的 Python 包，并配置 API 密钥。

### 💻 完整代码

```python
# 安装所需的包
# !pip install langchain langchain-openai python-dotenv
```

> **💡 代码解释**
> - `langchain`：构建 LLM 应用的核心框架
> - `langchain-openai`：LangChain 的 OpenAI 集成包
> - `python-dotenv`：用于安全地加载环境变量
>
> **⚠️ 新手注意**
> - 去掉 `!` 前面的 `#` 注释以在 Jupyter 中运行
> - 或在终端运行：`pip install langchain langchain-openai python-dotenv`

### 导入库并配置环境变量

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

import os
from dotenv import load_dotenv

# 从.env 文件加载环境变量
load_dotenv()

# 设置 OpenAI API 密钥环境变量
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
```

> **💡 代码解释**
> - `ChatOpenAI`：用于调用 OpenAI 的聊天模型（如 GPT-4）
> - `PromptTemplate`：用于创建结构化的提示词模板
> - `load_dotenv()`：从 `.env` 文件读取环境变量
>
> **⚠️ 新手注意**
> - 需要先创建 `.env` 文件，内容：`OPENAI_API_KEY=你的密钥`
> - API 密钥可以从 [OpenAI 官网](https://platform.openai.com/api-keys) 获取
> - **永远不要**把 API 密钥直接写在代码里！

---

## 🔄 第二种：查询重写 (Query Rewriting)

### 📖 这是什么？

**查询重写**就是把用户的原始问题改写成更具体、更详细的版本。

**生活化比喻**：
- 原始查询："苹果怎么样？"
- 重写后："苹果公司的财务状况和市场表现如何？"

这样检索系统就知道你要找的是"公司"而不是"水果"！

### 💻 完整代码

```python
# 创建用于查询重写的 LLM
re_write_llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)

# 创建查询重写的提示模板
query_rewrite_template = """你是一个 AI 助手，任务是重新表述用户查询以改进 RAG 系统中的检索。
给定原始查询，将其改写为更具体、更详细，并可能检索到相关信息的查询。

原始查询：{original_query}

改写后的查询："""

query_rewrite_prompt = PromptTemplate(
    input_variables=["original_query"],
    template=query_rewrite_template
)

# 创建用于查询重写的 LLMChain
query_rewriter = query_rewrite_prompt | re_write_llm

def rewrite_query(original_query):
    """
    改写原始查询以改进检索。

    参数：
    original_query (str): 原始用户查询

    返回：
    str: 改写后的查询
    """
    response = query_rewriter.invoke(original_query)
    return response.content
```

> **💡 代码解释**
> - `ChatOpenAI`：初始化 GPT-4 模型
> - `temperature=0`：输出更稳定，适合事实性任务
> - `PromptTemplate`：定义提示词模板，`{original_query}` 是占位符
> - `query_rewriter`：把提示词和 LLM 连接成一条"链"
> - `rewrite_query`：封装好的函数，输入原始查询，返回改写后的查询
>
> **⚠️ 新手注意**
> - `|` 是 LangChain 的链式操作符，相当于 `chain(prompt, llm)`
> - `invoke()` 是调用链的方法
> - `response.content` 提取 LLM 的文本回复

### 演示用例

```python
# 关于理解气候变化数据集的示例查询
original_query = "气候变化对环境有什么影响？"
rewritten_query = rewrite_query(original_query)
print("原始查询:", original_query)
print("\n改写后的查询:", rewritten_query)
```

> **💡 预期输出示例**
> ```
> 原始查询：气候变化对环境有什么影响？
>
> 改写后的查询：气候变化对环境的哪些方面产生了具体影响？包括但不限于温度变化、海平面上升、极端天气事件增加、生物多样性丧失、生态系统破坏等方面的详细影响是什么？
> ```
>
> **看到区别了吗？**
> - 原始查询只有 9 个字
> - 改写后有 60+ 字，列出了具体的影响维度
> - 这样检索系统就能找到更精准的结果！

---

## 🔙 第二种：后退提示 (Step-Back Prompting)

### 📖 这是什么？

**后退提示**与查询重写相反——它生成一个更广泛、更一般的问题，先获取大背景的上下文。

**生活化比喻**：
- 原始问题："这个数学题怎么解？"
- 后退问题："这类数学题的一般解题方法是什么？"

先搞清楚大框架，再回来解决具体问题！

### 💻 完整代码

```python
# 创建用于后退提示的 LLM
step_back_llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)

# 创建后退提示的提示模板
step_back_template = """你是一个 AI 助手，任务是生成更广泛、更一般的查询以改进 RAG 系统中的上下文检索。
给定原始查询，生成一个更一般的后退查询，可以帮助检索相关的背景信息。

原始查询：{original_query}

后退查询："""

step_back_prompt = PromptTemplate(
    input_variables=["original_query"],
    template=step_back_template
)

# 创建用于后退提示的 LLMChain
step_back_chain = step_back_prompt | step_back_llm

def generate_step_back_query(original_query):
    """
    生成后退查询以检索更广泛的上下文。

    参数：
    original_query (str): 原始用户查询

    返回：
    str: 后退查询
    """
    response = step_back_chain.invoke(original_query)
    return response.content
```

> **💡 代码解释**
> - 结构与查询重写几乎一样
> - 区别在于提示词：这里要求生成"更一般"的查询
> - `step_back_chain`：后退提示链

### 演示用例

```python
# 关于理解气候变化数据集的示例查询
original_query = "气候变化对环境有什么影响？"
step_back_query = generate_step_back_query(original_query)
print("原始查询:", original_query)
print("\n后退查询:", step_back_query)
```

> **💡 预期输出示例**
> ```
> 原始查询：气候变化对环境有什么影响？
>
> 后退查询：气候变化的主要影响和后果有哪些？
> ```
>
> **看到区别了吗？**
> - 原始查询具体问"对环境的影响"
> - 后退查询问"主要影响和后果"（更广泛）
> - 这样可以先检索到整体框架，再补充细节

---

## 🔀 第三种：子查询分解 (Sub-Query Decomposition)

### 📖 这是什么？

**子查询分解**把一个复杂问题拆成多个简单的小问题，分别检索后再组合答案。

**生活化比喻**：
- 原始问题："帮我规划一次日本之旅"
- 拆解后：
  1. "日本有哪些值得旅游的城市？"
  2. "去日本旅游的最佳时间是什么时候？"
  3. "日本旅游的预算大概多少？"
  4. "日本有哪些必去的景点？"

拆成小问题后，每个都更容易回答！

### 💻 完整代码

```python
# 创建用于子查询分解的 LLM
sub_query_llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)

# 创建子查询分解的提示模板
subquery_decomposition_template = """你是一个 AI 助手，任务是将复杂查询分解为 RAG 系统的更简单子查询。
给定原始查询，将其分解为 2-4 个更简单的子查询，当一起回答时，将提供对原始查询的全面响应。

原始查询：{original_query}

示例：气候变化对环境有什么影响？

子查询：
1. 气候变化对生物多样性有什么影响？
2. 气候变化如何影响海洋？
3. 气候变化对农业有什么影响？
4. 气候变化对人类健康有什么影响？"""


subquery_decomposition_prompt = PromptTemplate(
    input_variables=["original_query"],
    template=subquery_decomposition_template
)

# 创建用于子查询分解的 LLMChain
subquery_decomposer_chain = subquery_decomposition_prompt | sub_query_llm

def decompose_query(original_query: str):
    """
    将原始查询分解为更简单的子查询。

    参数：
    original_query (str): 原始复杂查询

    返回：
    List[str]: 更简单子查询的列表
    """
    response = subquery_decomposer_chain.invoke(original_query).content
    sub_queries = [q.strip() for q in response.split('\n') if q.strip() and not q.strip().startswith('子查询：')]
    return sub_queries
```

> **💡 代码解释**
> - 提示词中给出了示例（Few-Shot），帮助模型理解输出格式
> - `decompose_query`：输入复杂查询，返回子查询列表
> - `split('\n')`：按换行符分割，提取每个子查询
> - `not q.strip().startswith('子查询：')`：过滤掉标题行
>
> **⚠️ 新手注意**
> - 返回的是一个**列表**，包含 2-4 个子查询
> - 实际使用时，你需要对每个子查询分别检索，然后合并结果

### 演示用例

```python
# 关于理解气候变化数据集的示例查询
original_query = "气候变化对环境有什么影响？"
sub_queries = decompose_query(original_query)
print("\n子查询：")
for i, sub_query in enumerate(sub_queries, 1):
    print(sub_query)
```

> **💡 预期输出示例**
> ```
> 子查询：
> 1. 气候变化对生物多样性有什么影响？
> 2. 气候变化如何影响海洋？
> 3. 气候变化对天气模式有什么影响？
> 4. 气候变化对陆地环境有什么影响？
> ```
>
> **看到区别了吗？**
> - 原始问题是一个大问题
> - 拆成 4 个具体的小问题，每个都更容易检索到精准答案

---

## 📊 三种技术对比总结

| 技术 | 作用 | 适用场景 | 查询变化 |
|------|------|---------|---------|
| **查询重写** | 让查询更具体 | 查询太模糊时 | 短→长，抽象→具体 |
| **后退提示** | 先获取大背景 | 需要上下文时 | 具体→一般 |
| **子查询分解** | 拆分复杂问题 | 查询太复杂时 | 1 个→多个 |

### 使用建议

```
┌─────────────────────────────────────────┐
│          用户输入原始查询               │
└─────────────────┬───────────────────────┘
                  │
         ┌────────▼────────┐
         │  查询是否模糊？  │
         └───────┬─────────┘
            是   │   否
                 │
        ┌────────▼────────┐
        │  使用查询重写    │
        └─────────────────┘

         ┌────────▼────────┐
         │  查询是否复杂？  │
         └───────┬─────────┘
            是   │   否
                 │
        ┌────────▼────────┐
        │  使用子查询分解  │
        └─────────────────┘

         ┌────────▼────────┐
         │  需要上下文吗？  │
         └───────┬─────────┘
            是   │   否
                 │
        ┌────────▼────────┐
        │  使用后退提示    │
        └─────────────────┘
```

---

## 🧪 实战：完整 RAG 流程集成

### 📖 这是什么？

把查询转换技术集成到完整的 RAG 流程中。

### 💻 完整代码示例

```python
# 假设你已经有了向量存储和检索器
# from your_setup import retriever

def advanced_rag_query(original_query, strategy="rewrite"):
    """
    使用查询转换技术的增强 RAG 检索

    参数：
    original_query (str): 原始用户查询
    strategy (str): 转换策略，可选 "rewrite", "step_back", "decompose"

    返回：
    list: 检索到的相关文档
    """
    if strategy == "rewrite":
        # 策略 1：查询重写
        transformed_query = rewrite_query(original_query)
        print(f"改写后的查询：{transformed_query}")

    elif strategy == "step_back":
        # 策略 2：后退提示
        transformed_query = generate_step_back_query(original_query)
        print(f"后退查询：{transformed_query}")

    elif strategy == "decompose":
        # 策略 3：子查询分解
        sub_queries = decompose_query(original_query)
        print(f"子查询：{sub_queries}")

        # 对每个子查询检索并合并结果
        all_results = []
        for sub_q in sub_queries:
            results = retriever.invoke(sub_q)
            all_results.extend(results)

        # 去重（可选）
        unique_results = []
        seen_content = set()
        for doc in all_results:
            if doc.page_content not in seen_content:
                unique_results.append(doc)
                seen_content.add(doc.page_content)

        return unique_results

    # 使用转换后的查询进行检索
    results = retriever.invoke(transformed_query)
    return results

# 测试
query = "气候变化对环境有什么影响？"
print("=== 查询重写 ===")
rewrite_results = advanced_rag_query(query, strategy="rewrite")

print("\n=== 后退提示 ===")
step_back_results = advanced_rag_query(query, strategy="step_back")

print("\n=== 子查询分解 ===")
decompose_results = advanced_rag_query(query, strategy="decompose")
```

> **💡 代码解释**
> - `advanced_rag_query`：统一的入口函数
> - 根据 `strategy` 参数选择不同的转换策略
> - 子查询分解需要特殊处理：分别检索每个子查询，然后合并结果
> - 去重逻辑避免返回重复的文档

---

## ❓ 常见问题 FAQ

### Q1：我应该选择哪种转换技术？
**A**：根据你的查询特点：
- 查询太短/模糊 → **查询重写**
- 需要背景知识 → **后退提示**
- 查询包含多个问题 → **子查询分解**
- 不确定？可以都试试，看哪个效果最好！

### Q2：可以组合使用多种技术吗？
**A**：可以！例如：
1. 先用后退提示获取上下文
2. 再用查询重写细化具体问题
3. 最后用子查询分解处理复杂部分

### Q3：temperature 应该设多少？
**A**：
- 查询转换任务建议用 `temperature=0`（稳定、可预测）
- 如果需要更有创意的改写，可以用 `0.3-0.5`
- 不要用太高（>0.7），可能导致偏离原意

### Q4：可以用其他模型吗（不是 GPT-4）？
**A**：当然可以！教程代码稍作修改就能用：
- Claude（Anthropic）
- Llama（通过 Ollama 或 Groq）
- 文心一言、通义千问等国产模型

### Q5：查询转换会增加多少延迟？
**A**：
- 每次转换需要调用一次 LLM
- GPT-4 通常响应时间在 1-3 秒
- 子查询分解最慢（可能需要调用多次 LLM）
- 如果延迟是问题，可以考虑更小的模型（如 GPT-3.5）

---

## 🎯 进阶技巧

### 技巧 1：自定义提示词

你可以根据具体领域定制提示词：

```python
# 医疗领域的查询重写
medical_rewrite_template = """你是一个医学信息检索专家。
将患者的问题改写为专业的医学查询，包括：
- 疾病名称（使用医学术语）
- 相关症状
- 可能的诊断方法

原始查询：{original_query}

改写后的查询："""
```

### 技巧 2：添加过滤条件

```python
def rewrite_query_with_filters(original_query, date_range=None, topics=None):
    """带过滤条件的查询重写"""
    filters = []
    if date_range:
        filters.append(f"时间范围：{date_range}")
    if topics:
        filters.append(f"主题：{', '.join(topics)}")

    filter_str = "\n".join(filters)

    template = f"""改写查询，并考虑以下约束：
{filter_str}

原始查询：{{original_query}}

改写后的查询："""

    # ... 继续处理
```

### 技巧 3：记录日志便于调试

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rewrite_query(original_query):
    response = query_rewriter.invoke(original_query)
    result = response.content

    # 记录日志
    logger.info(f"原始查询：{original_query}")
    logger.info(f"改写后：{result}")

    return result
```

---

## 🎉 恭喜你学完了！

现在你已经掌握了：
1. ✅ 三种查询转换技术的原理
2. ✅ 完整的代码实现
3. ✅ 如何选择适合的技术

**下一步建议**：
- 用你的实际查询测试这三种技术
- 根据效果调整提示词
- 考虑组合使用多种技术

---

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--query-transformations)
