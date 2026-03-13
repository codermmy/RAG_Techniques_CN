# RAG 系统中用于改进检索的查询转换技术

## 概述

本代码实现了三种查询转换技术，以增强检索增强生成 (RAG) 系统中的检索过程：

1. 查询重写
2. 后退提示
3. 子查询分解

每种技术都旨在通过修改或扩展原始查询来提高检索信息的相关性和全面性。

## 动机

RAG 系统在检索最相关信息时经常面临挑战，尤其是在处理复杂或模糊的查询时。这些查询转换技术通过重新表述查询以更好地匹配相关文档或检索更全面的信息来解决这个问题。

## 关键组件

1. 查询重写：将查询重新表述为更具体和详细的查询。
2. 后退提示：生成更广泛的查询以更好地检索上下文。
3. 子查询分解：将复杂查询分解为更简单的子查询。

## 方法详情

### 1. 查询重写

- **目的**：使查询更具体和详细，提高检索相关信息的可能性。
- **实现**：
  - 使用带有自定义提示模板的 GPT-4 模型。
  - 获取原始查询并将其重新表述为更具体和详细的查询。

### 2. 后退提示

- **目的**：生成更广泛、更一般的查询，以帮助检索相关的背景信息。
- **实现**：
  - 使用带有自定义提示模板的 GPT-4 模型。
  - 获取原始查询并生成更一般的"后退"查询。

### 3. 子查询分解

- **目的**：将复杂查询分解为更简单的子查询，以实现更全面的信息检索。
- **实现**：
  - 使用带有自定义提示模板的 GPT-4 模型。
  - 将原始查询分解为 2-4 个更简单的子查询。

## 这些方法的好处

1. **提高相关性**：查询重写有助于检索更具体和相关的信息。
2. **更好的上下文**：后退提示允许检索更广泛的上下文和背景信息。
3. **全面的结果**：子查询分解能够检索涵盖复杂查询不同方面的信息。
4. **灵活性**：每种技术可以根据具体用例独立使用或组合使用。

## 实现细节

- 所有技术都使用 OpenAI 的 GPT-4 模型进行查询转换。
- 使用自定义提示模板来指导模型生成适当的转换。
- 代码为每种转换技术提供了单独的函数，便于集成到现有的 RAG 系统中。

## 示例用例

代码使用示例查询演示每种技术：
"气候变化对环境有什么影响？"

- **查询重写**将其扩展为包括温度变化和生物多样性等具体方面。
- **后退提示**将其概括为"气候变化的一般影响是什么？"
- **子查询分解**将其分解为有关生物多样性、海洋、天气模式和陆地环境的问题。

## 结论

这些查询转换技术提供了增强 RAG 系统检索能力的强大方法。通过以各种方式重新表述查询，它们可以显著提高检索信息的相关性、上下文和全面性。这些方法在查询可能复杂或多方面的领域特别有价值，例如科学研究、法律分析或全面的事实调查任务。

# 包安装和导入

下面的单元格安装运行此笔记本所需的所有包。


```python
# 安装所需的包
!pip install langchain langchain-openai python-dotenv
```

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

### 1 - 查询重写：重新表述查询以改进检索。

```python
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

### 演示用例

```python
# 关于理解气候变化数据集的示例查询
original_query = "气候变化对环境有什么影响？"
rewritten_query = rewrite_query(original_query)
print("原始查询:", original_query)
print("\n改写后的查询:", rewritten_query)
```

### 2 - 后退提示：生成更广泛的查询以更好地检索上下文。



```python
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

### 演示用例

```python
# 关于理解气候变化数据集的示例查询
original_query = "气候变化对环境有什么影响？"
step_back_query = generate_step_back_query(original_query)
print("原始查询:", original_query)
print("\n后退查询:", step_back_query)
```

### 3- 子查询分解：将复杂查询分解为更简单的子查询。

```python
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

### 演示用例

```python
# 关于理解气候变化数据集的示例查询
original_query = "气候变化对环境有什么影响？"
sub_queries = decompose_query(original_query)
print("\n子查询：")
for i, sub_query in enumerate(sub_queries, 1):
    print(sub_query)
```

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--query-transformations)
