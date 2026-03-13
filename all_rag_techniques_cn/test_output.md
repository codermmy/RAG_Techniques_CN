# 包安装和导入

下面的单元格安装运行此笔记本所需的所有必要包。


```python
# 安装所需的包
!pip install llama-index openai python-dotenv
```

```python
import nest_asyncio
import random

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
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
```

### 读取文档

```python
data_dir = "../data"
documents = SimpleDirectoryReader(data_dir).load_data()
```

### 创建评估问题并从中选取 k 个

```python
num_eval_questions = 25

eval_documents = documents[0:20]
data_generator = DatasetGenerator.from_documents(eval_documents)
eval_questions = data_generator.generate_questions_from_nodes()
k_eval_questions = random.sample(eval_questions, num_eval_questions)
```

### 定义指标评估器并修改 llama_index faithfulness 评估器提示以依赖上下文 

```python
# 我们将使用 GPT-4 来评估响应
gpt4 = OpenAI(temperature=0, model="gpt-4o")

# 为 LLM 设置适当的配置
Settings.llm = gpt4

# 定义基于 GPT-4 的 Faithfulness 评估器
faithfulness_gpt4 = FaithfulnessEvaluator()

faithfulness_new_prompt_template = PromptTemplate(""" Please tell if a given piece of information is directly supported by the context.
    You need to answer with either YES or NO.
    Answer YES if any part of the context explicitly supports the information, even if most of the context is unrelated. If the context does not explicitly support the information, answer NO. Some examples are provided below.

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

faithfulness_gpt4.update_prompts({"your_prompt_key": faithfulness_new_prompt_template}) # 用新的提示模板更新 prompts 字典

# 定义基于 GPT-4 的 Relevancy 评估器
relevancy_gpt4 = RelevancyEvaluator()
```

### 评估每个块大小指标的函数

```python
# 定义函数来计算给定块大小的平均响应时间、平均 faithfulness 和平均 relevancy 指标
# 我们使用 GPT-3.5-Turbo 来生成响应，使用 GPT-4 来评估它
def evaluate_response_time_and_accuracy(chunk_size, eval_questions):
    """
    评估 GPT-3.5-turbo 为给定块大小生成的响应的平均响应时间、faithfulness 和 relevancy。
    
    参数：
    chunk_size (int): 正在处理的数据块大小。
    
    返回：
    tuple: 包含平均响应时间、faithfulness 和 relevancy 指标的元组。
    """

    total_response_time = 0
    total_faithfulness = 0
    total_relevancy = 0

    # 创建向量索引
    llm = OpenAI(model="gpt-3.5-turbo")

    Settings.llm = llm
    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = chunk_size // 5 

    vector_index = VectorStoreIndex.from_documents(eval_documents)
    
    # 构建查询引擎
    query_engine = vector_index.as_query_engine(similarity_top_k=5)
    num_questions = len(eval_questions)

    # 遍历 eval_questions 中的每个问题来计算指标。
    # 虽然 BatchEvalRunner 可用于更快的评估（参见：https://docs.llamaindex.ai/en/latest/examples/evaluation/batch_eval.html），
    # 但我们在这里使用循环来专门测量不同块大小的响应时间。
    for question in eval_questions:
        start_time = time.time()
        response_vector = query_engine.query(question)
        elapsed_time = time.time() - start_time
        
        faithfulness_result = faithfulness_gpt4.evaluate_response(
            response=response_vector
        ).passing
        
        relevancy_result = relevancy_gpt4.evaluate_response(
            query=question, response=response_vector
        ).passing

        total_response_time += elapsed_time
        total_faithfulness += faithfulness_result
        total_relevancy += relevancy_result

    average_response_time = total_response_time / num_questions
    average_faithfulness = total_faithfulness / num_questions
    average_relevancy = total_relevancy / num_questions

    return average_response_time, average_faithfulness, average_relevancy
```

### 测试不同的块大小 

```python
chunk_sizes = [128, 256]

for chunk_size in chunk_sizes:
  avg_response_time, avg_faithfulness, avg_relevancy = evaluate_response_time_and_accuracy(chunk_size, k_eval_questions)
  print(f"Chunk size {chunk_size} - Average Response time: {avg_response_time:.2f}s, Average Faithfulness: {avg_faithfulness:.2f}, Average Relevancy: {avg_relevancy:.2f}")
```

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--choose-chunk-size)
