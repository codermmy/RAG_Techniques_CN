# RAG 系统与反馈循环：提升检索和响应质量

## 概述

本系统实现了一种带有集成反馈循环的检索增强生成（RAG）方法。它旨在通过整合用户反馈和动态调整检索过程，随时间提高响应的质量和相关性。

## 动机

传统 RAG 系统由于检索过程或底层知识库的局限性，有时会产生不一致或不相关的响应。通过实现反馈循环，我们可以：

1. 持续提高检索文档的质量
2. 增强生成响应的相关性
3. 随时间使系统适应用户偏好和需求

## 关键组件

1. **PDF 内容提取**：从 PDF 文档中提取文本
2. **向量存储**：存储和索引文档嵌入以进行高效检索
3. **检索器**：根据用户查询获取相关文档
4. **语言模型**：使用检索到的文档生成响应
5. **反馈收集**：收集用户对响应质量和相关性的反馈
6. **反馈存储**：持久化用户反馈以供将来使用
7. **相关性分数调整**：根据反馈修改文档相关性
8. **索引微调**：定期使用累积的反馈更新向量存储

## 方法详情

### 1. 初始设置
- 系统读取 PDF 内容并创建向量存储
- 使用向量存储初始化检索器
- 设置语言模型（LLM）用于响应生成

### 2. 查询处理
- 当用户提交查询时，检索器获取相关文档
- LLM 根据检索到的文档生成响应

### 3. 反馈收集
- 系统收集用户对响应相关性和质量的反馈
- 反馈存储在 JSON 文件中以保持持久性

### 4. 相关性分数调整
- 对于后续查询，系统加载之前的反馈
- LLM 评估过去反馈与当前查询的相关性
- 根据此评估调整文档相关性分数

### 5. 检索器更新
- 使用调整后的文档分数更新检索器
- 确保未来的检索受益于过去的反馈

### 6. 定期索引微调
- 系统定期微调索引
- 高质量反馈用于创建额外的文档
- 向量存储使用这些新文档更新，提高整体检索质量

## 此方法的优势

1. **持续改进**：系统从每次交互中学习，逐步提升性能
2. **个性化**：通过整合用户反馈，系统可以随时间适应个人或群体偏好
3. **提高相关性**：反馈循环有助于在未来的检索中优先考虑更相关的文档
4. **质量控制**：随着系统演进，不太可能重复低质量或不相关的响应
5. **适应性**：系统可以随时间适应用户需求或文档内容的变化

## 结论

这个带有反馈循环的 RAG 系统代表了相对于传统 RAG 实现的显著进步。通过持续从用户交互中学习，它提供了一种更加动态、适应性和以用户为中心的信息检索和响应生成方法。该系统在信息准确性和相关性至关重要的领域，以及用户需求可能随时间演变的场景中尤其有价值。

虽然与传统 RAG 系统相比，实现增加了复杂性，但在响应质量和用户满意度方面的优势使其成为需要高质量、上下文感知信息检索和生成的应用程序的值得投资。

<div style="text-align: center;">

<img src="../images/retrieval_with_feedback_loop.svg" alt="retrieval with feedback loop" style="width:40%; height:auto;">
</div>

# 包安装和导入

下面的单元格安装运行此 notebook 所需的所有必要包。

```python
# 安装所需的包
!pip install langchain langchain-openai python-dotenv
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
import os
import sys
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import json
from typing import List, Dict, Any


# 原始路径追加已替换为 Colab 兼容性
from helper_functions import *
from evaluation.evalute_rag import *

# 从.env 文件加载环境变量
load_dotenv()

# 设置 OpenAI API 密钥环境变量
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
```

### 定义文档路径

```python
# 下载所需的数据文件
import os
os.makedirs('data', exist_ok=True)

# 下载本 notebook 使用的 PDF 文档
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
!wget -O data/feedback_data.json https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/feedback_data.json
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
```

```python
path = "data/Understanding_Climate_Change.pdf"
```

### 创建向量存储和检索 QA 链

```python
content = read_pdf_to_string(path)
vectorstore = encode_from_string(content)
retriever = vectorstore.as_retriever()

llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
```

### 将用户反馈格式化为字典的函数

```python
def get_user_feedback(query, response, relevance, quality, comments=""):
    return {
        "query": query,
        "response": response,
        "relevance": int(relevance),
        "quality": int(quality),
        "comments": comments
    }
```

### 将反馈存储到 JSON 文件的函数

```python
def store_feedback(feedback):
    with open("data/feedback_data.json", "a") as f:
        json.dump(feedback, f)
        f.write("\n")
```

### 读取反馈文件的函数

```python
def load_feedback_data():
    feedback_data = []
    try:
        with open("data/feedback_data.json", "r") as f:
            for line in f:
                feedback_data.append(json.loads(line.strip()))
    except FileNotFoundError:
        print("No feedback data file found. Starting with empty feedback.")
    return feedback_data
```

### 根据反馈文件调整文件相关性的函数

```python
class Response(BaseModel):
    answer: str = Field(..., title="The answer to the question. The options can be only 'Yes' or 'No'")

def adjust_relevance_scores(query: str, docs: List[Any], feedback_data: List[Dict[str, Any]]) -> List[Any]:
    # 创建相关性检查的提示模板
    relevance_prompt = PromptTemplate(
        input_variables=["query", "feedback_query", "doc_content", "feedback_response"],
        template="""
        确定以下反馈响应是否与当前查询和文档内容相关。
        您还提供了用于生成反馈响应的反馈原始查询。
        当前查询：{query}
        反馈查询：{feedback_query}
        文档内容：{doc_content}
        反馈响应：{feedback_response}

        此反馈是否相关？请仅回答'Yes'或'No'。
        """
    )
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)

    # 创建相关性检查的 LLMChain
    relevance_chain = relevance_prompt | llm.with_structured_output(Response)

    for doc in docs:
        relevant_feedback = []

        for feedback in feedback_data:
            # 使用 LLM 检查相关性
            input_data = {
                "query": query,
                "feedback_query": feedback['query'],
                "doc_content": doc.page_content[:1000],
                "feedback_response": feedback['response']
            }
            result = relevance_chain.invoke(input_data).answer

            if result == 'yes':
                relevant_feedback.append(feedback)

        # 根据反馈调整相关性分数
        if relevant_feedback:
            avg_relevance = sum(f['relevance'] for f in relevant_feedback) / len(relevant_feedback)
            doc.metadata['relevance_score'] *= (avg_relevance / 3)  # 假设 1-5 的评分标准，3 为中性

    # 根据调整后的分数重新排序文档
    return sorted(docs, key=lambda x: x.metadata['relevance_score'], reverse=True)
```

### 微调向量索引以包含获得良好反馈的查询和答案的函数

```python
def fine_tune_index(feedback_data: List[Dict[str, Any]], texts: List[str]) -> Any:
    # 过滤高质量回复
    good_responses = [f for f in feedback_data if f['relevance'] >= 4 and f['quality'] >= 4]

    # 提取查询和回复，创建新文档
    additional_texts = []
    for f in good_responses:
        combined_text = f['query'] + " " + f['response']
        additional_texts.append(combined_text)

    # 将列表转换为字符串
    additional_texts = " ".join(additional_texts)

    # 使用原始文本和高质量文本创建新索引
    all_texts = texts + additional_texts
    new_vectorstore = encode_from_string(all_texts)

    return new_vectorstore
```

### 演示如何根据用户反馈检索答案

```python
query = "What is the greenhouse effect?"

# 从 RAG 系统获取回复
response = qa_chain(query)["result"]

relevance = 5
quality = 5

# 收集反馈
feedback = get_user_feedback(query, response, relevance, quality)

# 存储反馈
store_feedback(feedback)

# 为未来检索调整相关性分数
docs = retriever.get_relevant_documents(query)
adjusted_docs = adjust_relevance_scores(query, docs, load_feedback_data())

# 使用调整后的文档更新检索器
retriever.search_kwargs['k'] = len(adjusted_docs)
retriever.search_kwargs['docs'] = adjusted_docs
```

### 定期微调向量存储

```python
# 定期（例如每天或每周）微调索引
new_vectorstore = fine_tune_index(load_feedback_data(), content)
retriever = new_vectorstore.as_retriever()
```

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--retrieval-with-feedback-loop)
