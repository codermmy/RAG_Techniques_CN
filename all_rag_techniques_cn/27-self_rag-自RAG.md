# Self-RAG: 检索增强生成的动态方法

## 概述

Self-RAG 是一种先进的算法，它结合了自然语言处理中基于检索和基于生成的方法的优势。它能动态地决定是否使用检索到的信息以及如何在生成回复时最好地利用这些信息，旨在产生更准确、更相关且更有用的输出。

## 动机

传统问答系统在平衡检索信息的使用和新内容生成方面常常面临困难。有些系统可能过度依赖检索数据，导致回复缺乏灵活性；而另一些系统可能在生成回复时缺乏足够的事实依据。Self-RAG 通过实施多步骤流程来解决这些问题，该流程仔细评估检索信息的必要性和相关性，并评估生成回复的质量。

## 核心组件

1. **检索决策 (Retrieval Decision)**: 判断给定查询是否需要检索。
2. **文档检索**: 从向量存储中获取可能相关的文档。
3. **相关性评估**: 评估检索到的文档与查询的相关性。
4. **回复生成**: 基于相关上下文生成回复。
5. **支持度评估**: 评估生成的回复在多大程度上得到了上下文的支持。
6. **效用评估**: 对生成回复的有用性进行评分。

## 方法细节

1. **检索决策**: 算法首先判断给定查询是否需要检索。这一步骤可以避免对可以直接回答的查询进行不必要的检索。

2. **文档检索**: 如果判断需要检索，算法会从向量存储中获取最相似的前 k 个文档。

3. **相关性评估**: 对每个检索到的文档进行与查询相关性的评估。此步骤过滤掉无关信息，确保只使用相关的上下文进行生成。

4. **回复生成**: 算法使用相关上下文生成回复。如果没有找到相关上下文，则在不进行检索的情况下生成回复。

5. **支持度评估**: 评估每个生成的回复，确定其在多大程度上得到了上下文的支持。此步骤有助于识别基于所提供信息的回复。

6. **效用评估**: 对每个回复的效用进行评分，考虑其解决原始查询的程度。

7. **回复选择**: 最后一步是根据支持度评估和效用评估选择最佳回复。

## 方法优势

1. **动态检索**: 通过判断是否需要检索，系统可以高效适应不同类型的查询。

2. **相关性过滤**: 相关性评估步骤确保只使用相关信息，减少生成过程中的噪声。

3. **质量保证**: 支持度评估和效用评估提供了衡量生成回复质量的方法。

4. **灵活性**: 系统可以在有检索或无检索的情况下生成回复，适应可用信息的情况。

5. **提高准确性**: 通过将回复建立在相关检索信息之上并评估其支持度，系统可以产生更准确的输出。

## 结论

Self-RAG 代表了一种复杂的问答和信息检索方法。通过结合多个评估步骤并动态决定检索信息的使用，它旨在产生不仅相关和准确，而且对最终用户有用的回复。这种方法展示了以深思熟虑、评估性的方式结合检索和生成技术来提高 AI 生成回复质量的潜力。

<div style="text-align: center;">

<img src="../images/self_rag.svg" alt="Self RAG" style="width:80%; height:auto;">
</div>

# 包安装与导入

下面的单元格安装运行此 notebook 所需的所有必要包。


```python
# 安装所需的包!pip install langchain langchain-openai python-dotenv
```

```python
# 克隆仓库以访问辅助函数和评估模块!git clone https://github.com/NirDiamant/RAG_TECHNIQUES.gitimport syssys.path.append('RAG_TECHNIQUES')# 如果需要运行最新数据# !cp -r RAG_TECHNIQUES/data .
```

```python
import osimport sysfrom dotenv import load_dotenvfrom langchain.prompts import PromptTemplatefrom langchain_openai import ChatOpenAIfrom langchain_core.pydantic_v1 import BaseModel, Field# 原始路径追加已替换为 Colab 兼容from helper_functions import *from evaluation.evalute_rag import *# 从 .env 文件加载环境变量load_dotenv()# 设置 OpenAI API 密钥环境变量os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
```

### 定义文件路径

```python
# 下载所需的数据文件import osos.makedirs('data', exist_ok=True)# 下载本 notebook 中使用的 PDF 文档!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
```

```python
path = "data/Understanding_Climate_Change.pdf"
```

### 创建向量存储

```python
vectorstore = encode_pdf(path)
```

### 初始化语言模型


```python
llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1000, temperature=0)
```

### 定义提示模板

```python
class RetrievalResponse(BaseModel):    response: str = Field(..., title="Determines if retrieval is necessary", description="仅输出 'Yes' 或 'No'.")retrieval_prompt = PromptTemplate(    input_variables=["query"],    template="Given the query '{query}', 判断是否需要检索. 仅输出 'Yes' 或 'No'.")class RelevanceResponse(BaseModel):    response: str = Field(..., title="Determines if context is relevant", description="仅输出 'Relevant' 或 'Irrelevant'.")relevance_prompt = PromptTemplate(    input_variables=["query", "context"],    template="Given the query '{query}' and the context '{context}', 判断上下文是否相关. 仅输出 'Relevant' 或 'Irrelevant'.")class GenerationResponse(BaseModel):    response: str = Field(..., title="Generated response", description="The generated response.")generation_prompt = PromptTemplate(    input_variables=["query", "context"],    template="Given the query '{query}' and the context '{context}', 生成回复.")class SupportResponse(BaseModel):    response: str = Field(..., title="Determines if response is supported", description="Output 'Fully supported', 'Partially supported', or 'No support'.")support_prompt = PromptTemplate(    input_variables=["response", "context"],    template="Given the response '{response}' and the context '{context}', 判断回复是否得到上下文支持. Output 'Fully supported', 'Partially supported', or 'No support'.")class UtilityResponse(BaseModel):    response: int = Field(..., title="Utility rating", description="Rate the utility of the response from 1 to 5.")utility_prompt = PromptTemplate(    input_variables=["query", "response"],    template="Given the query '{query}' and the response '{response}', 对回复的效用进行 1-5 评分.")# 为每个步骤创建 LLMChainretrieval_chain = retrieval_prompt | llm.with_structured_output(RetrievalResponse)relevance_chain = relevance_prompt | llm.with_structured_output(RelevanceResponse)generation_chain = generation_prompt | llm.with_structured_output(GenerationResponse)support_chain = support_prompt | llm.with_structured_output(SupportResponse)utility_chain = utility_prompt | llm.with_structured_output(UtilityResponse)
```

### 定义 Self-RAG 逻辑流程

```python
def self_rag(query, vectorstore, top_k=3):    print(f"\n处理查询：{query}")        # 步骤 1：判断是否需要检索    print("步骤 1：判断是否需要检索...")    input_data = {"query": query}    retrieval_decision = retrieval_chain.invoke(input_data).response.strip().lower()    print(f"检索决策：{retrieval_decision}")        if retrieval_decision == 'yes':        # 步骤 2：检索相关文档        print("步骤 2：检索相关文档...")        docs = vectorstore.similarity_search(query, k=top_k)        contexts = [doc.page_content for doc in docs]        print(f"检索到 {len(contexts)} 个文档")                # 步骤 3：评估检索到的文档的相关性        print("步骤 3：评估检索到的文档的相关性...")        relevant_contexts = []        for i, context in enumerate(contexts):            input_data = {"query": query, "context": context}            relevance = relevance_chain.invoke(input_data).response.strip().lower()            print(f"文档 {i+1} 相关性：{relevance}")            if relevance == 'relevant':                relevant_contexts.append(context)                print(f"相关上下文数量：{len(relevant_contexts)}")                # 如果未找到相关上下文，无需检索直接生成        if not relevant_contexts:            print("未找到相关上下文。无需检索直接生成...")            input_data = {"query": query, "context": "No relevant context found."}            return generation_chain.invoke(input_data).response                # 步骤 4：使用相关上下文生成响应        print("步骤 4：使用相关上下文生成响应...")        responses = []        for i, context in enumerate(relevant_contexts):            print(f"正在为上下文 {i+1} 生成响应...")            input_data = {"query": query, "context": context}            response = generation_chain.invoke(input_data).response                        # 步骤 5：评估支持度            print(f"步骤 5：评估响应 {i+1} 的支持度...")            input_data = {"response": response, "context": context}            support = support_chain.invoke(input_data).response.strip().lower()            print(f"支持度评估：{support}")                        # 步骤 6：评估效用            print(f"步骤 6：评估响应 {i+1} 的效用...")            input_data = {"query": query, "response": response}            utility = int(utility_chain.invoke(input_data).response)            print(f"效用评分：{utility}")                        responses.append((response, support, utility))                # 基于支持度和效用选择最佳响应        print("选择最佳响应...")        best_response = max(responses, key=lambda x: (x[1] == 'fully supported', x[2]))        print(f"最佳响应支持度：{best_response[1]}，效用：{best_response[2]}")        return best_response[0]    else:        # 无需检索直接生成        print("无需检索直接生成...")        input_data = {"query": query, "context": "无需检索。"}        return generation_chain.invoke(input_data).response
```

### 测试 Self-RAG 函数 - 高相关性的简单查询

```python
query = "气候变化对环境的影响是什么？"
response = self_rag(query, vectorstore)

print("\n最终响应：")
print(response)
```

### 测试 Self-RAG 函数 - 低相关性的更具挑战性的查询

```python
query = "哈利是如何打败奇洛的？"
response = self_rag(query, vectorstore)

print("\n最终响应：")
print(response)
```

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--self-rag)
