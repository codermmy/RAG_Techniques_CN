# 文档搜索中的可解释检索

## 概述

本代码实现了一个可解释检索器，这是一个不仅根据查询检索相关文档，还解释为什么每个检索到的文档相关的系统。它结合了基于向量的相似性搜索和自然语言解释，增强了检索过程的透明度和可解释性。

## 动机

传统文档检索系统通常作为黑盒工作，提供结果而不解释为什么选择这些结果。在需要理解结果背后推理的场景中，这种缺乏透明度可能会带来问题。可解释检索器通过提供对每个检索文档相关性的洞察来解决这一问题。

## 关键组件

1. 从输入文本创建向量存储
2. 使用 FAISS 进行高效相似性搜索的基础检索器
3. 用于生成解释的语言模型（LLM）
4. 结合检索和解释生成的自定义 ExplainableRetriever 类

## 方法详情

### 文档预处理和向量存储创建

1. 输入文本使用 OpenAI 的嵌入模型转换为嵌入向量。
2. 从这些嵌入创建 FAISS 向量存储以进行高效相似性搜索。

### 检索器设置

1. 从向量存储创建基础检索器，配置为返回前 5 个最相似的文档。

### 解释生成

1. 使用 LLM（GPT-4）生成解释。
2. 定义自定义提示模板来指导 LLM 解释检索文档的相关性。

### ExplainableRetriever 类

1. 将基础检索器和解释生成组合到单一接口中。
2. `retrieve_and_explain` 方法：
   - 使用基础检索器检索相关文档。
   - 对于每个检索到的文档，生成其与查询相关性的解释。
   - 返回包含文档内容及其解释的字典列表。

## 此方法的优势

1. **透明度**：用户可以理解为什么检索到特定文档。
2. **信任**：解释建立用户对系统结果的信心。
3. **学习**：用户可以深入了解查询和文档之间的关系。
4. **调试**：更容易识别和纠正检索过程中的问题。
5. **定制化**：解释提示可以根据不同的用例或领域进行调整。

## 结论

可解释检索器代表了向更可解释和可信的信息检索系统迈出的重要一步。通过提供自然语言解释以及检索到的文档，它弥合了强大的基于向量的搜索技术与人类理解之间的差距。这种方法在信息检索背后的推理与检索到的信息本身同样重要的各个领域都有潜在应用，例如法律研究、医疗信息系统和教育工具。

# 包安装和导入

下面的单元格安装运行此 notebook 所需的所有必要包。

```python
# 安装所需的包
!pip install python-dotenv
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


# 原始路径追加已替换为 Colab 兼容性
from helper_functions import *
from evaluation.evalute_rag import *

# 从.env 文件加载环境变量
load_dotenv()

# 设置 OpenAI API 密钥环境变量
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
```

### 定义可解释检索器类

```python
class ExplainableRetriever:
    def __init__(self, texts):
        self.embeddings = OpenAIEmbeddings()

        self.vectorstore = FAISS.from_texts(texts, self.embeddings)
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=4000)


        # 创建基础检索器
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        # 创建解释链
        explain_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""
            分析以下查询和检索到的上下文之间的关系。
            解释为什么此次上下文与查询相关，以及它如何帮助回答查询。

            查询：{query}

            上下文：{context}

            解释：
            """
        )
        self.explain_chain = explain_prompt | self.llm

    def retrieve_and_explain(self, query):
        # 检索相关文档
        docs = self.retriever.get_relevant_documents(query)

        explained_results = []

        for doc in docs:
            # 生成解释
            input_data = {"query": query, "context": doc.page_content}
            explanation = self.explain_chain.invoke(input_data).content

            explained_results.append({
                "content": doc.page_content,
                "explanation": explanation
            })

        return explained_results
```

### 创建模拟示例和可解释检索器实例

```python
# 使用示例
texts = [
    "The sky is blue because of the way sunlight interacts with the atmosphere.",
    "Photosynthesis is the process by which plants use sunlight to produce energy.",
    "Global warming is caused by the increase of greenhouse gases in Earth's atmosphere."
]

explainable_retriever = ExplainableRetriever(texts)
```

### 显示结果

```python
query = "Why is the sky blue?"
results = explainable_retriever.retrieve_and_explain(query)

for i, result in enumerate(results, 1):
    print(f"结果{i}:")
    print(f"内容：{result['content']}")
    print(f"解释：{result['explanation']}")
    print()
```

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--explainable-retrieval)
