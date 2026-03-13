# 校正 RAG 流程: 带动态校正的检索增强生成

## 概述

校正 RAG (Corrective RAG) 流程是一个先进的信息检索和回复生成系统。它通过动态评估和校正检索过程扩展了标准 RAG 方法，结合了向量数据库、网络搜索和语言模型的力量，为用户查询提供准确且具有上下文意识的回复。

## 动机

虽然传统 RAG 系统改善了信息检索和回复生成，但当检索到的信息不相关或过时时，它们仍存在不足。校正 RAG 流程通过以下方式解决这些限制:

1. 利用现有知识库
2. 评估检索信息的相关性
3. 必要时动态搜索网络
4. 提炼和组合来自多个来源的知识
5. 基于最合适的知识生成类人回复

## 核心组件

1. **FAISS 索引**: 用于现有知识高效相似度搜索的向量数据库。
2. **检索评估器**: 评估检索文档与查询的相关性。
3. **知识提炼**: 必要时从文档中提取关键信息。
4. **网络搜索查询重写器**: 当本地知识不足时优化网络搜索的查询。
5. **回复生成器**: 基于积累的知识创建类人回复。

## 方法细节

1. **文档检索**: 
   - 在 FAISS 索引中执行相似度搜索以找到相关文档。
   - 检索前 k 个文档（默认 k=3）。

2. **文档评估**:
   - 计算每个检索文档的相关性分数。
   - 根据最高相关性分数确定最佳行动方案。

3. **校正性知识获取**:
   - 如果高相关性（分数 > 0.7）: 直接使用最相关的文档。
   - 如果低相关性（分数 < 0.3）: 通过重写查询进行网络搜索来校正。
   - 如果模糊（0.3 ≤ 分数 ≤ 0.7）: 通过结合最相关文档和网络搜索结果来校正。

4. **自适应知识处理**:
   - 对于网络搜索结果: 提炼知识以提取关键点。
   - 对于模糊情况: 将原始文档内容与提炼的网络搜索结果相结合。

5. **回复生成**:
   - 使用语言模型基于查询和获取的知识生成类人回复。
   - 在回复中包含来源信息以保持透明度。

## 校正 RAG 方法的优势

1. **动态校正**: 适应检索信息的质量，确保相关性和准确性。
2. **灵活性**: 根据需要利用现有知识和网络搜索。
3. **准确性**: 在使用信息前评估其相关性，确保高质量回复。
4. **透明度**: 提供来源信息，允许用户验证信息来源。
5. **高效性**: 使用向量搜索从大型知识库中快速检索。
6. **上下文理解**: 必要时组合多个信息来源以提供全面的回复。
7. **最新信息**: 可以用当前网络信息补充或替换过时的本地知识。

## 结论

校正 RAG 流程代表了标准 RAG 方法的复杂进化。通过智能评估和校正检索过程，它克服了传统 RAG 系统的常见限制。这种动态方法确保回复基于最相关和最新的可用信息，无论是来自本地知识库还是网络。系统根据相关性分数调整信息获取策略的能力使其特别适合需要高准确性和当前信息的应用，如研究辅助、动态知识库和高级问答系统。

<div style="text-align: center;">

<img src="../images/crag.svg" alt="Corrective RAG" style="width:80%; height:auto;">
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
import osimport sysfrom dotenv import load_dotenvfrom langchain.prompts import PromptTemplatefrom langchain_openai import ChatOpenAIfrom langchain_core.pydantic_v1 import BaseModel, Field# 原始路径追加已替换为 Colab 兼容from helper_functions import *from evaluation.evalute_rag import *# 从 .env 文件加载环境变量load_dotenv()# 设置 OpenAI API 密钥环境变量os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')from langchain.tools import DuckDuckGoSearchResults
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

### 初始化 OpenAI 语言模型


```python
llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1000, temperature=0)
```

### 初始化搜索工具

```python
search = DuckDuckGoSearchResults()
```

### 定义检索评估器、知识提炼和查询重写 LLM 链

```python
# 检索评估器class RetrievalEvaluatorInput(BaseModel):    relevance_score: float = Field(..., description="The relevance score of the document to the query. the score should be between 0 and 1.")def retrieval_evaluator(query: str, document: str) -> float:    prompt = PromptTemplate(        input_variables=["query", "document"],        template="在 0 到 1 的范围内，以下文档与查询的相关性如何? 查询：{query}\nDocument: {document}\nRelevance score:"    )    chain = prompt | llm.with_structured_output(RetrievalEvaluatorInput)    input_variables = {"query": query, "document": document}    result = chain.invoke(input_variables).relevance_score    return result# 知识精炼class KnowledgeRefinementInput(BaseModel):    key_points: str = Field(..., description="The document to extract key information from.")def knowledge_refinement(document: str) -> List[str]:    prompt = PromptTemplate(        input_variables=["document"],        template="以要点形式提取以下文档中的关键信息:\n{document}\nKey points:"    )    chain = prompt | llm.with_structured_output(KnowledgeRefinementInput)    input_variables = {"document": document}    result = chain.invoke(input_variables).key_points    return [point.strip() for point in result.split('\n') if point.strip()]# 网络搜索查询重写器class QueryRewriterInput(BaseModel):    query: str = Field(..., description="The query to rewrite.")def rewrite_query(query: str) -> str:    prompt = PromptTemplate(        input_variables=["query"],        template="重写以下查询以使其更适合网络搜索:\n{query}\nRewritten query:"    )    chain = prompt | llm.with_structured_output(QueryRewriterInput)    input_variables = {"query": query}    return chain.invoke(input_variables).query.strip()
```

### 解析搜索结果的辅助函数


```python
def parse_search_results(results_string: str) -> List[Tuple[str, str]]:    """    Parse a JSON string of search results into a list of title-link tuples.    Args:        results_string (str): A JSON-formatted string containing search results.    Returns:        List[Tuple[str, str]]: A list of tuples, where each tuple contains the title and link of a search result.                               If parsing fails, an empty list is returned.    """    try:        # 尝试解析 JSON 字符串        results = json.loads(results_string)        # 从每个结果中提取并返回标题和链接        return [(result.get('title', 'Untitled'), result.get('link', '')) for result in results]    except json.JSONDecodeError:        # 通过返回空列表处理 JSON 解码错误        print("Error parsing search results. Returning empty list.")        return []
```

### 定义 CRAG 流程的子函数

```python
def retrieve_documents(query: str, faiss_index: FAISS, k: int = 3) -> List[str]:
    """
    使用 FAISS 索引根据查询检索文档.

    Args:
        query (str): The query string to search for.
        faiss_index (FAISS): The FAISS index used for similarity search.
        k (int): The number of top documents to retrieve. Defaults to 3.

    Returns:
        List[str]: A list of the retrieved document contents.
    """
    docs = faiss_index.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]

def evaluate_documents(query: str, documents: List[str]) -> List[float]:
    """
    根据查询评估文档的相关性.

    Args:
        query (str): The query string.
        documents (List[str]): A list of document contents to evaluate.

    Returns:
        List[float]: A list of relevance scores for each document.
    """
    return [retrieval_evaluator(query, doc) for doc in documents]

def perform_web_search(query: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    根据查询执行网络搜索.

    Args:
        query (str): The query string to search for.

    Returns:
        Tuple[List[str], List[Tuple[str, str]]]: 
            - A list of refined knowledge obtained from the web search.
            - A list of tuples containing titles and links of the sources.
    """
    rewritten_query = rewrite_query(query)
    web_results = search.run(rewritten_query)
    web_knowledge = knowledge_refinement(web_results)
    sources = parse_search_results(web_results)
    return web_knowledge, sources

def generate_response(query: str, knowledge: str, sources: List[Tuple[str, str]]) -> str:
    """
    使用知识和源生成对查询的响应.

    Args:
        query (str): The query string.
        knowledge (str): The refined knowledge to use in the response.
        sources (List[Tuple[str, str]]): A list of tuples containing titles and links of the sources.

    Returns:
        str: The generated response.
    """
    response_prompt = PromptTemplate(
        input_variables=["query", "knowledge", "sources"],
        template="根据以下知识回答问题。在答案末尾包含源及其链接（如果有）:\n查询：{query}\nKnowledge: {knowledge}\n源： {sources}\nAnswer:"
    )
    input_variables = {
        "query": query,
        "knowledge": knowledge,
        "sources": "\n".join([f"{title}: {link}" if link else title for title, link in sources])
    }
    response_chain = response_prompt | llm
    return response_chain.invoke(input_variables).content
```

### CRAG 流程


```python
def crag_process(query: str, faiss_index: FAISS) -> str:    """    Process a query by retrieving, evaluating, and using documents or performing a web search to 生成回复.    Args:        query (str): The query string to process.        faiss_index (FAISS): The FAISS index used for document retrieval.    Returns:        str: The generated response based on the query.    """    print(f"\nProcessing query: {query}")    # 检索并评估文档    retrieved_docs = retrieve_documents(query, faiss_index)    eval_scores = evaluate_documents(query, retrieved_docs)        print(f"\n检索到 {len(retrieved_docs)} 个文档")    print(f"评估分数：{eval_scores}")    # 基于评估分数确定操作    max_score = max(eval_scores)    sources = []        if max_score > 0.7:        print("\n操作：正确 - 使用检索到的文档")        best_doc = retrieved_docs[eval_scores.index(max_score)]        final_knowledge = best_doc        sources.append(("Retrieved document", ""))    elif max_score < 0.3:        print("\n操作：不正确 - 执行网络搜索")        final_knowledge, sources = perform_web_search(query)    else:        print("\n操作：模糊 - 结合检索到的文档和网络搜索")        best_doc = retrieved_docs[eval_scores.index(max_score)]        # 精炼检索到的知识        retrieved_knowledge = knowledge_refinement(best_doc)        web_knowledge, web_sources = perform_web_search(query)        final_knowledge = "\n".join(retrieved_knowledge + web_knowledge)        sources = [("Retrieved document", "")] + web_sources    print("\n最终知识：")    print(final_knowledge)        print("\n源：")    for title, link in sources:        print(f"{title}: {link}" if link else title)    # 生成响应    print("\n生成响应中...")    response = generate_response(query, final_knowledge, sources)    print("\n响应已生成")    return response
```

### 示例查询 - 与文档高度相关

```python
query = "气候变化的主要原因是什么？"
result = crag_process(query, vectorstore)
print(f"查询：{query}")
print(f"答案：{result}")
```

### 示例查询 - 与文档低相关性

```python
query = "how did harry beat quirrell?"
result = crag_process(query, vectorstore)
print(f"查询：{query}")
print(f"答案：{result}")
```

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--crag)
