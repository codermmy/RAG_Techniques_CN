# Microsoft GraphRAG：使用知识图谱增强检索增强生成

 
## 概述

 
Microsoft GraphRAG是一个先进的检索增强生成（RAG）系统，它集成了知识图谱以提高大型语言模型（LLM）的性能。由Microsoft Research开发的GraphRAG通过使用LLM生成的知识图谱来增强文档分析并提高响应质量，解决了传统RAG方法的局限性。

## 动机

 
传统RAG系统在处理需要从不同来源综合信息的复杂查询时往往面临困难。GraphRAG旨在：
连接数据集中的相关信息。
增强对语义概念的理解。
提高全局性理解任务的性能。

## 关键组件

知识图谱生成：构建以实体为节点、关系为边的图。
社区检测：识别图中相关实体的聚类。
摘要生成：为每个社区生成摘要，为LLM提供上下文。
查询处理：使用这些摘要来增强LLM回答复杂问题的能力。

## 方法详情

索引阶段

 
文本分块：将源文本分割成可管理的块。
元素提取：使用LLM识别实体和关系。
图构建：从提取的元素构建图。
社区检测：应用Leiden等算法发现社区。
社区摘要：为每个社区创建摘要。

查询阶段

 
局部答案生成：使用社区摘要生成初步答案。
全局答案综合：将局部答案组合成全面的响应。


## GraphRAG的优势
GraphRAG是一个强大的工具，解决了基线RAG模型的一些关键局限性。与标准RAG模型不同，GraphRAG擅长识别不同信息片段之间的联系并从中获取洞察。这使其成为需要从大型数据集合或难以总结的文档中提取洞察的用户的理想选择。通过利用其先进的基于图的架构，GraphRAG能够提供对复杂语义概念的整体理解，使其成为任何需要快速准确查找信息的人的宝贵工具。无论您是研究人员、分析师，还是只是需要保持知情的人，GraphRAG都可以帮助您连接信息点并发现新的洞察。

## 结论

Microsoft GraphRAG代表了检索增强生成的重要进步，特别是对于需要全局理解数据集的任务。通过整合知识图谱，它提供了改进的性能，使其非常适合复杂的信息检索和分析。

对于那些有基础RAG系统经验的人来说，GraphRAG提供了探索更复杂解决方案的机会，尽管它可能并非所有用例都必需。

检索增强生成（RAG）通常通过将长文本分块、为每个块创建文本嵌入，并根据与查询的相似性搜索检索块以包含在LLM生成上下文中来执行。这种方法在许多场景中效果良好，具有引人注目的速度和成本权衡，但在需要详细理解文本的场景中并不总是表现良好。

GraphRag ( [microsoft.github.io/graphrag](https://microsoft.github.io/graphrag/) )

<div style="text-align: center;">

<img src="../images/Microsoft_GraphRag.svg" alt="adaptive retrieval" style="width:100%; height:auto;">
</div>

要运行此notebook，您可以使用OpenAI API密钥或Azure OpenAI密钥。
创建一个`.env`文件并填写您的OpenAI或Azure Open AI部署的凭据。以下代码加载这些环境变量并设置我们的AI客户端。

```python
AZURE_OPENAI_API_KEY=""
AZURE_OPENAI_ENDPOINT=""
GPT4O_MODEL_NAME="gpt-4o"
TEXT_EMBEDDING_3_LARGE_DEPLOYMENT_NAME=""
AZURE_OPENAI_API_VERSION="2024-06-01"

OPENAI_API_KEY=""
```

```python
%pip install graphrag
```

# 包安装和导入

下面的单元格安装运行此notebook所需的所有必要包。

```python
# 安装所需的包
!pip install beautifulsoup4 openai python-dotenv pyyaml
```

# 包安装

下面的单元格安装运行此notebook所需的所有必要包。如果您在新的环境中运行此notebook，请先执行此单元格以确保安装所有依赖项。

```python
# 安装所需的包
!pip install openai python-dotenv
```

```python
from dotenv import load_dotenvimport osload_dotenv()from openai import AzureOpenAI, OpenAIAZURE=True #改为 False 以使用 OpenAIif AZURE:    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")    GPT4O_DEPLOYMENT_NAME = os.getenv("GPT4O_MODEL_NAME")    TEXT_EMBEDDING_3_LARGE_NAME = os.getenv("TEXT_EMBEDDING_3_LARGE_DEPLOYMENT_NAME")    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")    oai = AzureOpenAI(azure_endpoint=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_API_KEY, api_version=AZURE_OPENAI_API_VERSION)else:    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")    oai = OpenAI(api_key=OPENAI_API_KEY)
```

我们首先获取一个文本来处理。使用维基百科上关于Elon Musk的文章

```python
import requests
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/Elon_Musk"  # 替换为您想抓取的网页URL
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

if not os.path.exists('data'): 
    os.makedirs('data')

if not os.path.exists('data/elon.md'):
    elon = soup.text.split('\nSee also')[0]
    with open('data/elon.md', 'w', encoding='utf-8') as f:
        f.write(elon)
else:
    with open('data/elon.md', 'r') as f:
        elon = f.read()
```

GraphRag有一组便捷的CLI命令可以使用。我们将首先配置系统，然后运行索引操作。使用GraphRag进行索引是一个更长的过程，并且成本显著更高，因为它不只是计算嵌入，GraphRag还进行许多LLM调用来分析文本、提取实体并构建图。不过这是一次性费用。

```python
import yaml

if not os.path.exists('data/graphrag'):
    !python -m graphrag.index --init --root data/graphrag

with open('data/graphrag/settings.yaml', 'r') as f:
    settings_yaml = yaml.load(f, Loader=yaml.FullLoader)
settings_yaml['llm']['model'] = "gpt-4o"
settings_yaml['llm']['api_key'] = AZURE_OPENAI_API_KEY if AZURE else OPENAI_API_KEY
settings_yaml['llm']['type'] = 'azure_openai_chat' if AZURE else 'openai_chat'
settings_yaml['embeddings']['llm']['api_key'] = AZURE_OPENAI_API_KEY if AZURE else OPENAI_API_KEY
settings_yaml['embeddings']['llm']['type'] = 'azure_openai_embedding' if AZURE else 'openai_embedding'
settings_yaml['embeddings']['llm']['model'] = TEXT_EMBEDDING_3_LARGE_NAME if AZURE else 'text-embedding-3-large'
if AZURE:
    settings_yaml['llm']['api_version'] = AZURE_OPENAI_API_VERSION
    settings_yaml['llm']['deployment_name'] = GPT4O_DEPLOYMENT_NAME
    settings_yaml['llm']['api_base'] = AZURE_OPENAI_ENDPOINT
    settings_yaml['embeddings']['llm']['api_version'] = AZURE_OPENAI_API_VERSION
    settings_yaml['embeddings']['llm']['deployment_name'] = TEXT_EMBEDDING_3_LARGE_NAME
    settings_yaml['embeddings']['llm']['api_base'] = AZURE_OPENAI_ENDPOINT

with open('data/graphrag/settings.yaml', 'w') as f:
    yaml.dump(settings_yaml, f)

if not os.path.exists('data/graphrag/input'):
    os.makedirs('data/graphrag/input')
    !cp data/elon.md data/graphrag/input/elon.txt
    !python -m graphrag.index --root ./data/graphrag
```

您应该会得到以下输出：🚀 所有工作流已成功完成。

要查询GraphRag，我们将再次使用其CLI，确保使用与我们嵌入搜索中使用的相同的上下文长度进行配置。

```python
import subprocess
import re
DEFAULT_RESPONSE_TYPE = 'Summarize and explain in 1-2 paragraphs with bullet points using at most 300 tokens'
DEFAULT_MAX_CONTEXT_TOKENS = 10000

def remove_data(text):
    return re.sub(r'\[Data:.*?\]', '', text).strip()


def ask_graph(query,method):
    env = os.environ.copy() | {
      'GRAPHRAG_GLOBAL_SEARCH_MAX_TOKENS': str(DEFAULT_MAX_CONTEXT_TOKENS),
    }
    command = [
      'python', '-m', 'graphrag.query',
      '--root', './data/graphrag',
      '--method', method,
      '--response_type', DEFAULT_RESPONSE_TYPE,
      query,
    ]
    output = subprocess.check_output(command, universal_newlines=True, env=env, stderr=subprocess.DEVNULL)
    return remove_data(output.split('Search Response: ')[1])
```

GraphRag提供2种类型的搜索：
1. 全局搜索：通过利用社区摘要对语料库的整体性问题进行推理。
2. 局部搜索：通过扩展到其邻居和相关概念对特定实体进行推理。

让我们看看局部搜索：

```python
from IPython.display import Markdown
local_query="What and how many companies and subsidieries founded by Elon Musk"
local_result = ask_graph(local_query,'local')

Markdown(local_result)
```

```python
global_query="What are the major accomplishments of Elon Musk?"
global_result = ask_graph(global_query,'global')

Markdown(global_result)
```

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--microsoft-graphrag)
