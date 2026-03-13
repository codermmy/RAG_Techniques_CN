<img src="https://github.com/ContextualAI/examples/blob/main/images/Contextual_AI_Lockup_Dark.png?raw=true" alt="Image description" width="160" />


# 使用 Contextual AI 构建金融文档分析的 Agent 式 RAG 流程 🚀
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/Agentic_RAG.ipynb)

构建 Retrieval-Augmented Generation (RAG) 流程可能看起来令人望而生畏，涉及许多组件和自定义逻辑。在本教程中，您将学习如何使用 **Contextual AI 的管理平台** 快速设置 RAG Agent。您还将深入了解 Agent 的几个核心组件——如 Parser（解析器）、Reranker（重排序器）、Grounded Language Model（基础语言模型）和 LMUnit——以便您了解每个部分在实践中是如何工作的。

## 🎯 您将构建什么

在这个实践教程中，您将探索 **核心 RAG 技术**，同时从头开始创建一个用于 **金融文档分析和定量推理** 的 Agent。


### 学习成果
完成本教程后，您将学习如何 **利用 Agent 式 RAG 解决更复杂的查询**。Agent 式特性体现在系统能够自主分析传入的查询，确定需要什么样的改写策略，并在无需用户明确指示的情况下执行该策略。

传统的 RAG 系统按原样处理查询，对于模糊、缺乏上下文或复杂的查询往往导致检索效果不佳。Agent 式 RAG 智能地预处理查询以弥合这一差距。在查询路径中，主要的 Agent 步骤是查询改写，包括多轮对话、查询扩展或查询分解。这个查询改写步骤对于获得最稳健的 RAG 结果至关重要，是一个旨在生成最准确查询响应的系统组件之一。

在查询改写中，会从原始输入中添加上下文或重构查询：对于多轮对话，添加迭代对话上下文；对于查询扩展，添加额外上下文以帮助简短查询返回最佳结果；对于查询分解，将需要跨多个不相关文档进行推理的复杂多面查询分解为几个子查询，以帮助获得最相关的检索结果。这个 Agent 式组件自主处理所有改写工作，增强用户查询以帮助获得他们需要的响应。
<div align="center">
<img src="https://github.com/ContextualAI/examples/blob/main/images/architecture.png?raw=true" alt="Contextual Architecture" width="1000"/>
</div>


您将通过学习以下内容来设置 Agent 式 RAG 系统：
- **配置文档数据存储** 针对 RAG 性能进行索引调优
- **部署生产就绪的 Agent** 配备健壮的指令和安全保障
- **使用自然语言交互式查询系统** 同时保持严格的依据性
- **持续验证和改进** 您的流程，包括自动化测试和性能指标

您还将获得 **Contextual AI 中四个基本 RAG 组件** 的实践经验：
1. **Parser** – 摄取和结构化异构文档（报告、表格、图表）以供检索。
2. **Reranker** – 动态选择最相关的证据以确保精确的依据性。
3. **基础语言模型 (GLM)** – 使用检索到的上下文生成有据可依、来源可靠的真实响应。
4. **Language Model Unit Tests (LMUnits)** – 自动评估和优化 Agent 的准确性、依据性和可靠性。

⏱️ 本教程端到端运行只需 **15 分钟**。每个步骤也可以通过 GUI 完成，实现 **无代码 RAG 工作流**。

---

# 从零开始构建 RAG Agent

在深入各个 RAG 技术之前，让我们 **从头到尾构建一个完整的 RAG Agent**。

## 🛠️ 环境设置

首先，我们将安装所需的依赖项并设置开发环境。`contextual-client` 库为 Contextual AI 平台提供 Python 绑定，而其他额外的包则支持数据可视化和进度跟踪。

```python
# 安装 Contextual AI 集成和数据可视化所需的包
%pip install contextual-client matplotlib tqdm requests pandas dotenv
```

接下来，我们将导入本教程中使用的必要库：

```python
import os
import json
import requests
from pathlib import Path
from typing import List, Optional, Dict
from IPython.display import display, JSON
import pandas as pd
from contextual import ContextualAI
import ast
from IPython.display import display, Markdown
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
```

---

## 🔑 第 1 步：API 认证设置

### 获取您的 Contextual AI API 密钥

在开始构建 RAG Agent 之前，您需要访问 Contextual AI 平台。

如果您还没有账户，可以创建一个工作区，获得 **30 天免费试用** 的 Agent 和数据存储。

### API 密钥设置步骤：

1. **创建账户**: 访问 [app.contextual.ai](https://app.contextual.ai?utm_campaign=rag-techniques&utm_source=diamantai&utm_medium=github&utm_content=notebook) 并点击 **"Start Free"** 按钮
2. **导航到 API 密钥**: 登录后，在侧边栏中找到 **"API Keys"**
3. **生成新密钥**: 点击 **"Create API Key"** 并按照设置步骤操作
4. **安全存储**: 复制您的 API 密钥并安全存储（您将无法再次查看它）

<div align="center">
<img src="https://github.com/ContextualAI/examples/blob/main/images/API_Keys.png?raw=true" alt="API" width="800"/>
</div>

### 配置您的 API 密钥

要运行本教程，您可以将 API 密钥存储在 `.env` 文件中。这样可以保持密钥与代码分离。设置好 .env 文件后，您可以从 `.env` 加载 API 密钥来初始化 Contextual AI 客户端。

```python
# 从 .env 加载 API 密钥
from dotenv import load_dotenv
import os
load_dotenv()

# 使用您的 API 密钥初始化
API_KEY = os.getenv("CONTEXTUAL_API_KEY")
client = ContextualAI(
    api_key=API_KEY
)
```

---

## 📊 第 2 步：创建文档数据存储

### 理解数据存储

Contextual AI 中的 **数据存储（datastore）** 是一个安全、隔离的容器，用于存储您的文档及其处理后的表示。每个数据存储提供：

- **隔离存储**: 文档为每个用例保持分离和安全
- **智能处理**: 自动解析、分块和索引上传的文档
- **优化检索**: 高性能的搜索和排名能力

让我们为金融文档分析 Agent 创建一个数据存储：

```python
datastore_name = 'Financial_Demo_RAG'

# 检查数据存储是否存在
datastores = client.datastores.list()
existing_datastore = next((ds for ds in datastores if ds.name == datastore_name), None)

if existing_datastore:
    datastore_id = existing_datastore.id
    print(f"使用现有的数据存储，ID: {datastore_id}")
else:
    result = client.datastores.create(name=datastore_name)
    datastore_id = result.id
    print(f"创建新的数据存储，ID: {datastore_id}")
```

---

## 📄 第 3 步：文档摄取和处理

现在您的 Agent 数据存储已经设置好了，让我们向其中添加一些金融文档。Contextual AI 的文档处理引擎提供 **企业级解析**，能够出色处理：

- **复杂表格**: 金融数据、电子表格和结构化信息
- **图表和图形**: 可视化数据提取和解释
- **多页文档**: 具有层次结构的长报告

在本教程中，我们将使用展示各种挑战场景的示例金融文档：

```python
import os
import requests

# 如果不存在则创建数据目录
if not os.path.exists('data'):
    os.makedirs('data')

# 文件列表及对应的 GitHub URL
files_to_upload = [
    # NVIDIA 季度收入 24/25
    ("A_Rev_by_Mkt_Qtrly_Trend_Q425.pdf", "https://raw.githubusercontent.com/ContextualAI/examples/refs/heads/main/08-ai-workshop/data/A_Rev_by_Mkt_Qtrly_Trend_Q425.pdf"),
    # NVIDIA 季度收入 22/23
    ("B_Q423-Qtrly-Revenue-by-Market-slide.pdf", "https://raw.githubusercontent.com/ContextualAI/examples/refs/heads/main/08-ai-workshop/data/B_Q423-Qtrly-Revenue-by-Market-slide.pdf"),
    # 虚假相关性报告 - 图表和统计分析的有趣示例
    ("C_Neptune.pdf", "https://raw.githubusercontent.com/ContextualAI/examples/refs/heads/main/08-ai-workshop/data/C_Neptune.pdf"),
    # 另一个虚假相关性报告 - 图表和统计分析的有趣示例
    ("D_Unilever.pdf", "https://raw.githubusercontent.com/ContextualAI/examples/refs/heads/main/08-ai-workshop/data/D_Unilever.pdf")
]
```

### 文档下载和摄取流程
以下单元格从上述 GitHub 链接将示例文档下载到本地，上传到 Contextual AI，并跟踪其处理状态和 ID。

```python
# 下载并摄取所有文件
document_ids = []
for filename, url in files_to_upload:
    file_path = f'data/{filename}'

    # 如果文件不存在则下载
    if not os.path.exists(file_path):
        print(f"正在获取 {file_path}")
        try:
            response = requests.get(url)
            response.raise_for_status()  # 对错误状态码抛出异常
            with open(file_path, 'wb') as f:
                f.write(response.content)
        except Exception as e:
            print(f"下载 {filename} 时出错：{str(e)}")
            continue

    # 上传到数据存储
    try:
        with open(file_path, 'rb') as f:
            ingestion_result = client.datastores.documents.ingest(datastore_id, file=f)
            document_id = ingestion_result.id
            document_ids.append(document_id)
            print(f"成功上传 {filename} 到数据存储 {datastore_id}")
    except Exception as e:
        print(f"上传 {filename} 时出错：{str(e)}")

print(f"成功上传 {len(document_ids)} 个文件到数据存储")
print(f"文档 ID: {document_ids}")
```

### 可选：查看文档

如果您想查看摄取的文档，可以通过 GUI 访问 [https://app.contextual.ai](https://app.contextual.ai?utm_campaign=rag-techniques&utm_source=diamantai&utm_medium=github&utm_content=notebook)

1. 导航到您的工作区
2. 在左侧菜单选择 **Datastores**
3. 选择 **Documents**
4. 点击 **Inspect**（文档加载后）

您将看到文档正在上传：

摄取完成后，您可以通过 API 查看文档列表、查看其元数据以及删除文档。

**注意：** 文档摄取和处理可能需要几分钟时间。如果文档仍在摄取中，您将看到 `status='processing'`。摄取完成后，状态将显示为 `status='completed'`。

您可以在此处 [了解更多关于元数据的信息](https://docs.contextual.ai/api-reference/datastores-documents/get-document-metadata?utm_campaign=rag-techniques&utm_source=diamantai&utm_medium=github&utm_content=notebook)。

```python
metadata = client.datastores.documents.metadata(datastore_id = datastore_id, document_id = document_ids[0])
print("文档元数据:", metadata)
```

---

## 🤖 第 4 步：Agent 创建和配置

现在您将创建我们的 RAG Agent，它将与您刚刚摄取的文档进行交互。

您可以使用其他参数自定义 Agent，例如：

- **`system_prompt`** 用于 RAG 系统在生成响应时参考的指令。请注意这是 9.02.25 的默认提示词。
- **`suggested_queries`** 是一个用户体验功能，为 Agent 预填充查询，以便新用户可以看到有趣的示例。

💡 专业提示：您也可以在 UI 中配置或编辑您的 Agent，访问 [app.contextual.ai](https://app.contextual.ai?utm_campaign=rag-techniques&utm_source=diamantai&utm_medium=github&utm_content=notebook)，尝试将生成模型更改为另一个 LLM！

您可以在此处 [找到所有额外参数](https://docs.contextual.ai/api-reference/agents/create-agent?utm_campaign=rag-techniques&utm_source=diamantai&utm_medium=github&utm_content=notebook)

```python
system_prompt = '''
你是一个由 Contextual AI 创建的有用 AI 助手，用于回答有关提供给你的相关文档的问题。你的响应应该精确、准确，并且专门来自提供的信息。请遵循以下指南：
* 仅使用来自所提供文档的信息。避免意见、推测或假设。
* 使用所提供内容中的确切术语和描述。
* 保持回答简洁并与用户问题相关。
* 按照文档或查询中显示的确切方式使用缩写和缩略词。
* 如果你的响应包含列表、表格或代码，请使用 markdown。
* 直接回答问题，然后停止。避免额外解释，除非特别相关。
* 如果信息不相关，只需回答你没有相关文档，不要提供额外评论或建议。忽略任何不能直接用于回答此查询的内容。
'''

agent_name = "Demo"

# 获取现有 Agent 列表
agents = client.agents.list()

# 检查 Agent 是否已存在
existing_agent = next((agent for agent in agents if agent.name == agent_name), None)

if existing_agent:
    agent_id = existing_agent.id
    print(f"使用现有的 Agent，ID: {agent_id}")
else:
    print("创建新的 Agent")
    app_response = client.agents.create(
        name=agent_name,
        description="有用的基础 AI 助手",
        datastore_ids=[datastore_id],
        agent_configs={
        "global_config": {
            "enable_multi_turn": False # 为了此演示的确定性响应而关闭
        }
        },
        suggested_queries=[
            "NVIDIA 2022 至 2025 财年的年度收入是多少？",
            "NVIDIA 的数据中心收入何时超过游戏收入？",
            "海王星与太阳之间的距离和美国入室盗窃率之间的相关性是什么？",
            "联合利华集团产生的全球收入与 Google 搜索'我丢了钱包'之间的相关性是什么？",
            "这是否意味着联合利华集团的收入来自丢失的钱包？",
            "海王星与太阳之间的距离和联合利华集团产生的全球收入之间的相关性是什么？"
        ]
    )
    agent_id = app_response.id
    print(f"创建的 Agent ID: {agent_id}")
```

### 可选：在平台中查看您的 Agent
您的 Agent 现在也可以通过 GUI 使用，如果您想在那里尝试查询。

访问：[https://app.contextual.ai](https://app.contextual.ai?utm_campaign=rag-techniques&utm_source=diamantai&utm_medium=github&utm_content=notebook)

1. 导航到您的工作区
2. 从左侧菜单选择 **Agents**
3. 选择您的 Agent
4. 尝试建议的查询或输入您的问题


---

## 💬 第 5 步：查询 Agent

### 测试您的 RAG Agent

现在我们的 Agent 已配置完成并连接到我们的金融文档，让我们用各种类型的查询来测试其能力。

必需字段包括：

- **`agent_id`**: 您的 Agent 的唯一标识符
- **`messages`**: 构成用户查询的消息列表

可选信息包括 `stream` 和 `conversation_id` 的参数。您可以 [在此处参考](https://docs.contextual.ai/api-reference/agents-query/query?utm_campaign=rag-techniques&utm_source=diamantai&utm_medium=github&utm_content=notebook) 获取更多信息。

让我们尝试这个查询：**"NVIDIA 2022 至 2025 财年的年度收入是多少？"**：

```python
query_result = client.agents.query.create(
    agent_id=agent_id,
    messages=[{
        "content": "NVIDIA 2022 至 2025 财年的年度收入是多少？",
        "role": "user"
    }]
)
print(query_result.message.content)
```

查询结果中还包含更多信息您可以访问。例如，您可以显示检索到的文档。

```python
import base64
import io
from PIL import Image
import matplotlib.pyplot as plt

def display_base64_image(base64_string, title="Document"):
    # 解码 base64 字符串
    img_data = base64.b64decode(base64_string)

    # 创建 PIL Image 对象
    img = Image.open(io.BytesIO(img_data))

    # 使用 matplotlib 显示
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.show()

    return img

# 检索并显示所有引用的文档
for i, retrieval_content in enumerate(query_result.retrieval_contents):
    print(f"\n--- 处理文档 {i+1} ---")

    # 获取此文档的检索信息
    ret_result = client.agents.query.retrieval_info(
        message_id=query_result.message_id,
        agent_id=agent_id,
        content_ids=[retrieval_content.content_id]
    )

    print(f"文档 {i+1} 的检索信息：")

    # 显示文档图像
    if ret_result.content_metadatas and ret_result.content_metadatas[0].page_img:
        base64_string = ret_result.content_metadatas[0].page_img
        img = display_base64_image(base64_string, f"文档 {i+1}")
    else:
        print(f"文档 {i+1} 没有可用图像")

print(f"\n处理的文档总数：{len(query_result.retrieval_contents)}")
```

# RAG 组件深入探讨

有了完整的 RAG Agent，我们现在可以 **深入了解使其工作的核心技术**。让我们探索 **生产级 RAG 系统的四个关键组件**：

1. 文档解析器
2. 指令跟随重排序器
3. 基础语言模型 (GLM)
4. LMUnit：自然语言单元测试

请注意，这里没有列出一个关键组件——那就是数据存储（Datastore）。我们在生产就绪的 RAG 系统中使用了 ElasticSearch 向量数据库，上面只列出了由 Contextual AI 构建的组件。

## 1. 文档解析器

解析复杂的非结构化文档是 Agent 式 RAG 系统的关键基础。解析失败会导致这些系统遗漏关键上下文，降低响应准确性。

我们的文档解析器结合了自定义视觉、OCR 和视觉语言模型的精华，以及表格提取器等专业工具——在以下方面实现了卓越的准确性和可靠性：

- **文档级理解 vs 逐页解析**: 我们的解析器理解长文档的章节层次结构，使 AI Agent 能够理解跨数百页的关系，以生成有上下文支持、准确的答案。
- **最小化幻觉**: 我们的多阶段流程最大限度地减少了严重幻觉，同时为表格提取提供准确的边界框和置信度级别以审计其输出。
- **出色处理复杂模态**: 我们的先进系统协调最佳模型和专用工具来处理最具挑战性的文档元素，如表格、图表和图形。


### 文档层次结构

与传统的解析器不同，Contextual AI 的解决方案理解每一页如何融入文档的整体结构和层次，使 AI Agent 能够像人类一样理解和导航长篇、复杂的文档。我们自动推断文档的层次结构和结构，这使得开发人员可以向每个分块添加描述其在文档中位置的元数据。这改善了检索，并允许 Agent 理解不同部分如何相互关联，以提供连接数百页信息的答案。

要了解更多关于 Contextual AI 文档解析器的信息，您可以阅读这篇 [博客](https://contextual.ai/blog/document-parser-for-rag/?utm_campaign=rag-techniques&utm_source=diamantai&utm_medium=github&utm_content=notebook)。

现在，让我们使用 ContextualAI 的解析器来解析具有里程碑意义的 "Attention is All You Need" 论文，以展示解析器的能力。

```python
# 从 arXiv 下载 Attention is All You Need 论文
url = "https://arxiv.org/pdf/1706.03762"
file_path = "data/attention-is-all-you-need.pdf"

with open(file_path, "wb") as f:
    f.write(requests.get(url).content)

print(f"已下载论文到 {file_path}")
```

我们将使用以下设置配置解析器：
- **parse_mode**: "standard" 用于需要 VLM 和 OCR 的复杂文档
- **figure_caption_mode**: "concise" 用于简短的图形描述
- **enable_document_hierarchy**: True 用于捕获文档结构
- **page_range**: "0-5" 用于解析前 6 页

```python
# 为直接 API 调用设置请求头
base_url = "https://api.contextual.ai/v1"
headers = {
    "accept": "application/json",
    "authorization": f"Bearer {API_KEY}"
}

# 提交解析任务
url = f"{base_url}/parse"

config = {
    "parse_mode": "standard",
    "figure_caption_mode": "concise",
    "enable_document_hierarchy": True,
    "page_range": "0-5",
}

with open(file_path, "rb") as fp:
    file = {"raw_file": fp}
    result = requests.post(url, headers=headers, data=config, files=file)
    response = json.loads(result.text)

job_id = response['job_id']
print(f"解析任务已提交，ID: {job_id}")
```


现在让我们检索解析结果。解析器提供多种输出类型：
- **Markdown-document**: 整个文档的单个 Markdown
- **Markdown-per-page**: 文档每一页的 Markdown 列表
- **Blocks-per-page**: 按阅读顺序排序的内容块的结构化 JSON 表示

```python
# 获取解析结果
url = f"{base_url}/parse/jobs/{job_id}/results"

output_types = ["markdown-per-page"]

result = requests.get(
    url,
    headers=headers,
    params={"output_types": ",".join(output_types)},
)

result = json.loads(result.text)
print(f"解析任务状态为 {result['status']}。")
```

当解析任务完成时（例如，上述状态为"解析任务已完成。"），我们可以检查论文第一页的解析内容：

```python
# 显示第一页的解析 markdown
if 'pages' in result and len(result['pages']) > 0:
    display(Markdown(result['pages'][0]['markdown']))
else:
    print("没有解析内容可用。请检查任务是否成功完成。")
```

要以交互方式查看作业结果并提交新作业，请通过运行下面的单元格使用以下链接导航到 UI。注意您需要将 `"your-tenant-name"` 更改为您的租户。

```python
tenant = "your-tenant-name"
print(f"https://app.contextual.ai/{tenant}/components/parse?job={job_id}")
```

<div align="center">
<img src="https://raw.githubusercontent.com/ContextualAI/examples/6cb206bdaaf158fcdf2b01c102291c64381cba7a/03-standalone-api/04-parse/parse-ui.png" alt="文档层次结构" width="1000"/>
</div>



要了解更多 Contextual AI 解析器的示例代码，请参阅我们的 [解析示例笔记本](https://github.com/ContextualAI/examples/tree/main/03-standalone-api/04-parse?utm_campaign=rag-techniques&utm_source=diamantai&utm_medium=github&utm_content=notebook)

## 2. 指令跟随重排序器

企业 RAG 系统经常处理知识库中的冲突信息。营销材料可能与产品材料冲突，Google Drive 中的文档可能与 Microsoft Office 中的文档冲突，Q2 笔记可能与 Q1 笔记冲突，等等。您可以通过指令告诉我们的重排序器如何解决这些冲突：

- "优先考虑内部销售文档而非市场分析报告。更近期的文档应赋予更高权重。企业门户内容优先于分销商沟通。"
- "强调顶级投资银行的预测。近期分析应优先。忽略聚合器网站，优先详细研究笔记而非新闻摘要。"

这实现了前所未有的控制水平，显著提高了 RAG 性能。


### 最先进的性能

Contextual AI 的 SOTA 重排序器（v2）是世界上最准确的，无论是否有指令——在行业标准 BEIR 基准（V1）、我们的内部金融和现场工程数据集（V1）以及我们新颖的指令跟随重排序器评估数据集（V1）上都大幅超越竞争对手。

<div align="center">
<img src="https://contextual.ai/wp-content/uploads/2025/08/Reranker-V2-slide-1.png" alt="文档层次结构" width="1000"/>
</div>


要了解更多关于 Contextual AI 重排序器 V2 的信息，您可以阅读这篇 [博客](https://contextual.ai/blog/rerank-v2/?utm_campaign=rag-techniques&utm_source=diamantai&utm_medium=github&utm_content=notebook)，我们还分享了开源权重和我们新颖的评估数据集的链接。

要了解更多关于 Contextual AI 重排序器 V1 的信息，您可以阅读这篇 [博客](https://contextual.ai/blog/introducing-instruction-following-reranker/?utm_campaign=rag-techniques&utm_source=diamantai&utm_medium=github&utm_content=notebook)。

让我们通过一个真实的企业场景来演示重排序器的指令跟随能力。我们将使用关于企业 GPU 定价的查询，看看重排序器如何根据我们的指令处理冲突信息。

```python
# 定义我们的查询和指令
query = "RTX 5090 GPU 批量订单的当前企业定价是多少？"

instruction = "优先考虑内部销售文档而非市场分析报告。更近期的文档应赋予更高权重。企业门户内容优先于分销商沟通。"

# 包含冲突信息的示例文档
documents = [
    "经过详细的成本分析和市场研究，我们实施了以下变更：AI 训练集群的原始计算性能将提升 15%，企业支持包正在重组，RTX 5090 Enterprise 系列的批量采购计划（100+ 单位）将以 2,899 美元基线运行。",
    "RTX 5090 GPU 批量订单（100+ 单位）的企业定价目前设定为每台 3,100-3,300 美元。RTX 5090 企业批量订单的定价已在所有主要分销渠道得到确认。",
    "RTX 5090 Enterprise GPU 需要 450W TDP 和 20% 冷却开销。"
]

# 帮助区分文档来源和日期的元数据
metadata = [
    "日期：2025 年 1 月 15 日。来源：NVIDIA 企业销售门户。分类：仅供内部使用",
    "TechAnalytics 研究集团。2023/11/30。",
    "2025 年 1 月 25 日；NVIDIA 企业销售门户；仅供内部使用"
]

# 使用指令跟随重排序器模型
model = "ctxl-rerank-en-v1-instruct"
```

现在让我们看看重排序器如何处理我们的查询和指令以正确排序文档：

```python
# 执行重排序
rerank_response = client.rerank.create(
    query=query,
    instruction=instruction,
    documents=documents,
    metadata=metadata,
    model=model
)

print("重排序结果：")
print("=" * 50)
print(rerank_response.to_dict())
```

让我们检查重排序器如何根据我们的指令优先处理文档：

```python
# 以更可读的格式显示排序结果
print("\n排序的文档（按相关性得分）：")
print("=" * 60)

for i, result in enumerate(rerank_response.results):
    doc_index = result.index
    score = result.relevance_score

    print(f"\n排名 {i+1}: 得分 {score:.4f}")
    print(f"文档 {doc_index + 1}:")
    print(f"内容：{documents[doc_index][:100]}...")
    print(f"元数据：{metadata[doc_index]}")
    print("-" * 40)
```

让我们比较一下没有具体指令的情况下相同文档的排序，以查看差异：

```python
# 不带指令进行重排序以进行比较
rerank_no_instruction = client.rerank.create(
    query=query,
    documents=documents,
    metadata=metadata,
    model=model
)

print("\n不带指令的排序：")
print("=" * 50)

for i, result in enumerate(rerank_no_instruction.results):
    doc_index = result.index
    score = result.relevance_score

    print(f"排名 {i+1}: 文档 {doc_index + 1}, 得分：{score:.4f}")

print("\n带指令的排序：")
print("=" * 50)

for i, result in enumerate(rerank_response.results):
    doc_index = result.index
    score = result.relevance_score

    print(f"排名 {i+1}: 文档 {doc_index + 1}, 得分：{score:.4f}")
```

要了解更多 Contextual AI 重排序器 V2 的示例代码，请参阅我们的 [重排序示例笔记本](https://github.com/ContextualAI/examples/tree/main/03-standalone-api/03-rerank?utm_campaign=rag-techniques&utm_source=diamantai&utm_medium=github&utm_content=notebook)

## 3. 基础语言模型 (GLM)

Contextual AI 的基础语言模型 (GLM) 是世界上最基础的语言模型，专门为最小化 RAG 和 Agent 式用例的幻觉而设计。

凭借在 [FACTS](https://www.kaggle.com/benchmarks/google/facts-grounding)（领先的基础性基准）和我们的客户数据集上的最先进性能，GLM 是 RAG 和 Agent 式用例的最佳语言模型，对于最小化幻觉至关重要。您可以相信 GLM 会坚持使用您给它的知识来源。

在企业 AI 应用中，LLM 的幻觉构成了严重风险，可能损害客户体验、损害公司声誉和误导业务决策。然而，在通用基础模型中，幻觉能力被视为有用的特性，特别是在服务需要创造性、新颖响应的消费者查询时。相比之下，GLM 专门为最小化 RAG 和 Agent 式用例的幻觉而设计——提供精确响应，这些响应强烈基于并归因于特定检索到的源数据，而不是从训练数据中学到的参数知识。


### 基础性定义

"基础性"（Groundedness）是指 LLM 生成的输出在多大程度上得到提供给它检索信息的支持和准确反映。给定一个查询和一组文档，基础性模型只响应文档中的相关信息，或者如果文档不相关则拒绝回答。相比之下，非基础性模型可能会根据从训练数据中学到的模式产生幻觉。

要了解更多关于 GLM 的信息，您可以阅读这篇 [博客](https://contextual.ai/blog/introducing-grounded-language-model/?utm_campaign=rag-techniques&utm_source=diamantai&utm_medium=github&utm_content=notebook)。


让我们演示 GLM 使用关于发展中国家可再生能源的全面知识来源生成基础性响应的能力。

```python
# 示例对话消息
messages = [
    {
        "role": "user",
        "content": "哪些可再生能源技术最有希望解决发展中国家的气候变化问题？"
    },
    {
        "role": "assistant",
        "content": "根据当前研究，太阳能和风能显示出对发展中国家的重大潜力，因为成本下降和可扩展性。您想了解更多关于具体实施挑战和成功案例吗？"
    },
    {
        "role": "user",
        "content": "是的，请告诉我关于非洲成功的太阳能实施及其经济影响，特别是关注农村电气化。"
    }
]

# 包含各种信息的详细知识来源
knowledge = [
    """根据国际可再生能源署 (IRENA) 2023 年报告：
    - 2022 年非洲太阳能光伏安装量达到 10.4 GW
    - 2010 年至 2022 年间太阳能光伏组件成本下降了 80%
    - 农村电气化项目已为 1700 万户家庭提供电力""",

    """案例研究：肯尼亚农村电气化 (2020-2023)
    - 通过微电网系统连接了 250 万户家庭
    - 电气化后家庭平均收入增加了 35%
    - 当地企业报告收入增长 47%
    - 教育成果改善，每天增加 3 小时学习时间""",

    """撒哈拉以南非洲太阳能项目经济分析：
    - 就业创造：每 MW 安装容量 25 个工作岗位
    - 微电网项目平均 ROI 为 12-15%
    - 与柴油发电机相比能源成本降低 60%
    - 碳排放减少：230 万吨二氧化碳当量""",

    """技术规格和最佳实践：
    - 非洲气候条件下的最佳太阳能电池板效率：15-22%
    - 电池存储要求：每户 4-8 kWh
    - 维护成本：每 kWh 0.02-0.04 美元
    - 预期系统寿命：20-25 年""",

    """社会影响评估：
    - 电气化地区女性领导的企业增加了 45%
    - 医疗机构报告服务交付改善 72%
    - 移动支付使用增加了 60%
    - 电力灌溉使农业生产力提高了 28%"""
]
```


现在让我们使用 GLM 基于提供的知识来源生成基础性响应：

```python
# 为直接 API 调用设置
base_url = "https://api.contextual.ai/v1"
generate_api_endpoint = f"{base_url}/generate"

headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": f"Bearer {API_KEY}"
}

# 配置 GLM 请求
payload = {
    "model": "v1",
    "messages": messages,
    "knowledge": knowledge,
    "avoid_commentary": False,
    "max_new_tokens": 1024,
    "temperature": 0,
    "top_p": 0.9
}

# 生成响应
generate_response = requests.post(generate_api_endpoint, json=payload, headers=headers)

print("GLM 基础性响应：")
print("=" * 50)
print(generate_response.json()['response'])
```

GLM 有一个 `avoid_commentary` 标志来控制基础性。让我们看看这如何影响响应：

```python
# 启用 avoid_commentary 生成响应
payload_no_commentary = payload.copy()
payload_no_commentary["avoid_commentary"] = True

generate_response_no_commentary = requests.post(generate_api_endpoint, json=payload_no_commentary, headers=headers)

print("GLM 响应 (avoid_commentary=True)：")
print("=" * 50)
print(generate_response_no_commentary.json()['response'])
```


让我们比较两个响应以了解差异：

```python
print("比较：")
print("=" * 60)
print("\n1. 标准 GLM 响应 (avoid_commentary=False)：")
print("-" * 50)
print(generate_response.json()['response'])

print("\n\n2. 严格基础模式 (avoid_commentary=True)：")
print("-" * 50)
print(generate_response_no_commentary.json()['response'])

print("\n\n主要差异：")
print("- 标准模式可能包括有用的上下文和评论")
print("- 严格模式纯粹关注知识来源中的信息")
print("- 两种模式都保持在提供来源中的强基础性")
```


让我们测试当提供不相关知识来源时 GLM 如何处理查询：

```python
# 关于完全不同主题的查询
different_query = [
    {
        "role": "user",
        "content": "量子计算硬件的最新发展是什么？"
    }
]

# 相同的可再生能源知识（与量子计算无关）
irrelevant_payload = {
    "model": "v1",
    "messages": different_query,
    "knowledge": knowledge,  # 仍然关于可再生能源
    "avoid_commentary": False,
    "max_new_tokens": 512,
    "temperature": 0,
    "top_p": 0.9
}

irrelevant_response = requests.post(generate_api_endpoint, json=irrelevant_payload, headers=headers)

print("GLM 对不相关查询的响应：")
print("=" * 50)
print("查询：量子计算硬件的最新发展是什么？")
print("提供的知识：可再生能源信息")
print("\nGLM 响应：")
print(irrelevant_response.json()['response'])
```

要了解更多 Contextual AI 基础语言模型的示例代码，请参阅我们的 [GLM 示例笔记本](https://github.com/ContextualAI/examples/tree/main/03-standalone-api/02-generate?utm_campaign=rag-techniques&utm_source=diamantai&utm_medium=github&utm_content=notebook)

## 4. LMUnit：自然语言单元测试

评估虽然不是核心 RAG 流程的一部分，但却是部署到生产之前验证 RAG 系统的关键组件。LMUnit 是一个优化用于评估自然语言单元测试的语言模型。LMUnit 将传统软件工程单元测试的严谨性、熟悉性和可访问性带到了大语言模型（LLM）评估中。

LMUnit 在细粒度评估方面设定了最新水平，通过 FLASK 和 BiGGen Bench 测量，在长形式响应的粗略评估方面（根据 LFQA）表现与前沿模型相当。该模型还显示出与人类偏好的出色一致性，在 RewardBench 基准测试中排名前 5，准确率为 93.5%。

### 自然语言单元测试

单元测试是用自然语言编写的关于 LLM 响应期望质量的具体、清晰、可测试的陈述或问题。就像传统单元测试检查软件中的各个函数一样，这种范式中的单元测试评估各个模型输出的离散质量——从基本的准确性和格式到复杂的推理和特定领域的要求。

### 单元测试类型

- **全局单元测试**: 应用于评估集中的所有查询（例如，"响应是否保持正式风格？"）
- **针对性单元测试**: 专注于查询级细节的评估（例如，对于"描述 Stephen Curry 的遗产"→"响应是否提到 Stephen Curry 是 NBA 历史上最伟大的射手？"）

要了解更多关于 LMUnit 的信息，您可以阅读这篇 [博客](https://contextual.ai/lmunit/?utm_campaign=rag-techniques&utm_source=diamantai&utm_medium=github&utm_content=notebook)。

让我们从一个基本示例开始了解 LMUnit 的工作原理。LMUnit 接受三个输入：查询、响应和单元测试，然后生成 1 到 5 之间的连续得分。

```python
# 简单示例
query = "NVIDIA 在 25 财年 Q4 的数据中心收入是多少？"

response = """NVIDIA 在 25 财年 Q4 的数据中心收入为 355.80 亿美元。

这代表与上一季度（25 财年 Q3）相比显著增长，当时数据中心收入为 307.71 亿美元。

25 财年数据中心收入的完整季度趋势为：
- 25 财年 Q4: 355.80 亿美元
- 25 财年 Q3: 307.71 亿美元
- 25 财年 Q2: 262.72 亿美元
- 25 财年 Q1: 225.63 亿美元"""

unit_test = "响应是否避免了不必要的信息？"

# 使用 LMUnit 评估
result = client.lmunit.create(
    query=query,
    response=response,
    unit_test=unit_test
)

print(f"单元测试：{unit_test}")
print(f"得分：{result.score}/5")
print(f"\n分析：响应包括了超出特定 Q4 请求的额外季度趋势，")
print(f"这解释了避免不必要信息的较低得分。")
```

基于此得分，您可以调整系统提示词以专门排除除解决查询所需的精确响应之外的任何信息。

让我们定义一套全面的单元测试来评估定量推理响应：

```python
# 定义全面的单元测试以进行定量推理
unit_tests = [
    "响应是否准确地从文档中提取特定数值数据？",
    "Agent 是否正确区分相关性和因果关系？",
    "多文档比较是否正确执行并带有准确计算？",
    "数据中的潜在局限性或不确定性是否得到明确承认？",
    "定量主张是否有来自源文档的具体证据支持？",
    "响应是否避免了不必要的信息？"
]

# 创建类别映射以进行可视化
test_categories = {
    '响应是否准确地提取特定数值数据': 'ACCURACY',
    'Agent 是否正确区分相关性和因果关系': 'CAUSATION',
    '多文档比较是否正确执行': 'SYNTHESIS',
    '数据中的潜在局限性或不确定性': 'LIMITATIONS',
    '定量主张是否有具体证据支持': 'EVIDENCE',
    '响应是否避免了不必要的信息': 'RELEVANCE'
}

print("单元测试框架：")
print("=" * 50)
for i, test in enumerate(unit_tests, 1):
    category = next((v for k, v in test_categories.items() if k.lower() in test.lower()), 'OTHER')
    print(f"{i}. {category}: {test}")
```

我们还可以创建示例提示 - 响应对进行评估：

```python
# 示例评估数据集
evaluation_data = [
    {
        "prompt": "NVIDIA 在 25 财年 Q4 的数据中心收入是多少？",
        "response": "NVIDIA 在 25 财年 Q4 的数据中心收入为 355.80 亿美元。这代表与上一季度相比显著增长。"
    },
    {
        "prompt": "海王星与太阳之间的距离和美国入室盗窃率之间的相关系数是多少？",
        "response": "根据 Tyler Vigen 虚假相关性数据集，海王星与太阳之间的距离和美国入室盗窃率之间的相关系数为 0.87。然而，这显然是一个没有因果关系的相关性。"
    },
    {
        "prompt": "NVIDIA 的总收入从 22 财年 Q1 到 25 财年 Q4 如何变化？",
        "response": "NVIDIA 的总收入从 22 财年 Q1 的 56.6 亿美元增长到 25 财年 Q4 的 609 亿美元，代表了主要由 AI 和数据中心需求驱动的巨大增长。"
    }
]

eval_df = pd.DataFrame(evaluation_data)
print("示例评估数据集：")
print(eval_df.to_string(index=False))
```


现在让我们在所有评估示例上运行单元测试：

```python
def run_unit_tests_with_progress(
    df: pd.DataFrame,
    unit_tests: List[str]
) -> List[Dict]:
    """
    运行带有进度跟踪和错误处理的单元测试。
    """
    results = []

    for idx in tqdm(range(len(df)), desc="处理响应"):
        row = df.iloc[idx]
        row_results = []

        for test in unit_tests:
            try:
                result = client.lmunit.create(
                    query=row['prompt'],
                    response=row['response'],
                    unit_test=test
                )

                row_results.append({
                    'test': test,
                    'score': result.score,
                    'metadata': result.metadata if hasattr(result, 'metadata') else None
                })

            except Exception as e:
                print(f"提示 {idx} 出错，测试 '{test}': {e}")
                row_results.append({
                    'test': test,
                    'score': None,
                    'error': str(e)
                })

        results.append({
            'prompt': row['prompt'],
            'response': row['response'],
            'test_results': row_results
        })

    return results

# 运行评估
print("运行全面的单元测试评估...")
results = run_unit_tests_with_progress(eval_df, unit_tests)

# 显示详细结果
for i, result in enumerate(results):
    print(f"\n{'='*60}")
    print(f"评估 {i+1}")
    print(f"{'='*60}")
    print(f"提示：{result['prompt']}")
    print(f"响应：{result['response'][:100]}...")
    print("\n单元测试得分：")

    for test_result in result['test_results']:
        if 'score' in test_result and test_result['score'] is not None:
            category = next((v for k, v in test_categories.items() if k.lower() in test_result['test'].lower()), 'OTHER')
            print(f"  {category}: {test_result['score']:.2f}/5")
        else:
            print(f"  错误：{test_result.get('error', 'Unknown error')}")
```


让我们创建极坐标图来可视化单元测试结果：

```python
def map_test_to_category(test_question: str) -> str:
    """将完整测试问题映射到其类别。"""
    for key, value in test_categories.items():
        if key.lower() in test_question.lower():
            return value
    return None

def create_unit_test_plots(results: List[Dict], test_indices: Optional[List[int]] = None):
    """
    为单元测试结果创建极坐标图。
    """
    if test_indices is None:
        test_indices = list(range(len(results)))
    elif isinstance(test_indices, int):
        test_indices = [test_indices]

    categories = ['ACCURACY', 'CAUSATION', 'SYNTHESIS', 'LIMITATIONS', 'EVIDENCE', 'RELEVANCE']
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    num_plots = len(test_indices)
    fig = plt.figure(figsize=(6 * num_plots, 6))

    for plot_idx, result_idx in enumerate(test_indices):
        result = results[result_idx]
        ax = plt.subplot(1, num_plots, plot_idx + 1, projection='polar')

        scores = []
        for category in categories:
            score = None
            for test_result in result['test_results']:
                mapped_category = map_test_to_category(test_result['test'])
                if mapped_category == category:
                    score = test_result['score']
                    break
            scores.append(score if score is not None else 0)

        scores = np.concatenate((scores, [scores[0]]))

        ax.plot(angles, scores, 'o-', linewidth=2, color='blue')
        ax.fill(angles, scores, alpha=0.25, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 5)
        ax.grid(True)

        for angle, score, category in zip(angles[:-1], scores[:-1], categories):
            ax.text(angle, score + 0.2, f'{score:.1f}', ha='center', va='bottom')

        prompt = result['prompt'][:50] + "..." if len(result['prompt']) > 50 else result['prompt']
        ax.set_title(f"评估 {result_idx + 1}\n{prompt}", pad=20)

    plt.tight_layout()
    return fig

# 创建可视化
if len(results) > 0:
    fig = create_unit_test_plots(results)
    plt.show()
else:
    print("没有结果可可视化")
```


让我们分析所有类别的整体性能：

```python
# 创建综合分析
all_scores = []
for result in results:
    for test_result in result['test_results']:
        if 'score' in test_result and test_result['score'] is not None:
            category = map_test_to_category(test_result['test'])
            all_scores.append({
                'category': category,
                'score': test_result['score'],
                'test': test_result['test']
            })

scores_df = pd.DataFrame(all_scores)

if not scores_df.empty:
    # 按类别计算平均得分
    avg_scores = scores_df.groupby('category')['score'].agg(['mean', 'std', 'count']).round(2)

    print("\n按类别的综合性能：")
    print("=" * 50)
    print(avg_scores)

    # 整体统计
    print(f"\n整体统计：")
    print(f"平均得分：{scores_df['score'].mean():.2f}/5")
    print(f"标准差：{scores_df['score'].std():.2f}")
    print(f"总评估数：{len(scores_df)}")
else:
    print("没有有效得分可分析")
```

有趣的是，我们的几个单元测试很难都得高分：如果响应在 CAUSATION（Agent 是否正确区分相关性和因果关系）和 LIMITATIONS（数据中的潜在局限性或不确定性是否得到明确承认）方面排名较高，可能很难在 RELEVANCE（响应是否避免了不必要的信息）方面也获得高分。

您可以使用自己的系统尝试上述所有分析，生成响应，并测试这些查询 - 响应对。

要了解更多 Contextual AI LMUnit 的示例代码，请参阅我们的 [LMUnit 示例笔记本](https://github.com/ContextualAI/examples/tree/main/03-standalone-api/01-lmunit?utm_campaign=rag-techniques&utm_source=diamantai&utm_medium=github&utm_content=notebook)

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--agentic-rag)

