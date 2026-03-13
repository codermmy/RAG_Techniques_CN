# 🌟 新手入门：使用 Contextual AI 构建金融文档分析的 Agent 式 RAG 流程

> **💡 给新手的说明**：本教程在原版基础上添加了详细解释。每个章节都包含：
> - 📖 **通俗解释** - 用生活化例子解释概念
> - 💻 **完整代码** - 保留所有技术细节
> - ⚠️ **注意事项** - 新手容易踩坑的地方
>
> **难度**：⭐⭐⭐（需要基础 Python 知识）
> **预计时间**：30-45 分钟

---

## 📖 第一部分：核心概念理解

### 什么是 RAG？为什么需要它？

**通俗理解**：

想象你在参加一场特殊的考试...

**传统 AI（没有 RAG）** 就像一个**闭卷考试的学生**：
- 只能靠脑子里训练时记住的知识答题
- 如果问到它"没学过"的内容（比如 2024 年的新闻），它就答不上来
- 可能会"瞎编"（术语叫"幻觉"）

**带 RAG 的 AI** 就像一个**可以翻书考试的学生**：
- 先**检索 (Retrieval)** = 翻书找相关内容
- 再**增强生成 (Augmented Generation)** = 结合找到的资料组织答案
- 可以回答最新问题，因为它能"查资料"

### 什么是 Agent（智能体）式 RAG？

**通俗理解**：

**普通 RAG** = 一个**听话但死板的助手**
- 你问："英伟达 2022 到 2025 年的收入是多少？"
- 它直接搜索这句话，可能只找到某一年的数据

**Agent 式 RAG** = 一个**会思考的聪明助手**
- 你问同样的问题
- 它会先**分析**："这是要查询多年财务数据"
- 然后**规划**："我需要分别搜索 2022、2023、2024、2025 四年的数据，然后汇总"
- 最后**执行**：分别搜索，整理成表格给你

> **💡 关键理解**：Agent 式的核心价值在于**查询改写**——它能理解你的真实意图，而不是死板地匹配关键词。

---

## 🎯 本教程学习目标

完成本教程后，你将能够：

1. ✅ **理解 RAG 和 Agent 的核心概念** - 不只是会用，还懂原理
2. ✅ **创建一个金融文档分析 Agent** - 能帮你阅读财报、回答问题
3. ✅ **掌握 Contextual AI 的四大组件**：
   - **Parser（解析器）** - 把复杂 PDF 转成 AI 能理解的格式
   - **Reranker（重排序器）** - 从一堆资料中挑出最相关的
   - **GLM（基础语言模型）** - 生成准确答案，不瞎编
   - **LMUnit** - 自动测试你的 Agent 好不好用

---

## 🛠️ 环境设置

### 第一步：安装必要的工具包

**通俗解释**：
就像做饭需要锅碗瓢盆一样，我们需要准备一些 Python 工具包。

```python
# 安装 Contextual AI 集成和数据可视化所需的包
# 这行代码会自动从网上下载并安装所有需要的工具
%pip install contextual-client matplotlib tqdm requests pandas dotenv
```

> **⚠️ 新手注意**：
> - 如果你用的是 Google Colab，运行完这行后需要**重启运行时**
> - 如果安装失败，可能是网络问题，可以尝试换网络

### 第二步：导入工具

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

> **💡 这些导入是做什么的？**
> - `os`, `Path` - 处理文件和路径
> - `requests` - 从网上下载文件
> - `pandas` - 处理表格数据
> - `ContextualAI` - 连接 Contextual AI 平台的主要工具
> - `matplotlib`, `plt` - 画图用的
> - `tqdm` - 显示进度条

---

## 🔑 第 1 步：API 认证设置

### 📖 什么是 API 密钥？

**通俗理解**：

**API 密钥 = 进入 Contextual AI 平台的"门票" + "身份证"**

- 没有这个密钥，你就无法使用他们的服务
- 这个密钥代表"你是谁"，用来计费和权限控制
- **重要**：就像银行密码一样，不要分享给任何人！

### 获取 API 密钥的步骤（超详细）

**1. 访问网站**
- 打开浏览器，访问：[app.contextual.ai](https://app.contextual.ai?utm_campaign=rag-techniques&utm_source=diamantai&utm_medium=github&utm_content=notebook)
- 点击页面上的 **"Start Free"** 按钮

**2. 注册账户**
- 填写邮箱和密码
- 你会获得 **30 天免费试用**（不需要信用卡！）

**3. 获取 API 密钥**
- 登录后，在左侧菜单栏找到 **"API Keys"**
- 点击 **"Create API Key"**
- 系统会生成一串类似 `ctxl_xxxxxxx` 的字符
- **立刻复制保存！** 关闭页面后就看不到了

**4. 安全存储**
- 把密钥复制到安全的地方
- 推荐：密码管理器、加密笔记

<div align="center">
<img src="https://github.com/ContextualAI/examples/blob/main/images/API_Keys.png?raw=true" alt="API 密钥示例" width="800"/>
</div>

### 配置 API 密钥（安全做法）

**❌ 错误做法（不要这样写）：**
```python
# 不要把密钥直接写在代码里！
API_KEY = "ctxl_abc123456..."  # 危险！别人会看到
```

**✅ 正确做法（推荐）：**
```python
# 从 .env 加载 API 密钥
from dotenv import load_dotenv
import os

load_dotenv()  # 读取 .env 文件

# 使用您的 API 密钥初始化
API_KEY = os.getenv("CONTEXTUAL_API_KEY")
client = ContextualAI(
    api_key=API_KEY
)
```

### 如何创建 .env 文件？

**步骤 1**：在你的项目文件夹中，创建一个名为 `.env` 的文件（注意前面有个点）

**步骤 2**：在文件中写入：
```
CONTEXTUAL_API_KEY=你的密钥粘贴在这里
```

**步骤 3**：保存文件

**步骤 4**：运行上面的代码，它会自动读取

> **⚠️ 新手常见问题**：
> - Q: `.env` 文件应该放在哪里？A: 和你的代码在同一目录
> - Q: 为什么运行后说找不到 `CONTEXTUAL_API_KEY`？A: 检查文件名是不是 `.env`（不是 `.env.txt`）
> - Q: 密钥格式是什么样的？A: 通常以 `ctxl_` 开头

---

## 📊 第 2 步：创建文档数据存储

### 📖 什么是数据存储（Datastore）？

**通俗理解**：

**数据存储（Datastore）= 专门存放文档的"图书馆"**

想象一下：
- 你要让 AI 分析很多金融文档（财报、报告等）
- 这些文档需要放在一个专门的地方
- AI 需要"整理"这些文档（分块、建索引）才能快速查找
- 这个"整理好的图书馆"就是 **Datastore**

**它的作用**：
1. 📦 **存储文档** - 把你的 PDF、Word 等文件存进去
2. 🔍 **自动整理** - 自动把文档分成小块，建好索引
3. 🚀 **快速搜索** - AI 可以快速找到相关内容

### 创建数据存储的代码

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

> **💡 代码解释**：
> - 先检查是否已经创建过（避免重复创建）
> - 如果存在，就使用现有的
> - 如果不存在，就创建一个新的

> **⚠️ 新手注意**：
> - `datastore_id` 很重要，后面创建 Agent 时会用到
> - 每个数据存储的名字必须是唯一的

---

## 📄 第 3 步：文档摄取和处理

### 📖 什么是文档摄取？

**通俗理解**：

**文档摄取 = 把文档放进"图书馆"并让它变得可搜索**

Contextual AI 的厉害之处在于能处理各种复杂文档：
- 📊 **复杂表格** - 财务报表、数据表格
- 📈 **图表和图形** - 折线图、柱状图、饼图
- 📝 **多页文档** - 几十页的年报

它会自动：
1. 识别文档结构（哪里是标题、哪里是表格）
2. 提取文字和数据
3. 建立索引方便搜索

### 下载示例文档

本教程使用 4 个示例金融文档：

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

> **💡 这些文档的作用**：
> - **文档 A 和 B**：NVIDIA 的财报数据，用来测试 AI 能否正确提取收入数字
> - **文档 C 和 D**：有趣的相关性报告，用来测试 AI 能否区分"相关性"和"因果关系"

### 上传文档到数据存储

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

> **⚠️ 新手注意**：
> - 下载可能需要一些时间（取决于网络）
> - 上传后文档需要处理几分钟，状态会显示 `processing`
> - 处理完成后状态会变成 `completed`

### 查看文档处理状态

```python
# 查看第一个文档的处理状态
metadata = client.datastores.documents.metadata(datastore_id = datastore_id, document_id = document_ids[0])
print("文档元数据:", metadata)
```

> **💡 元数据包含什么？**
> - 文档名称、大小
> - 处理状态（processing/completed）
> - 页数、分块数量
> - 创建时间等

### 📱 可选：在网页界面查看

如果你想用图形界面查看文档：

1. 访问 [https://app.contextual.ai](https://app.contextual.ai)
2. 登录你的账户
3. 左侧菜单选择 **Datastores**
4. 选择你创建的数据存储
5. 选择 **Documents**
6. 点击 **Inspect** 查看文档详情

---

## 🤖 第 4 步：Agent 创建和配置

### 📖 什么是 Agent？

**通俗理解**：

**Agent = 图书管理员 + 分析师**

- **Datastore** = 图书馆（存放文档）
- **Agent** = 图书管理员（帮你找书、理解问题、组织答案）

这个"图书管理员"的特点是：
1. 🧠 **会思考** - 理解你的问题意图
2. 📚 **会查资料** - 从文档中找到相关信息
3. ✍️ **会写答案** - 组织语言回答问题
4. 📋 **有依据** - 告诉你答案来自哪里

### 创建 Agent

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
            "enable_multi_turn": False  # 为了此演示的确定性响应而关闭
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

> **💡 代码解释**：
> - `system_prompt` - 告诉 AI 应该如何回答（有点像"工作手册"）
> - `datastore_ids` - 告诉 Agent 去哪个"图书馆"找书
> - `suggested_queries` - 预设的问题示例，方便用户快速尝试

> **⚠️ 新手注意**：
> - `enable_multi_turn = False` 表示每次问答是独立的（不会记住上下文）
> - 如果想让 AI 记住之前的对话，可以改成 `True`

### 📱 可选：在网页界面查看 Agent

1. 访问 [https://app.contextual.ai](https://app.contextual.ai)
2. 左侧菜单选择 **Agents**
3. 选择你创建的 Agent
4. 可以点击预设的问题，或者输入自己的问题

---

## 💬 第 5 步：查询 Agent

### 测试你的 RAG Agent

```python
# 查询 Agent
query_result = client.agents.query.create(
    agent_id=agent_id,
    messages=[{
        "content": "NVIDIA 2022 至 2025 财年的年度收入是多少？",
        "role": "user"
    }]
)
print(query_result.message.content)
```

> **💡 运行后会看到什么？**
> AI 会分析 4 个文档，找到 NVIDIA 各财年的收入数据，然后整理成答案告诉你

> **⚠️ 新手注意**：
> - 如果回答不正确，可能是文档还没处理完
> - 可以等几分钟再试

### 查看 AI 参考了哪些文档

AI 回答问题时会告诉你它参考了哪些文档，这样你可以验证答案的准确性：

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

---

## 🔬 RAG 组件深入探讨

现在你已经有了一个完整的 RAG Agent，让我们深入了解使其工作的**四个核心技术组件**：

```
┌─────────────────────────────────────────────────────┐
│  生产级 RAG 系统的四个关键组件                        │
├─────────────────────────────────────────────────────┤
│  1. 文档解析器 (Parser) - 读懂复杂文档              │
│  2. 指令跟随重排序器 (Reranker) - 挑出最相关的      │
│  3. 基础语言模型 (GLM) - 生成准确答案不瞎编         │
│  4. LMUnit - 自动测试你的 Agent 好不好用            │
└─────────────────────────────────────────────────────┘
```

> **💡 重要说明**：
> 这四个组件都是 Contextual AI 提供的核心技术。它们建立在 ElasticSearch 向量数据库之上，但上面这四个组件是 Contextual AI 的核心创新。

---

## 1. 📄 文档解析器 (Parser)

### 📖 为什么需要解析器？

**通俗理解**：

想象你有一本厚厚的财报 PDF...

**普通解析器** = 只会"认字"
- 把 PDF 转成纯文字
- 但不知道哪里是表格、哪里是图表
- 不知道章节结构

**Contextual AI 解析器** = "理解"文档结构
- 知道哪里是标题、哪里是正文
- 能识别表格、图表、图形
- 理解章节层次结构
- 知道第 5 页的"收入"和第 20 页的"收入"是不是同一回事

### 解析器有多厉害？

- **文档级理解** - 理解整本书的结构，不只是单页
- **最小化幻觉** - 准确识别表格边界，不乱猜
- **处理复杂模态** - 表格、图表都能处理

### 实战：解析"Attention is All You Need"论文

这篇论文是深度学习领域的经典论文，结构复杂，非常适合测试解析器。

```python
# 从 arXiv 下载论文
url = "https://arxiv.org/pdf/1706.03762"
file_path = "data/attention-is-all-you-need.pdf"

with open(file_path, "wb") as f:
    f.write(requests.get(url).content)

print(f"已下载论文到 {file_path}")
```

### 配置解析器

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
    "parse_mode": "standard",  # 标准模式，处理复杂文档
    "figure_caption_mode": "concise",  # 简洁的图形描述
    "enable_document_hierarchy": True,  # 捕获文档结构
    "page_range": "0-5",  # 只解析前 6 页（演示用）
}

with open(file_path, "rb") as fp:
    file = {"raw_file": fp}
    result = requests.post(url, headers=headers, data=config, files=file)
    response = json.loads(result.text)

job_id = response['job_id']
print(f"解析任务已提交，ID: {job_id}")
```

> **💡 参数解释**：
> - `parse_mode: "standard"` - 使用 VLM 和 OCR 处理复杂文档
> - `enable_document_hierarchy: True` - 自动识别章节结构
> - `page_range: "0-5"` - 只解析前 6 页（节省时间）

### 获取解析结果

```python
# 获取解析结果
url = f"{base_url}/parse/jobs/{job_id}/results"

output_types = ["markdown-per-page"]  # 每一页的 Markdown

result = requests.get(
    url,
    headers=headers,
    params={"output_types": ",".join(output_types)},
)

result = json.loads(result.text)
print(f"解析任务状态为 {result['status']}。")
```

> **⚠️ 新手注意**：
> - 解析可能需要几秒钟到几分钟
> - 如果状态不是"完成"，等一会儿再检查

### 查看解析结果

```python
# 显示第一页的解析 markdown
if 'pages' in result and len(result['pages']) > 0:
    display(Markdown(result['pages'][0]['markdown']))
else:
    print("没有解析内容可用。请检查任务是否成功完成。")
```

> **💡 你会看到**：
> 解析后的 Markdown 格式内容，包括标题、表格、公式等

### 📱 可选：在网页界面查看

```python
tenant = "your-tenant-name"  # 改成你的租户名
print(f"https://app.contextual.ai/{tenant}/components/parse?job={job_id}")
```

<div align="center">
<img src="https://raw.githubusercontent.com/ContextualAI/examples/6cb206bdaaf158fcdf2b01c102291c64381cba7a/03-standalone-api/04-parse/parse-ui.png" alt="文档层次结构" width="1000"/>
</div>

---

## 2. 🔄 指令跟随重排序器 (Reranker)

### 📖 为什么需要重排序器？

**通俗理解**：

想象你在网上搜索"苹果"...

**没有重排序器**：
- 返回的结果可能是：水果苹果、苹果公司、苹果电影...
- 乱七八糟混在一起

**有重排序器**：
- 根据你的具体问题，把最相关的结果排前面
- 如果你问"苹果股票"，公司相关的排前面
- 如果你问"苹果营养"，水果相关的排前面

### Contextual AI 的重排序器有什么特别？

**核心能力**：**指令跟随**

你可以给它"指令"，告诉它如何判断"相关性"：

> **示例指令**：
> "优先考虑内部销售文档而非市场分析报告。更近期的文档应赋予更高权重。企业门户内容优先于分销商沟通。"

这样，重排序器就会：
1. 识别哪些是"内部销售文档"
2. 识别文档的"日期"
3. 识别哪些是"企业门户内容"
4. 根据这些规则排序

### 实战：企业 GPU 定价查询

**场景**：公司想知道 RTX 5090 GPU 的企业采购价格，但网上信息鱼龙混杂...

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

> **💡 这些文档的"陷阱"**：
> - 文档 1：内部文档，最新（2025 年 1 月），价格 2899 美元
> - 文档 2：外部分析报告，旧（2023 年），价格 3100-3300 美元
> - 文档 3：内部文档，但只讲技术规格，不讲价格
>
> **正确答案**：应该优先选择文档 1（内部、最新、有价格）

### 执行重排序

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

### 查看排序结果

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

> **💡 预期结果**：
> 文档 1 应该排第一（因为它符合指令：内部文档、最新、有价格信息）

### 对比：有无指令的差异

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

> **💡 你会发现**：
> 有指令时，内部最新文档排第一；没有指令时，可能是其他文档排第一

---

## 3. 🧠 基础语言模型 (GLM)

### 📖 什么是 GLM？为什么需要它？

**通俗理解**：

**GLM（Grounded Language Model）= 不会瞎编的 AI**

**问题背景**：
普通 AI（LLM）有时会"瞎编"（术语叫"幻觉"）：
- 问："2024 年 NVIDIA 的收入是多少？"
- 瞎编："约 500 亿美元"（实际可能完全不同）

**GLM 的解决方案**：
- 只根据提供的资料回答
- 资料里没有的答案，就直接说"我不知道"
- 不瞎编、不猜测

### 什么是"基础性"（Groundedness）？

**基础性** = AI 的回答在多大程度上是基于提供的资料

- **高基础性** = 回答完全来自资料，有根有据
- **低基础性** = 回答是 AI 自己编的，不可信

### 实战：可再生能源问答

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
    - 非洲气候条件下的太阳能电池板效率：15-22%
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

### 使用 GLM 生成基础性响应

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
    "avoid_commentary": False,  # 允许一些评论
    "max_new_tokens": 1024,
    "temperature": 0,  # 温度 0 = 最确定、最不随机
    "top_p": 0.9
}

# 生成响应
generate_response = requests.post(generate_api_endpoint, json=payload, headers=headers)

print("GLM 基础性响应：")
print("=" * 50)
print(generate_response.json()['response'])
```

### 控制基础性程度

GLM 有一个 `avoid_commentary` 参数：
- `False` = 允许一些补充说明
- `True` = 严格基于资料，不添加任何评论

```python
# 启用 avoid_commentary 生成响应
payload_no_commentary = payload.copy()
payload_no_commentary["avoid_commentary"] = True

generate_response_no_commentary = requests.post(generate_api_endpoint, json=payload_no_commentary, headers=headers)

print("GLM 响应 (avoid_commentary=True)：")
print("=" * 50)
print(generate_response_no_commentary.json()['response'])
```

### 比较两种模式

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

### 测试：当资料不包含答案时

**关键测试**：如果用户问的问题，提供的资料里没有答案，GLM 会怎么做？

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
    "knowledge": knowledge,  # 还是关于可再生能源的资料
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

> **💡 预期结果**：
> GLM 应该说类似"提供的资料不包含量子计算相关信息，我无法回答这个问题"
>
> 而**不会**瞎编一些量子计算的内容

---

## 4. 📊 LMUnit：自然语言单元测试

### 📖 什么是 LMUnit？为什么需要它？

**通俗理解**：

**LMUnit = AI 考试评分系统**

就像软件需要"单元测试"来验证功能是否正常一样，AI 系统也需要测试：

- **传统软件单元测试**：验证函数输出是否正确
- **AI 的单元测试**：验证 AI 回答是否准确、有用、不瞎编

### LMUnit 是怎么工作的？

LMUnit 接收三个输入：
1. **查询** - 用户问的问题
2. **响应** - AI 给出的答案
3. **单元测试** - 你要检查的质量标准

然后输出一个 **1-5 分的评分**：
- 5 分 = 完美符合
- 3 分 = 一般
- 1 分 = 完全不符合

### 单元测试类型

| 类型 | 说明 | 示例 |
|------|------|------|
| **全局单元测试** | 适用于所有问题 | "响应是否保持正式风格？" |
| **针对性单元测试** | 针对特定问题 | "是否提到 Stephen Curry 是 NBA 史上最伟大射手？" |

### 实战：测试定量推理能力

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

> **💡 评分解读**：
> - 用户只问 Q4 的数据
> - AI 回答了 Q4，但还额外给了 Q1-Q3 的数据
> - 虽然这些信息有用，但严格来说"不必要"
> - 得分可能是 3-4 分（不是最差，但也不是完美）

### 创建全面的单元测试框架

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
    '响应是否准确地提取特定数值数据': 'ACCURACY',  # 准确性
    'Agent 是否正确区分相关性和因果关系': 'CAUSATION',  # 因果关系
    '多文档比较是否正确执行': 'SYNTHESIS',  # 综合分析
    '数据中的潜在局限性或不确定性': 'LIMITATIONS',  # 局限性
    '定量主张是否有具体证据支持': 'EVIDENCE',  # 证据支持
    '响应是否避免了不必要的信息': 'RELEVANCE'  # 相关性
}

print("单元测试框架：")
print("=" * 50)
for i, test in enumerate(unit_tests, 1):
    category = next((v for k, v in test_categories.items() if k.lower() in test.lower()), 'OTHER')
    print(f"{i}. {category}: {test}")
```

### 创建测试数据集

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

### 批量运行单元测试

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
```

### 查看详细结果

```python
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

### 可视化测试结果（雷达图）

```python
def map_test_to_category(test_question: str) -> str:
    """将完整测试问题映射到其类别。"""
    for key, value in test_categories.items():
        if key.lower() in test_question.lower():
            return value
    return None

def create_unit_test_plots(results: List[Dict], test_indices: Optional[List[int]] = None):
    """
    为单元测试结果创建极坐标图（雷达图）。
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

### 综合性能分析

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

> **💡 有趣的发现**：
> 某些单元测试之间可能存在"权衡"：
> - 如果一个响应在 **CAUSATION**（区分相关性和因果关系）方面得分高
> - 并且在 **LIMITATIONS**（承认数据局限性）方面得分高
> - 可能在 **RELEVANCE**（避免不必要信息）方面得分较低
>
> 这是因为解释因果关系和局限性需要额外文字，可能显得"冗余"

---

## 🎉 恭喜完成！

你已经完成了整个新手教程！现在你应该理解了：

### ✅ 核心概念
- RAG 是什么，为什么需要它
- Agent 式 RAG 与普通 RAG 的区别
- 四个核心组件的作用

### ✅ 实践能力
- 创建数据存储和上传文档
- 创建和配置 Agent
- 查询 Agent 并查看参考文档
- 使用解析器处理复杂文档
- 使用重排序器进行智能排序
- 使用 GLM 生成准确答案
- 使用 LMUnit 测试 Agent 质量

### ✅ 下一步学习
- 尝试上传自己的文档
- 定制 Agent 的 system_prompt
- 调整重排序器的指令
- 创建自己的单元测试

---

## 📚 学习资源

- Contextual AI 官方文档：https://docs.contextual.ai
- 示例代码仓库：https://github.com/ContextualAI/examples
- 原版教程：https://github.com/NirDiamant/RAG_Techniques

---

> **💡 给新手的最后建议**：
>
> 1. **不要怕犯错** - 每个错误都是学习机会
> 2. **多动手实践** - 只看不练学不会
> 3. **善用官方文档** - 大部分问题都有答案
> 4. **加入社区** - 遇到问题可以问

**祝你学习顺利！🚀**
