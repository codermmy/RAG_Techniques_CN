# 🌟 新手入门：微软 GraphRAG

> **💡 给新手的说明**
> - **难度等级**：⭐⭐⭐☆☆（中等）
> - **预计学习时间**：45-60 分钟
> - **前置知识**：了解基本的 RAG 概念，有 Python 编程经验
> - **学完你将掌握**：如何使用微软开源的 GraphRAG 工具包构建企业级知识检索系统
>
> **🤔 为什么要学这个？** 这是微软官方开源的 GraphRAG 实现，相比学术版本更注重实用性和可扩展性，适合企业级应用！

---

## 📖 核心概念理解

### 什么是微软 GraphRAG？

**微软 GraphRAG** 是 Microsoft Research 开发的开源工具包，它利用**知识图谱**和**社区摘要**来增强 RAG 系统。

### 通俗理解：从"查字典"到"读百科全书"

#### 传统 RAG 的局限

想象你在一个巨大的图书馆找资料：

**传统 RAG 方式：**
> 你："我想了解 Elon Musk 的成就"
>
> 系统：给你一堆包含"Elon Musk"这个词的句子片段
>
> 问题：这些片段之间有什么关系？哪些是重要的？不知道。

#### GraphRAG 的方式

**微软 GraphRAG 方式：**
> 你："我想了解 Elon Musk 的成就"
>
> 系统：
> 1. 先识别出知识图谱中的关键实体：
>    - Elon Musk → Tesla (创始人)
>    - Elon Musk → SpaceX (创始人)
>    - Elon Musk → Twitter (收购者)
> 2. 按"社区"组织信息：
>    - 商业帝国社区
>    - 太空探索社区
>    - 社交媒体社区
> 3. 为每个社区生成摘要，然后回答你的问题

### 核心概念解释

| 术语 | 解释 | 生活化比喻 |
|------|------|-----------|
| **索引阶段** | 构建知识图谱的过程 | 图书管理员整理图书、编目录 |
| **查询阶段** | 使用图谱回答问题的过程 | 读者根据目录查找信息 |
| **社区检测** | 发现图中紧密连接的实体群 | 把相关的人分到同一个"圈子" |
| **社区摘要** | 为每个社区生成概括性描述 | 给每个圈子写一个简介 |
| **局部搜索** | 针对特定实体的深入查询 | 了解某个人的详细信息 |
| **全局搜索** | 针对整体概念的综合性查询 | 了解整个行业的发展概况 |

### 系统架构一览

```
┌─────────────────────────────────────────────────────────┐
│                    索引阶段 (一次性的)                    │
├─────────────────────────────────────────────────────────┤
│  文本分块 → 元素提取 → 图构建 → 社区检测 → 社区摘要      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                    查询阶段 (可多次使用)                   │
├─────────────────────────────────────────────────────────┤
│                    用户查询                              │
│              ↓                    ↓                      │
│        局部搜索 (特定实体)    全局搜索 (整体概念)         │
│              ↓                    ↓                      │
│        局部答案生成          全局答案综合                 │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ 第一步：环境准备

### 📖 这是什么？

安装微软 GraphRAG 所需的依赖包。微软提供了专门的 `graphrag` 包，简化了使用流程。

### 💻 完整代码

```python
# 安装核心 graphrag 包
!pip install graphrag

# 安装辅助工具包
# - beautifulsoup4: 网页爬虫，用于获取测试数据
# - openai: OpenAI API 客户端
# - python-dotenv: 环境变量管理
# - pyyaml: YAML 文件读写（用于配置）
!pip install beautifulsoup4 openai python-dotenv pyyaml
```

> **💡 代码解释**
>
> **graphrag 包包含什么？**
> - 自动化的知识图谱构建流程
> - 社区检测算法（Leiden 算法）
> - 社区摘要生成
> - 局部/全局搜索接口
>
> **⚠️ 新手注意**
> - `graphrag` 包比较大，安装可能需要几分钟
> - 如果安装失败，尝试升级 pip：`pip install --upgrade pip`
> - 建议使用 Python 3.9 或更高版本
>
> **❓ 常见问题**
>
> **Q: 需要 Azure 账号吗？**
>
> A: 不一定！GraphRAG 支持：
> - OpenAI API（个人账号即可）
> - Azure OpenAI（企业用户）
> 本教程使用 OpenAI API 演示。

### 导入库和设置环境

```python
from dotenv import load_dotenv
import os
from openai import AzureOpenAI, OpenAI

# 从 .env 文件加载环境变量
load_dotenv()

# 配置标志：True 使用 Azure，False 使用 OpenAI
AZURE = True  # 改成 False 以使用 OpenAI

if AZURE:
    # Azure OpenAI 配置
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    GPT4O_DEPLOYMENT_NAME = os.getenv("GPT4O_MODEL_NAME")
    TEXT_EMBEDDING_3_LARGE_NAME = os.getenv("TEXT_EMBEDDING_3_LARGE_DEPLOYMENT_NAME")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

    oai = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION
    )
else:
    # OpenAI API 配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    oai = OpenAI(api_key=OPENAI_API_KEY)
```

> **💡 代码解释**
>
> **环境变量说明：**
>
> | 变量名 | 用途 | 示例值 |
> |--------|------|--------|
> | `OPENAI_API_KEY` | OpenAI API 密钥 | `sk-xxx` |
> | `AZURE_OPENAI_API_KEY` | Azure API 密钥 | `xxx` |
> | `AZURE_OPENAI_ENDPOINT` | Azure 服务端点 | `https://xxx.openai.azure.com` |
> | `GPT4O_MODEL_NAME` | GPT-4o 部署名 | `gpt-4o` |
> | `TEXT_EMBEDDING_3_LARGE_DEPLOYMENT_NAME` | 嵌入模型部署名 | `text-embedding-3-large` |
>
> **⚠️ 新手注意**
> - 创建 `.env` 文件存储密钥（不要提交到 git！）
> - 格式：`KEY=VALUE`，每行一个
> - 示例 `.env` 文件内容：
> ```
> OPENAI_API_KEY=sk-your-api-key-here
> ```

---

## 🛠️ 第二步：获取测试数据

### 📖 这是什么？

我们需要一些文本来测试系统。这里使用维基百科上关于 Elon Musk 的页面作为示例。

### 💻 完整代码

```python
import requests
from bs4 import BeautifulSoup

# 维基百科页面 URL
url = "https://en.wikipedia.org/wiki/Elon_Musk"

# 发送 HTTP 请求获取页面内容
response = requests.get(url)

# 使用 BeautifulSoup 解析 HTML
soup = BeautifulSoup(response.text, "html.parser")

# 创建 data 目录（如果不存在）
if not os.path.exists('data'):
    os.makedirs('data')

# 保存或读取文本
if not os.path.exists('data/elon.md'):
    # 提取正文（去除"参见"部分）
    elon = soup.text.split('\nSee also')[0]

    # 写入文件
    with open('data/elon.md', 'w', encoding='utf-8') as f:
        f.write(elon)
    print(f"已保存 Elon Musk 维基百科页面，共 {len(elon)} 字符")
else:
    # 读取已有文件
    with open('data/elon.md', 'r', encoding='utf-8') as f:
        elon = f.read()
    print(f"已加载 Elon Musk 维基百科页面，共 {len(elon)} 字符")
```

> **💡 代码解释**
>
> **代码流程：**
> 1. `requests.get(url)`：下载网页
> 2. `BeautifulSoup(...)`：解析 HTML，提取文本
> 3. `split('\nSee also')[0]`：截取"参见"之前的正文部分
> 4. 保存到本地文件，避免重复下载
>
> **⚠️ 新手注意**
> - 网络请求可能需要几秒，耐心等待
> - 如果访问维基百科慢，可以替换成其他文本源
> - 文件编码用 `utf-8` 避免中文乱码

---

## 🛠️ 第三步：配置 GraphRAG

### 📖 这是什么？

GraphRAG 使用 YAML 文件进行配置。我们需要：
1. 初始化默认配置
2. 修改配置以使用我们的 API 密钥

### 💻 完整代码

```python
import yaml
import subprocess

# 检查并初始化配置目录
if not os.path.exists('data/graphrag'):
    # 运行初始化命令
    # --init 表示创建默认配置
    # --root 指定工作目录
    print("正在初始化 GraphRAG 配置...")
    result = subprocess.run(
        ['python', '-m', 'graphrag.index', '--init', '--root', 'data/graphrag'],
        capture_output=True,
        text=True
    )
    print(result.stdout)

# 读取默认配置
with open('data/graphrag/settings.yaml', 'r') as f:
    settings_yaml = yaml.load(f, Loader=yaml.FullLoader)

# 修改配置以使用我们的 API
print("正在配置 API 密钥...")

# 设置主 LLM 配置
settings_yaml['llm']['model'] = "gpt-4o"
settings_yaml['llm']['api_key'] = AZURE_OPENAI_API_KEY if AZURE else OPENAI_API_KEY
settings_yaml['llm']['type'] = 'azure_openai_chat' if AZURE else 'openai_chat'

# 设置嵌入模型配置
settings_yaml['embeddings']['llm']['api_key'] = AZURE_OPENAI_API_KEY if AZURE else OPENAI_API_KEY
settings_yaml['embeddings']['llm']['type'] = 'azure_openai_embedding' if AZURE else 'openai_embedding'
settings_yaml['embeddings']['llm']['model'] = TEXT_EMBEDDING_3_LARGE_NAME if AZURE else 'text-embedding-3-large'

# 如果是 Azure，需要额外配置
if AZURE:
    settings_yaml['llm']['api_version'] = AZURE_OPENAI_API_VERSION
    settings_yaml['llm']['deployment_name'] = GPT4O_DEPLOYMENT_NAME
    settings_yaml['llm']['api_base'] = AZURE_OPENAI_ENDPOINT
    settings_yaml['embeddings']['llm']['api_version'] = AZURE_OPENAI_API_VERSION
    settings_yaml['embeddings']['llm']['deployment_name'] = TEXT_EMBEDDING_3_LARGE_NAME
    settings_yaml['embeddings']['llm']['api_base'] = AZURE_OPENAI_ENDPOINT

# 保存修改后的配置
with open('data/graphrag/settings.yaml', 'w') as f:
    yaml.dump(settings_yaml, f)

print("✓ 配置完成！")

# 准备输入目录
if not os.path.exists('data/graphrag/input'):
    os.makedirs('data/graphrag/input')

    # 复制文本到输入目录
    import shutil
    shutil.copy('data/elon.md', 'data/graphrag/input/elon.txt')

    print("正在处理文档，这可能需要几分钟...")

    # 运行索引流程
    result = subprocess.run(
        ['python', '-m', 'graphrag.index', '--root', './data/graphrag'],
        capture_output=True,
        text=True
    )

    # 打印输出
    if result.returncode == 0:
        print("🚀 所有工作流已成功完成！")
    else:
        print(f"处理失败：{result.stderr}")
```

> **💡 代码解释**
>
> **GraphRAG 索引流程：**
>
> ```
> 输入文本
>    ↓
> 1. 文本分块 (Chunking)
>    ↓
> 2. 元素提取 (Extraction) - 识别实体和关系
>    ↓
> 3. 图构建 (Graph Building)
>    ↓
> 4. 社区检测 (Community Detection) - Leiden 算法
>    ↓
> 5. 社区摘要 (Community Summarization)
>    ↓
> 输出：知识图谱 + 社区摘要
> ```
>
> **⚠️ 新手注意**
> - 索引是**一次性**的过程，之后可以重复查询
> - 处理时间：取决于文本长度，Elon Musk 页面约需 3-5 分钟
> - **成本提示**：索引过程会调用多次 LLM，产生 API 费用
>
> **❓ 常见问题**
>
> **Q: 索引过程为什么这么慢？**
>
> A: 因为要执行多个步骤：
> 1. 提取实体（调用 LLM）
> 2. 提取关系（调用 LLM）
> 3. 生成社区摘要（调用 LLM）
> 4. 每一步都需要等待 API 响应
>
> **Q: 费用大概多少？**
>
> A: 以 Elon Musk 维基百科页面（约 15000 词）为例：
> - GPT-4o 调用：约 $0.5-1.0
> - 嵌入生成：约 $0.01
> 具体费用取决于文本长度和复杂度。

---

## 🛠️ 第四步：创建查询工具函数

### 📖 这是什么？

微软 GraphRAG 提供了 CLI（命令行接口）工具来查询。我们创建包装函数，方便在 Python 中调用。

### 💻 完整代码

```python
import subprocess
import re

# 默认配置
DEFAULT_RESPONSE_TYPE = 'Summarize and explain in 1-2 paragraphs with bullet points using at most 300 tokens'
DEFAULT_MAX_CONTEXT_TOKENS = 10000

def remove_data(text):
    """
    移除输出中的 [Data: ...] 标记
    这是 GraphRAG 的内部调试信息，不需要展示给用户
    """
    return re.sub(r'\[Data:.*?\]', '', text).strip()

def ask_graph(query, method):
    """
    向 GraphRAG 提问

    参数:
        query: 用户问题
        method: 搜索方法 ('local' 或 'global')

    返回:
        答案字符串

    工作原理:
        1. 设置环境变量（上下文长度限制）
        2. 调用 graphrag.query 命令行工具
        3. 解析输出，提取答案部分
    """
    # 设置环境变量
    env = os.environ.copy() | {
        'GRAPHRAG_GLOBAL_SEARCH_MAX_TOKENS': str(DEFAULT_MAX_CONTEXT_TOKENS),
    }

    # 构建命令
    command = [
        'python', '-m', 'graphrag.query',
        '--root', './data/graphrag',
        '--method', method,           # 'local' 或 'global'
        '--response_type', DEFAULT_RESPONSE_TYPE,
        query,                        # 用户问题
    ]

    # 执行命令
    output = subprocess.check_output(
        command,
        universal_newlines=True,      # 返回字符串而非字节
        env=env,
        stderr=subprocess.DEVNULL     # 隐藏错误输出
    )

    # 提取答案部分（移除 "Search Response: " 前缀）
    # 并清理 [Data: ...] 标记
    return remove_data(output.split('Search Response: ')[1])
```

> **💡 代码解释**
>
> **subprocess 模块：**
> - 用于在 Python 中调用外部命令
> - 这里调用的是 `graphrag.query` 模块
>
> **正则表达式 `re.sub`：**
> - `r'\[Data:.*?\]'` 匹配 `[Data: ...]` 格式的内容
> - `.*?` 是非贪婪匹配，匹配最短的内容
> - 替换成空字符串，即删除这些内容
>
> **⚠️ 新手注意**
> - `subprocess.check_output` 会在命令失败时抛出异常
> - `stderr=subprocess.DEVNULL` 隐藏错误信息，便于调试

---

## 🛠️ 第五步：局部搜索测试

### 📖 这是什么？

**局部搜索（Local Search）** 用于查询特定实体或概念。它会深入探索与该实体直接相关的信息。

### 💻 完整代码

```python
from IPython.display import Markdown

# 定义问题：关于 Elon Musk 创立的公司
local_query = "What and how many companies and subsidiaries founded by Elon Musk"

# 执行局部搜索
print(f"执行局部搜索：{local_query}")
local_result = ask_graph(local_query, 'local')

# 以 Markdown 格式显示结果（支持格式化）
display(Markdown(local_result))
```

> **💡 代码解释**
>
> **局部搜索特点：**
> - 针对**具体实体**的查询
> - 深入探索该实体的直接关系
> - 适合回答"谁"、"什么"、"何时"等具体问题
>
> **📊 预期输出示例：**
>
> ```markdown
> Elon Musk 创立或共同创立了多家公司：
>
> - **Zip2** (1995) - 与弟弟 Kimbal 共同创立，后被 Compaq 收购
> - **X.com/PayPal** (1999) - 在线支付公司
> - **SpaceX** (2002) - 航天探索公司，担任 CEO 和首席工程师
> - **Tesla** (2004 年加入) - 电动汽车公司，CEO 和产品架构师
> - **SolarCity** (2006) - 太阳能服务公司，后被 Tesla 收购
> - **OpenAI** (2015) - 非营利 AI 研究公司，共同创立者
> - **Neuralink** (2016) - 脑机接口公司
> - **The Boring Company** (2016) - 隧道建设公司
> - **xAI** (2023) - AI 公司
> ```
>
> **⚠️ 新手注意**
> - `display(Markdown(...))` 只在 Jupyter 中有效
> - 普通 Python 脚本用 `print(local_result)` 即可

---

## 🛠️ 第六步：全局搜索测试

### 📖 这是什么？

**全局搜索（Global Search）** 用于查询整体性、综合性的问题。它会遍历整个知识图谱，综合多个社区的信息。

### 💻 完整代码

```python
from IPython.display import Markdown

# 定义问题：关于 Elon Musk 的整体成就
global_query = "What are the major accomplishments of Elon Musk?"

# 执行全局搜索
print(f"执行全局搜索：{global_query}")
global_result = ask_graph(global_query, 'global')

# 以 Markdown 格式显示结果
display(Markdown(global_result))
```

> **💡 代码解释**
>
> **全局搜索特点：**
> - 针对**整体概念**的查询
> - 综合多个社区的信息
> - 适合回答"总结"、"评价"、"影响"等综合性问题
>
> **📊 预期输出示例：**
>
> ```markdown
> Elon Musk 的主要成就包括：
>
> **商业成就**
> - 建立了价值数千亿美元的商业帝国
> - Tesla 成为全球市值最高的汽车公司
> - SpaceX 实现了可回收火箭技术
>
> **技术贡献**
> - 推动了电动汽车的普及
> - 降低了航天发射成本
> - 推动了可持续能源发展
>
> **社会影响**
> - 改变了人们对太空探索的看法
> - 加速了全球向可持续能源的转型
> - 成为世界上最富有的人之一
> ```
>
> **❓ 常见问题**
>
> **Q: 局部搜索和全局搜索有什么区别？**
>
> A:
> | 特性 | 局部搜索 | 全局搜索 |
> |------|---------|---------|
> | 范围 | 特定实体及其直接关系 | 整个知识图谱 |
> | 速度 | 较快 | 较慢 |
> | 适用问题 | "X 创立了哪些公司？" | "X 的主要成就是什么？" |
> | 信息深度 | 深入 | 广泛 |

---

## 🎯 完整代码整合

### 一站式完整流程

```python
# ============== 环境准备 ==============
!pip install graphrag beautifulsoup4 openai python-dotenv pyyaml

import os
import re
import yaml
import subprocess
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
oai = OpenAI(api_key=OPENAI_API_KEY)

# ============== 获取数据 ==============
url = "https://en.wikipedia.org/wiki/Elon_Musk"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

os.makedirs('data', exist_ok=True)
if not os.path.exists('data/elon.md'):
    elon = soup.text.split('\nSee also')[0]
    with open('data/elon.md', 'w', encoding='utf-8') as f:
        f.write(elon)

# ============== 配置 GraphRAG ==============
if not os.path.exists('data/graphrag'):
    subprocess.run(
        ['python', '-m', 'graphrag.index', '--init', '--root', 'data/graphrag'],
        capture_output=True
    )

with open('data/graphrag/settings.yaml', 'r') as f:
    settings_yaml = yaml.load(f, Loader=yaml.FullLoader)

settings_yaml['llm']['model'] = "gpt-4o"
settings_yaml['llm']['api_key'] = OPENAI_API_KEY
settings_yaml['llm']['type'] = 'openai_chat'
settings_yaml['embeddings']['llm']['api_key'] = OPENAI_API_KEY
settings_yaml['embeddings']['llm']['type'] = 'openai_embedding'
settings_yaml['embeddings']['llm']['model'] = 'text-embedding-3-large'

with open('data/graphrag/settings.yaml', 'w') as f:
    yaml.dump(settings_yaml, f)

if not os.path.exists('data/graphrag/input'):
    os.makedirs('data/graphrag/input')
    os.system('cp data/elon.md data/graphrag/input/elon.txt')
    print("正在处理文档...")
    subprocess.run(
        ['python', '-m', 'graphrag.index', '--root', './data/graphrag'],
        capture_output=True
    )
    print("🚀 索引完成！")

# ============== 创建查询函数 ==============
DEFAULT_RESPONSE_TYPE = 'Summarize and explain in 1-2 paragraphs with bullet points'
DEFAULT_MAX_CONTEXT_TOKENS = 10000

def remove_data(text):
    return re.sub(r'\[Data:.*?\]', '', text).strip()

def ask_graph(query, method):
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
    output = subprocess.check_output(command, universal_newlines=True, env=env)
    return remove_data(output.split('Search Response: ')[1])

# ============== 测试查询 ==============
# 局部搜索
local_query = "What companies did Elon Musk found?"
local_result = ask_graph(local_query, 'local')
print(f"\n局部搜索结果：\n{local_result}")

# 全局搜索
global_query = "What are Elon Musk's major accomplishments?"
global_result = ask_graph(global_query, 'global')
print(f"\n全局搜索结果：\n{global_result}")
```

---

## 📚 GraphRAG 的优势

### 相比传统 RAG

| 特性 | 传统 RAG | 微软 GraphRAG |
|------|---------|--------------|
| **信息组织** | 扁平的文档块 | 层次化的知识图谱 |
| **关系理解** | 无 | 理解实体间关系 |
| **全局理解** | 弱 | 强（社区摘要） |
| **可解释性** | 低 | 高（可视化图谱） |
| **复杂查询** | 困难 | 擅长 |

### 适用场景

| 场景 | 是否适合 | 说明 |
|------|---------|------|
| 企业知识库检索 | ✅ 非常适合 | 需要理解文档间关系 |
| 研究文献分析 | ✅ 非常适合 | 需要综合多来源信息 |
| 法律文档审查 | ✅ 非常适合 | 需要追溯信息来源 |
| 简单 FAQ | ⚠️ 大材小用 | 传统 RAG 就够了 |
| 实时对话 | ⚠️ 需要预索引 | 索引是离线过程 |

---

## 🎓 术语解释表

| 术语 | 英文 | 解释 |
|------|------|------|
| 索引 | Index | 构建知识图谱的预处理过程 |
| 社区 | Community | 图中紧密连接的节点群 |
| Leiden 算法 | Leiden Algorithm | 社区检测算法，发现图中的聚类 |
| 局部搜索 | Local Search | 针对特定实体的深入查询 |
| 全局搜索 | Global Search | 针对整体概念的综合性查询 |
| 社区摘要 | Community Summary | 用 LLM 为每个社区生成的概括性描述 |
| YAML | YAML | 一种人类可读的配置格式 |
| CLI | Command Line Interface | 命令行接口 |

---

## ❓ 常见问题 FAQ

### Q1: GraphRAG 和本教程前面的 GraphRAG 有什么区别？

**A:** 主要区别：

| 特性 | 学术版 GraphRAG | 微软 GraphRAG |
|------|----------------|--------------|
| 来源 | 研究论文实现 | 微软官方开源 |
| 成熟度 | 教学演示 | 生产就绪 |
| 配置 | 代码配置 | YAML 配置 |
| 社区摘要 | 无 | 有（核心特性） |
| 文档 | 代码注释 | 完整文档 |

### Q2: 为什么索引过程需要这么久？

**A:** GraphRAG 索引包含多个步骤：

```
1. 文本分块 (快速)
2. 实体提取 (慢 - 调用 LLM)
3. 关系提取 (慢 - 调用 LLM)
4. 图构建 (中等)
5. 社区检测 (快速)
6. 社区摘要 (慢 - 调用 LLM)
```

其中步骤 2、3、6 需要调用 LLM，是主要耗时部分。

### Q3: 可以用中文文档吗？

**A:** 可以，但需要注意：
- GPT-4o 支持多语言，包括中文
- 嵌入模型 `text-embedding-3-large` 也支持中文
- 社区摘要提示词可能需要调整以优化中文输出

### Q4: 索引完成后可以添加新文档吗？

**A:** 可以，但需要：
- 将新文档放入 `input` 目录
- 重新运行索引（会增量更新）
- 或者删除旧索引重新构建

### Q5: 如何降低使用成本？

**A:** 建议：
1. 先用小样本测试（几页文档）
2. 调整配置降低 LLM 调用次数
3. 使用较便宜的模型（如 GPT-3.5-Turbo）
4. 缓存索引结果，避免重复构建

---

## 📊 配置文件详解

### 完整 settings.yaml 示例

```yaml
# 主 LLM 配置
llm:
  type: openai_chat       # 或 azure_openai_chat
  model: gpt-4o           # 模型名称
  api_key: ${OPENAI_API_KEY}  # 支持环境变量
  temperature: 0          # 输出确定性
  max_tokens: 4000        # 最大输出长度

# 嵌入模型配置
embeddings:
  llm:
    type: openai_embedding
    model: text-embedding-3-large
    api_key: ${OPENAI_API_KEY}

# 图构建配置
graph:
  # 实体提取配置
  extraction:
    entity_types:       # 要提取的实体类型
      - person
      - organization
      - location
      - event
    max_gleanings: 1    # 额外提取次数

# 社区检测配置
community:
  algorithm: leiden     # Leiden 算法
  max_level: 3          # 最大层级数

# 报告生成配置
reports:
  generate: true        # 生成社区摘要
```

---

## ✅ 学习检查清单

- [ ] 我理解了索引阶段和查询阶段的区别
- [ ] 我知道局部搜索和全局搜索的适用场景
- [ ] 我能配置 GraphRAG 的 YAML 文件
- [ ] 我理解了社区检测的作用
- [ ] 我知道如何运行索引和查询
- [ ] 我了解 GraphRAG 的优缺点

---

## 🚀 下一步学习建议

1. **尝试自己的数据**：用企业文档、研究论文等替换 Elon Musk 页面
2. **调整配置参数**：修改 `settings.yaml`，观察输出变化
3. **比较不同查询**：用局部和全局搜索问不同类型的问题
4. **学习 RAPTOR**：继续学习下一个教程，了解层次化摘要方法

---

> **💪 恭喜！** 你已经完成了微软 GraphRAG 的新手教程！现在你掌握了如何使用工业级 GraphRAG 工具包构建知识图谱增强检索系统。这是构建企业级智能问答系统的重要技能！
