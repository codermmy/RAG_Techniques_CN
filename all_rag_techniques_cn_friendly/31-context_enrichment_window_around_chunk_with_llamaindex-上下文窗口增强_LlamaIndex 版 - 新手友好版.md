# 🌟 新手入门：上下文窗口增强（LlamaIndex 版）

> **💡 给新手的说明**
> - **难度等级**：⭐⭐⭐☆☆（中级）
> - **预计时间**：35-50 分钟
> - **前置知识**：了解基础 RAG 概念，熟悉 LlamaIndex 基础用法
> - **学习目标**：理解上下文窗口增强的原理，掌握使用 LlamaIndex 实现更连贯的检索结果

---

## 📖 核心概念理解

### 什么是上下文窗口增强？

**上下文窗口增强**（Context Enrichment Window）是一种改进 RAG 检索质量的技术。它通过为每个检索到的文本块添加"前后文"，让回答更连贯、更完整。

### 🍕 通俗理解：读书的比喻

想象你在读一本很厚的书，有人问你一个问题：

1. **普通检索**：就像随机撕下一页纸给你看——你可能看不懂前因后果
2. **上下文窗口检索**：不仅给你看那一页，还把前后几页也给你——你能理解完整的故事线

**为什么需要上下文窗口？**

```
普通检索的问题：
文档被切成：[块 A][块 B][块 C][块 D][块 E]
                  ↓
             检索到块 C
                  ↓
        "他决定接受这个提议。"  ← 缺少上下文，不知道"他"是谁，"提议"是什么

上下文窗口检索：
                  ↓
        检索到块 C + 前后文
                  ↓
    "[块 B 的结尾] 张三考虑了很久。
     [块 C] 他决定接受这个提议。
     [块 D 的开头] 第二天，他开始了新工作。"
                  ↓
        现在你知道"他"是张三，"提议"是新工作！
```

### 🔑 核心组件解释

| 组件 | 作用 | 生活比喻 |
|------|------|----------|
| **SentenceWindowNodeParser** | 按句子分割并保存前后文 | 智能切书机，保留前后章节信息 |
| **SentenceSplitter** | 普通句子分割器 | 普通切书机 |
| **MetadataReplacementPostProcessor** | 查询时还原完整上下文 | 上下文拼接器 |
| **window_size** | 控制前后各保留几句 | 控制"前后文"的厚度 |

### 📊 技术原理图解

```
原文档：
"句子 1。句子 2。句子 3。句子 4。句子 5。句子 6。句子 7。"

普通分块（SentenceSplitter）：
块 1: "句子 1。句子 2。句子 3。"
块 2: "句子 4。句子 5。句子 6。"

上下文窗口分块（SentenceWindowNodeParser，window_size=1）：
块 1: "句子 2。"  ← 实际检索的是这句
     但元数据中包含："句子 1。句子 2。句子 3。"  ← 完整的上下文窗口

查询时，MetadataReplacementPostProcessor 会把完整窗口还原！
```

---

## 🛠️ 第一步：环境准备

### 📖 这是什么？

安装必要的 Python 库。这些与基础 RAG 教程类似。

### 💻 完整代码

```python
# ============================================
# 安装所需的包
# ============================================
# 每个包的作用：
# - faiss-cpu: Facebook 的高效相似度搜索库
# - llama-index: LlamaIndex 框架核心
# - python-dotenv: 管理 API 密钥

!pip install faiss-cpu llama-index python-dotenv
```

> **💡 代码解释**
> - 这些包在前面的教程中已经用过
> - 如果已经安装过，可以跳过这一步
>
> **⚠️ 新手注意**
> - 如使用国内网络，可添加清华源

---

## 🔑 第二步：配置 API 密钥和导入库

### 📖 这是什么？

设置 OpenAI API 密钥并导入所有需要的库，包括上下文窗口特有的组件。

### 💻 完整代码

```python
# ============================================
# 导入必要的库并配置 API 密钥
# ============================================
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
import faiss
import os
from dotenv import load_dotenv
from pprint import pprint  # 用于漂亮地打印字典

# 从 .env 文件加载环境变量
load_dotenv()

# 设置 OpenAI API 密钥
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# ============================================
# 配置 LlamaIndex 全局设置
# ============================================
EMBED_DIMENSION = 512  # Embedding 向量维度

# 设置使用的 LLM 模型
Settings.llm = OpenAI(model="gpt-3.5-turbo")

# 设置 Embedding 模型
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small", 
    dimensions=EMBED_DIMENSION
)
```

> **💡 代码解释**
> - `SentenceWindowNodeParser`：关键组件，按句子分割并保存上下文窗口
> - `MetadataReplacementPostProcessor`：查询时还原上下文的后处理器
> - `pprint`：用于格式化打印字典（查看元数据时很有用）

---

## 📄 第三步：下载和读取文档

### 📖 这是什么？

下载示例 PDF 文档并读取内容。

### 💻 完整代码

```python
# ============================================
# 创建目录并下载示例 PDF
# ============================================
import os

# 创建 data 目录
os.makedirs('data', exist_ok=True)

# 下载示例 PDF 文档（关于气候变化）
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf

# ============================================
# 读取 PDF 文档
# ============================================
path = "data/"
reader = SimpleDirectoryReader(
    input_dir=path, 
    required_exts=['.pdf']  # 只读取 PDF 文件
)
documents = reader.load_data()

print(f"✓ 文档加载完成！")
print(f"文档数量：{len(documents)}")
print(f"第一个文档的前 200 个字符：")
print(documents[0].text[:200])
```

> **💡 代码解释**
> - 这与基础 RAG 教程中的代码相同
> - `required_exts=['.pdf']` 限制只读取 PDF 文件

---

## 🗄️ 第四步：创建向量存储

### 📖 这是什么？

创建 FAISS 向量存储来保存文档的向量表示。

### 💻 完整代码

```python
# ============================================
# 创建 FAISS 向量存储
# ============================================
# 创建 FAISS 索引（使用 L2 距离）
fais_index = faiss.IndexFlatL2(EMBED_DIMENSION)

# 创建 LlamaIndex 的 FAISS 向量存储包装器
vector_store = FaissVectorStore(faiss_index=fais_index)
```

---

## 🔄 第五步：创建两种数据摄入流水线

### 📖 这是什么？

我们将创建两种不同的处理方式来进行对比：
1. **普通分块**：使用 `SentenceSplitter`，标准处理方式
2. **上下文窗口**：使用 `SentenceWindowNodeParser`，增强处理方式

### 💻 完整代码

#### 方式 1：普通句子分割器

```python
# ============================================
# 使用 SentenceSplitter 创建基础流水线
# ============================================
base_pipeline = IngestionPipeline(
    transformations=[SentenceSplitter()],  # 普通句子分割
    vector_store=vector_store
)

# 运行流水线
base_nodes = base_pipeline.run(documents=documents)

print(f"✓ 基础分块完成！共生成 {len(base_nodes)} 个节点")
```

> **💡 代码解释**
> - `SentenceSplitter()` 是标准的分块方式
> - 每个节点只包含分割后的文本内容

#### 方式 2：上下文窗口分割器

```python
# ============================================
# 使用 SentenceWindowNodeParser 创建上下文窗口流水线
# ============================================
node_parser = SentenceWindowNodeParser(
    # window_size: 要捕获的两侧句子数量
    # 设置为 3 意味着：前面 3 句 + 当前句 + 后面 3 句 = 共 7 句
    window_size=3,
    
    # window_metadata_key: 存储完整上下文窗口的元数据键名
    window_metadata_key="window",
    
    # original_text_metadata_key: 存储原始句子的元数据键名
    original_text_metadata_key="original_sentence"
)

# 创建流水线
pipeline = IngestionPipeline(
    transformations=[node_parser],
    vector_store=vector_store,
)

# 运行流水线
windowed_nodes = pipeline.run(documents=documents)

print(f"✓ 上下文窗口分块完成！共生成 {len(windowed_nodes)} 个节点")
```

> **💡 代码解释**
> - `window_size=3` 表示每个节点会保存前后各 3 个句子
> - 实际检索时只用"当前句"进行相似度匹配
> - 但元数据中保存了完整的"窗口"上下文
>
> **📊 window_size 设置建议**
> - `window_size=1`：轻量级，每边 1 句
> - `window_size=3`：中等，每边 3 句（推荐）
> - `window_size=5`：重量级，每边 5 句
>
> **⚠️ 新手注意**
> - `window_size` 越大，上下文越完整，但可能包含不相关信息
> - 检索时只有"当前句"用于匹配，所以不会影响检索准确性

### 💻 查看节点结构对比

```python
# ============================================
# 对比两种分块方式的节点结构
# ============================================
print("=" * 50)
print("普通分块的节点内容：")
print("=" * 50)
print(base_nodes[0].text[:300])
print(f"\n元数据：{base_nodes[0].metadata}")

print("\n" + "=" * 50)
print("上下文窗口分块的节点内容：")
print("=" * 50)
print(windowed_nodes[0].text[:100])  # 只有当前句
print(f"\n元数据键：{list(windowed_nodes[0].metadata.keys())}")
```

> **📊 预期输出**
> ```
> 普通分块的节点内容：
> "气候变化是一个复杂的问题，涉及多个因素。
> 主要原因包括化石燃料的燃烧、森林砍伐等。"
> 
> 元数据：{}
> 
> 上下文窗口分块的节点内容：
> "气候变化是一个复杂的问题，涉及多个因素。"
> 
> 元数据键：['window', 'original_sentence']
> ```

---

## 🔍 第六步：查询对比 - 普通检索 vs 上下文窗口检索

### 📖 这是什么？

通过实际查询来对比两种方法的效果差异。

### 💻 完整代码

```python
# ============================================
# 定义测试查询
# ============================================
query = "Explain the role of deforestation and fossil fuels in climate change"
print(f"测试查询：{query}\n")
```

### 方式 1：普通检索（无元数据替换）

```python
# ============================================
# 从基础节点创建索引和查询引擎
# ============================================
base_index = VectorStoreIndex(base_nodes)

base_query_engine = base_index.as_query_engine(
    similarity_top_k=1,  # 只返回最相关的 1 个结果
)

# 执行查询
base_response = base_query_engine.query(query)

print("=" * 50)
print("【普通检索结果】")
print("=" * 50)
print(f"答案：{base_response}\n")

# 查看检索到的源节点元数据
print("检索节点的元数据：")
pprint(base_response.source_nodes[0].node.metadata)
```

> **💡 代码解释**
> - `similarity_top_k=1` 只返回最相关的 1 个结果
> - 普通检索返回的只有分割后的文本块

### 方式 2：上下文窗口检索（带元数据替换）

```python
# ============================================
# 从上下文窗口节点创建索引和查询引擎
# ============================================
windowed_index = VectorStoreIndex(windowed_nodes)

# 使用 MetadataReplacementPostProcessor 实例化查询引擎
windowed_query_engine = windowed_index.as_query_engine(
    similarity_top_k=1,
    node_postprocessors=[
        MetadataReplacementPostProcessor(
            target_metadata_key="window"  # 使用 window 元数据替换原始内容
            # 这个 key 是在 SentenceWindowNodeParser 中定义的 window_metadata_key
        )
    ],
)

# 执行查询
windowed_response = windowed_query_engine.query(query)

print("=" * 50)
print("【上下文窗口检索结果】")
print("=" * 50)
print(f"答案：{windowed_response}\n")

# 查看检索到的源节点元数据
print("检索节点的元数据：")
print("(注意：窗口和原始句子都被添加到元数据中了)")
pprint(windowed_response.source_nodes[0].node.metadata)
```

> **💡 代码解释**
> - `MetadataReplacementPostProcessor` 是关键！
> - 它会在检索后，用元数据中的"window"内容替换原始节点内容
> - 这样 LLM 收到的就是包含上下文的完整文本
>
> **🔍 "元数据替换"的工作原理**
> ```
> 1. 检索阶段：用"当前句"的向量进行相似度匹配
> 2. 后处理阶段：用 MetadataReplacementPostProcessor
> 3. 替换操作：把节点内容替换为元数据中的"window"
> 4. 结果：LLM 看到的是包含上下文的完整文本
> ```

---

## 📊 第七步：效果对比分析

### 💻 完整代码

```python
# ============================================
# 对比两种方法的结果
# ============================================
print("=" * 60)
print("效果对比分析")
print("=" * 60)

print("\n【普通检索】")
print(f"检索到的文本长度：{len(base_response.source_nodes[0].node.text)} 字符")
print(f"文本内容预览：{base_response.source_nodes[0].node.text[:200]}...")

print("\n【上下文窗口检索】")
windowed_text = windowed_response.source_nodes[0].node.text
print(f"检索到的文本长度：{len(windowed_text)} 字符")
print(f"文本内容预览：{windowed_text[:300]}...")

print("\n" + "=" * 60)
print("分析结论：")
print("=" * 60)
print("上下文窗口检索返回的文本通常更长、更连贯")
print("因为它包含了检索句子的前后文！")
```

> **📊 预期输出示例**
> ```
> 【普通检索】
> 检索到的文本长度：150 字符
> 文本内容预览："森林砍伐减少了碳汇，加剧了气候变化。"...
> 
> 【上下文窗口检索】
> 检索到的文本长度：450 字符
> 文本内容预览："化石燃料的燃烧是温室气体排放的主要来源。
>               森林砍伐减少了碳汇，加剧了气候变化。
>               这两者 combined 导致了全球变暖加速。"...
> 
> 分析结论：
> 上下文窗口检索返回的文本通常更长、更连贯
> 因为它包含了检索句子的前后文！
> ```

---

## ⚠️ 常见问题与调试

### Q1: `window_size` 应该设置多大？

**建议**：
- **小文档/简单查询**：`window_size=1` 或 `2`
- **中等文档**：`window_size=3`（推荐）
- **大文档/复杂查询**：`window_size=5`

**调整策略**：
```python
# 如果回答太短、缺乏上下文 → 增大 window_size
node_parser = SentenceWindowNodeParser(window_size=5)

# 如果回答包含太多不相关信息 → 减小 window_size
node_parser = SentenceWindowNodeParser(window_size=1)
```

### Q2: 为什么需要 `MetadataReplacementPostProcessor`？

**解释**：
- `SentenceWindowNodeParser` 只在**存储时**保存上下文到元数据
- 但检索时默认只返回"当前句"的向量
- `MetadataReplacementPostProcessor` 在**查询后**把上下文还原

**流程**：
```
存储：当前句 → 元数据保存完整窗口
      ↓
检索：用当前句向量匹配
      ↓
后处理：用元数据中的窗口替换当前句  ← 这里发生！
      ↓
输出：完整上下文窗口
```

### Q3: 可以同时使用多个后处理器吗？

**可以！**
```python
from llama_index.core.postprocessor import SimilarityPostprocessor

query_engine = index.as_query_engine(
    similarity_top_k=5,
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window"),
        SimilarityPostprocessor(similarity_cutoff=0.7),  # 过滤低相似度
    ],
)
```

### Q4: 这个技术适合中文吗？

**适合！** 但需要注意：
- LlamaIndex 的句子分割对中文支持可能不如英文
- 可以考虑使用自定义的句子分割器

```python
# 中文友好的句子分割
import re

def chinese_sentence_split(text):
    # 使用中文标点分割
    sentences = re.split(r'(?<=[。！？.!?])\s*', text)
    return [s.strip() for s in sentences if s.strip()]
```

---

## 📚 总结

### 核心要点回顾

1. **上下文窗口增强的核心价值**：
   - 解决普通检索返回"孤立片段"的问题
   - 提供连贯、完整的上下文信息

2. **关键组件**：
   - `SentenceWindowNodeParser`：分割时保存上下文
   - `MetadataReplacementPostProcessor`：查询时还原上下文

3. **工作流程**：
   ```
   文档 → SentenceWindowNodeParser → 节点（含上下文元数据）
         ↓
   查询 → 向量检索 → MetadataReplacementPostProcessor
         ↓
   输出完整上下文窗口
   ```

### 进阶方向

1. **自定义窗口大小**：根据文档类型动态调整
2. **多层窗口**：同时使用不同大小的窗口
3. **混合检索**：结合关键词和向量检索

### 实际应用建议

- **法律文书**：需要完整上下文，适合用大窗口
- **技术文档**：可能只需要精确片段，小窗口即可
- **对话系统**：中等窗口，保持对话连贯性

---

## 🔗 相关资源

- [SentenceWindowNodeParser 文档](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/#sentencewindownodeparser)
- [MetadataReplacementPostProcessor 文档](https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors/#metadatareplacementpostprocessor)
- [LlamaIndex 后处理器指南](https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/)

<div style="text-align: center;">
<img src="../images/vector-search-comparison_context_enrichment.svg" alt="上下文增强窗口对比" style="width:70%; height:auto;">
</div>
