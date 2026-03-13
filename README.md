# RAG 技术中文教程 🚀

> **📚 项目定位**：帮助中文开发者 **从零到一掌握并精通 RAG 技术** 的完整导航地图
>
> **🌐 原项目**：[NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques)
>
> **📖 新手友好版**：包含 35 篇通俗易懂的中文教程，适合 AI 新手系统学习

---

## 🎯 为什么需要这个教程？

### 你是否遇到过这些问题：

- ❌ **想学 RAG**，但英文教程读得吃力，概念理解不透彻
- ❌ **跟着代码做**，但不知道每个步骤背后的原理
- ❌ **技术名词多**，HyDE、HyPE、Rerank、GraphRAG...分不清什么时候用什么
- ❌ **代码能跑**，但换个场景就不会调整优化
- ❌ **想深入理解**，但缺乏系统性的学习路径

### 本教程的价值

```
🎯 零基础友好    →   生活化比喻 + 详细注释 + FAQ 解答
📚 系统性覆盖    →   35 种技术，从基础到高级，完整学习路径
💻 代码可运行    →   保留所有原始代码，每行都有中文注释
🚀 实战为导向    →   每个技术都有应用场景和避坑指南
```

---

## 📋 快速开始

### 环境准备

```bash
# 1. 克隆仓库
git clone https://github.com/codermmy/RAG_Techniques_CN.git
cd RAG_Techniques_CN

# 2. 安装基础依赖
pip install langchain openai faiss-cpu python-dotenv
```

### 第一个 RAG 系统（5 分钟）

参考教程：[02-simple_rag-基础 RAG-新手友好版.md](all_rag_techniques_cn_friendly/02-simple_rag-基础 RAG-新手友好版.md)

---

## 🗺️ 学习路线图

```
                    RAG 技术学习路线
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   🟢 基础篇         🟡 进阶篇         🔴 高级篇
   (入门必读)       (核心技能)       (专家进阶)
    01-05 号        06-22 号         23-35 号
        │                │                │
        └────────────────┴────────────────┘
                         ▼
                  🏆 精通 RAG 技术
```

---

## 📚 教程目录

### 基础篇 (01-05)

| 序号 | 技术 | 教程链接 | 难度 |
|------|------|----------|------|
| 01 | Agent 式 RAG | [查看](all_rag_techniques_cn_friendly/01-Agentic_RAG-Agent 式 RAG 新手友好版.md) | ⭐⭐⭐ |
| 02 | 基础 RAG | [查看](all_rag_techniques_cn_friendly/02-simple_rag-基础 RAG-新手友好版.md) | ⭐ |
| 03 | CSV 文件 RAG | [查看](all_rag_techniques_cn_friendly/03-simple_csv_rag-CSV 文件 RAG-新手友好版.md) | ⭐ |
| 04 | 可靠 RAG | [查看](all_rag_techniques_cn_friendly/04-reliable_rag-可靠 RAG-新手友好版.md) | ⭐⭐ |
| 05 | 优化分块大小 | [查看](all_rag_techniques_cn_friendly/05-choose_chunk_size-优化分块大小 新手友好版.md) | ⭐ |

### 进阶篇 (06-12)

| 序号 | 技术 | 教程链接 | 难度 |
|------|------|----------|------|
| 06 | 命题分块 | [查看](all_rag_techniques_cn_friendly/06-proposition_chunking-命题分块 新手入门.md) | ⭐⭐ |
| 07 | 查询转换 | [查看](all_rag_techniques_cn_friendly/07-query_transformations-查询转换 新手入门.md) | ⭐⭐ |
| 08 | HyDe | [查看](all_rag_techniques_cn_friendly/08-HyDe_Hypothetical_Document_Embedding-假设性文档嵌入 新手入门.md) | ⭐⭐⭐ |
| 09 | HyPE | [查看](all_rag_techniques_cn_friendly/09-HyPE_Hypothetical_Prompt_Embeddings-假设性提示嵌入 新手入门.md) | ⭐⭐⭐ |
| 10 | 上下文分块标题 | [查看](all_rag_techniques_cn_friendly/10-contextual_chunk_headers-上下文分块头部 新手友好版.md) | ⭐⭐ |
| 11 | 相关片段提取 | [查看](all_rag_techniques_cn_friendly/11-relevant_segment_extraction-相关片段提取 新手友好版.md) | ⭐⭐⭐ |
| 12 | 上下文窗口增强 | [查看](all_rag_techniques_cn_friendly/12-context_enrichment_window_around_chunk-上下文窗口增强 新手友好版.md) | ⭐⭐ |

### 高级篇 (13-22)

<details>
<summary>点击展开 10 篇高级教程</summary>

| 序号 | 技术 | 教程链接 | 难度 |
|------|------|----------|------|
| 13 | 语义分块 | [查看](all_rag_techniques_cn_friendly/13-semantic_chunking-语义分块.md) | ⭐⭐⭐ |
| 14 | 上下文压缩 | [查看](all_rag_techniques_cn_friendly/14-contextual_compression-上下文压缩.md) | ⭐⭐⭐ |
| 15 | 文档增强 | [查看](all_rag_techniques_cn_friendly/15-document_augmentation-文档增强.md) | ⭐⭐⭐ |
| 16 | 融合检索 | [查看](all_rag_techniques_cn_friendly/16-fusion_retrieval-融合检索.md) | ⭐⭐⭐ |
| 17 | 重排序 | [查看](all_rag_techniques_cn_friendly/17-reranking-重排序.md) | ⭐⭐⭐ |
| 18 | 层次索引 | [查看](all_rag_techniques_cn_friendly/18-hierarchical_indices-层次索引 新手入门.md) | ⭐⭐⭐⭐ |
| 19 | 飞镖板检索 | [查看](all_rag_techniques_cn_friendly/19-dartboard-飞镖板检索 新手入门.md) | ⭐⭐⭐⭐ |
| 20 | 多模态 RAG | [查看](all_rag_techniques_cn_friendly/20-multi_model_rag_with_captioning-多模态 RAG 标注 新手入门.md) | ⭐⭐⭐⭐ |
| 21 | 反馈循环检索 | [查看](all_rag_techniques_cn_friendly/21-retrieval_with_feedback_loop-反馈循环检索.md) | ⭐⭐⭐ |
| 22 | 自适应检索 | [查看](all_rag_techniques_cn_friendly/22-adaptive_retrieval-自适应检索.md) | ⭐⭐⭐⭐ |

</details>

### 架构篇 (23-27)

<details>
<summary>点击展开 5 篇架构教程</summary>

| 序号 | 技术 | 教程链接 | 难度 |
|------|------|----------|------|
| 23 | 可解释检索 | [查看](all_rag_techniques_cn_friendly/23-explainable_retrieval-可解释检索_新手版.md) | ⭐⭐⭐ |
| 24 | 图 RAG | [查看](all_rag_techniques_cn_friendly/24-graph_rag-图 RAG_新手版.md) | ⭐⭐⭐⭐⭐ |
| 25 | 微软 GraphRAG | [查看](all_rag_techniques_cn_friendly/25-Microsoft_GraphRag-微软 GraphRAG_新手版.md) | ⭐⭐⭐⭐⭐ |
| 26 | RAPTOR | [查看](all_rag_techniques_cn_friendly/26-raptor-RAPTOR_新手版.md) | ⭐⭐⭐⭐ |
| 27 | Self-RAG | [查看](all_rag_techniques_cn_friendly/27-self_rag-自 RAG_新手版.md) | ⭐⭐⭐⭐ |

</details>

### LlamaIndex 版 (28-35)

<details>
<summary>点击展开 8 篇 LlamaIndex 实现教程</summary>

| 序号 | 技术 | 教程链接 | 难度 |
|------|------|----------|------|
| 28 | 校正 RAG | [查看](all_rag_techniques_cn_friendly/28-crag-校正 RAG-新手友好版.md) | ⭐⭐⭐⭐ |
| 29 | 基础 RAG (LlamaIndex) | [查看](all_rag_techniques_cn_friendly/29-simple_rag_with_llamaindex-基础 RAG_LlamaIndex 版 - 新手友好版.md) | ⭐⭐ |
| 30 | CSV RAG (LlamaIndex) | [查看](all_rag_techniques_cn_friendly/30-simple_csv_rag_with_llamaindex-CSV 文件 RAG_LlamaIndex 版 - 新手友好版.md) | ⭐⭐ |
| 31 | 上下文窗口增强 (LlamaIndex) | [查看](all_rag_techniques_cn_friendly/31-context_enrichment_window_around_chunk_with_llamaindex-上下文窗口增强_LlamaIndex 版 - 新手友好版.md) | ⭐⭐⭐ |
| 32 | 融合检索 (LlamaIndex) | [查看](all_rag_techniques_cn_friendly/32-fusion_retrieval_with_llamaindex-融合检索_LlamaIndex 版 - 新手友好版.md) | ⭐⭐⭐ |
| 33 | 重排序 (LlamaIndex) | [查看](all_rag_techniques_cn_friendly/33-reranking_with_llamaindex-重排序_LlamaIndex 版 - 新手友好版.md) | ⭐⭐⭐ |
| 34 | 图 RAG (Milvus) | [查看](all_rag_techniques_cn_friendly/34-graphrag_with_milvus_vectordb-图 RAG_Milvus 版 - 新手友好版.md) | ⭐⭐⭐⭐⭐ |
| 35 | 多模态 RAG (ColPali) | [查看](all_rag_techniques_cn_friendly/35-multi_model_rag_with_colpali-多模态 RAG_ColPali 版 - 新手友好版.md) | ⭐⭐⭐⭐ |

</details>

---

## 🎓 学习路径建议

### 🟢 入门级（0 基础）

```
第 1 步：02-Simple_RAG (理解基础流程)
    ↓
第 2 步：05-分块大小优化 (理解关键参数)
    ↓
第 3 步：16-融合检索 (提升检索质量)
    ↓
第 4 步：17-重排序 (进一步优化)
```

### 🟡 进阶级（有 RAG 经验）

```
第 1 步：06-命题分块 + 13-语义分块 (高级分块技术)
    ↓
第 2 步：07-查询转换 + 08-HyDe + 09-HyPE (查询增强)
    ↓
第 3 步：18-层次索引 + 26-RAPTOR (树形结构)
    ↓
第 4 步：24-图 RAG + 25-微软 GraphRAG (知识图谱)
```

### 🔴 专家级（构建生产系统）

```
第 1 步：27-Self-RAG + 28-CRAG (动态决策)
    ↓
第 2 步：22-自适应检索 (智能路由)
    ↓
第 3 步：21-反馈循环 (持续优化)
    ↓
第 4 步：23-可解释检索 (透明度与信任)
```

---

## 🛠️ 技术选型速查表

### 按应用场景选择

| 场景 | 推荐技术 | 文件编号 |
|------|----------|----------|
| **快速原型验证** | Simple RAG + 融合检索 + 重排序 | 02, 16, 17 |
| **企业文档检索** | 层次索引 + 上下文窗口 + 可靠 RAG | 18, 12, 04 |
| **金融/法律分析** | 图 RAG + 命题分块 + 可解释检索 | 24, 06, 23 |
| **客服问答系统** | Self-RAG + 查询转换 + 反馈循环 | 27, 07, 21 |
| **科研文献检索** | RAPTOR + 语义分块 + 自适应检索 | 26, 13, 22 |
| **多模态文档** | 多模态 RAG (标注版/ColPali) | 20, 35 |

---

## 📦 项目结构

```
RAG_Techniques_CN/
├── all_rag_techniques_cn_friendly/    # 新手友好版教程 (35 篇)
│   ├── 01-Agentic_RAG-Agent 式 RAG 新手友好版.md
│   ├── 02-simple_rag-基础 RAG-新手友好版.md
│   └── ... (共 35 篇)
├── all_rag_techniques/                # 原始英文教程
├── all_rag_techniques_cn/             # 原始中文翻译
├── README_CN.md                       # 原始中文 README
└── README.md                          # 本文件
```

---

## 🤝 与原项目关系

本项目是 [NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques) 的中文翻译和本地化版本。

**主要差异**：

| 特性 | 原项目 | 本中文项目 |
|------|--------|-----------|
| 语言 | 英文 | 简体中文 |
| 目标读者 | 有经验的开发者 | AI 新手和中文开发者 |
| 教程风格 | 技术文档风格 | 新手友好版（比喻 + 注释+FAQ） |
| 代码注释 | 英文 | 详细中文注释 |
| 学习路径 | 无明确路径 | 分阶段学习建议 |

**感谢原项目作者**：[NirDiamant](https://github.com/NirDiamant)

---

## 📖 每个教程的特色

每个新手友好版教程都包含：

```
┌─────────────────────────────────────────────────┐
│  🌟 新手入门：[技术名称]                         │
├─────────────────────────────────────────────────┤
│  💡 给新手的说明                                 │
│     - 难度等级：⭐⭐⭐☆☆                         │
│     - 预计时间：45-60 分钟                       │
│     - 前置知识：了解基础 RAG 概念                │
├─────────────────────────────────────────────────┤
│  📖 核心概念理解                                 │
│     - 生活化比喻（如"RAG=开卷考试"）             │
│     - 核心组件表格                               │
│     - 工作流程图解                               │
├─────────────────────────────────────────────────┤
│  🛠️ 分步教程（完整代码 + 详细注释）              │
├─────────────────────────────────────────────────┤
│  ⚠️ 新手注意                                    │
│     - 常见错误及解决方法                        │
│     - API 费用提示                               │
│     - 网络问题解决方案                          │
├─────────────────────────────────────────────────┤
│  ❓ 常见问题 FAQ                                 │
├─────────────────────────────────────────────────┤
│  📚 学习总结 + 🚀 下一步学习建议                 │
└─────────────────────────────────────────────────┘
```

---

## 🔗 相关资源

| 资源 | 链接 |
|------|------|
| 原项目仓库 | https://github.com/NirDiamant/RAG_Techniques |
| LlamaIndex 文档 | https://docs.llamaindex.ai |
| LangChain 文档 | https://python.langchain.com |
| OpenAI API 文档 | https://platform.openai.com/docs |

---

## 🎯 学完本教程后，你将能够：

```
✅ 理解 RAG 技术的完整技术栈
✅ 根据场景选择合适的 RAG 方案
✅ 独立实现和优化 RAG 系统
✅ 诊断和解决常见的 RAG 问题
✅ 评估和改进 RAG 系统性能
✅ 理解最新的 RAG 研究进展
```

---

## 📝 贡献指南

欢迎提交 Issue 和 Pull Request！

- 发现翻译错误或不准确？→ 提交 Issue
- 想改进某个教程？→ 提交 PR
- 有新的 RAG 技术想分享？→ 提交 PR

---

## 📄 许可证

本项目遵循原项目的许可证。详见 [LICENSE](LICENSE) 文件。

---

> **💪 开始你的 RAG 学习之旅吧！**
>
> 从 [02-simple_rag-基础 RAG-新手友好版.md](all_rag_techniques_cn_friendly/02-simple_rag-基础 RAG-新手友好版.md) 开始，循序渐进，你一定能掌握这项改变信息检索方式的技术！
>
> **📚 学习愉快！**
