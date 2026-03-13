### 概述：
本代码实现了多模态RAG的多种方式之一。本项目处理PDF文件，使用Colpali检索相关内容，并使用多模态RAG系统生成答案。该过程包括文档索引、查询和使用Gemini模型进行内容生成。

### 关键组件：
- **RAGMultiModalModel**：用于文档索引和检索。
- **PDF处理**：下载并处理"Attention is All You Need"论文。
- **Gemini模型**：用于从检索的图像和查询生成内容。
- **Base64编码/解码**：管理搜索过程中检索的图像数据。

### 架构图：
   <img src="../images/multi_model_rag_with_colpali.svg" alt="多模态RAG" width="300">

### 动机：
实现对多模态文档（包含文本和图像的PDF）的高效查询和内容生成，以响应自然语言查询。

### 方法详情：
- **索引**：使用`RAGMultiModalModel`对PDF进行索引，同时存储文本和图像数据。
- **查询**：自然语言查询检索相关文档片段。
- **图像处理**：解码文档中的图像，显示并与Gemini模型结合用于内容生成。

### 优势：
- 支持文本和图像的多模态处理。
- 简化的检索和摘要管道。
- 使用先进LLM（Gemini模型）进行灵活的内容生成。

### 实现：
- 对PDF进行索引，将内容分割为文本和图像片段。
- 对索引文档运行查询以获取相关结果。
- 解码检索到的图像数据并通过Gemini模型生成答案。

### 总结：
本项目在多模态环境中集成了文档索引、检索和内容生成，能够对研究论文等复杂文档进行高效查询。

## 环境设置

```python
# 安装所需的包!pip install pillow python-dotenv
```

```python
!pip install -q git+https://github.com/huggingface/transformers.git qwen-vl-utils flash-attn optimum auto-gptq bitsandbytes
```

# 包安装和导入

下面的单元格安装运行此notebook所需的所有必要包。

```python
# 安装所需的包!pip install base64 byaldi os ragmultimodalmodel
```

# 包安装

下面的单元格安装运行此notebook所需的所有必要包。如果你在新环境中运行此notebook，请先执行此单元格以确保安装所有依赖项。

```python
# 安装所需的包!pip install byaldi
```

```python
import base64
import os
os.environ["HF_token"] = 'your-huggingface-api-key' # to download the ColPali model
from byaldi import RAGMultiModalModel
```

```python
RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2", verbose=1)
```

### 下载"Attention is All You Need"论文

```python
!wget https://arxiv.org/pdf/1706.03762
!mkdir docs
!mv 1706.03762 docs/attention_is_all_you_need.pdf
```

### 索引

```python
RAG.index(
    input_path="./docs/attention_is_all_you_need.pdf",
    index_name="attention_is_all_you_need",
    store_collection_with_index=True, # set this to false if you don't want to store the base64 representation
    overwrite=True
)
```

### 查询

```python
query = "What is the BLEU score of the Transformer (base model)?"
```

```python
results = RAG.search(query, k=1)
```

### 实际图像数据

```python
image_bytes = base64.b64decode(results[0].base64)
```

```python
filename = 'image.jpg'  # I assume you have a JPG file
with open(filename, 'wb') as f:
  f.write(image_bytes)
```

```python
from IPython.display import Image

display(Image(filename))
```

## 使用gemini-1.5-flash测试

```python
import google.generativeai as genai

genai.configure(api_key='your-api-key')
model = genai.GenerativeModel(model_name="gemini-1.5-flash")
```

```python
from PIL import Image
image = Image.open(filename)
```

```python
response = model.generate_content([image, query])
print(response.text)
```

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--multi-model-rag-with-colpali)
