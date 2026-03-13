# 使用标注的多模态 RAG

## 概述

本代码实现了一个多模态检索增强生成（RAG）系统，该系统结合了图像理解和文本检索。该系统使用视觉语言模型为图像生成描述性标注（caption），然后将这些标注与文本内容一起处理以进行检索。

## 动机

传统 RAG 系统通常只处理文本数据，忽略了图像等其他模态中包含的宝贵信息。这个多模态方法通过结合视觉和文本信息来解决这一限制，从而能够更全面地理解和使用多种内容类型。

## 关键组件

1. 图像处理：加载和准备图像进行处理
2. 视觉语言模型：使用如 GPT-4V 等模型为图像生成描述性标注
3. 文本处理和分块：将标注和文本内容分割成可管理的块
4. 向量存储创建：使用 embeddings 创建 FAISS 向量存储
5. 检索器设置：配置检索器以获取相关文本和图像信息

## 方法详情

### 图像处理

1. 从指定目录加载图像
2. 准备图像进行视觉语言模型处理

### 图像标注生成

1. 使用视觉语言模型（如 GPT-4V）分析每个图像
2. 为每个图像生成详细的文本描述（标注）
3. 这些标注捕获图像的关键视觉元素和上下文

### 文本处理

1. 将生成的标注与任何现有文本内容组合
2. 使用 RecursiveCharacterTextSplitter 将组合的文本分割成块
3. 确保块大小适合 embedding 和检索

### 向量存储创建

1. 使用 OpenAI embeddings 为文本块创建向量表示
2. 从这些 embeddings 创建 FAISS 向量存储
3. 存储允许高效相似性搜索

### 检索

1. 配置检索器以获取给定查询的最相关块
2. 检索的块可能包含来自原始文本和图像标注的信息

## 此方法的优势

1. **多模态理解**：结合视觉和文本信息以获得更全面的理解
2. **改进上下文**：图像标注提供可能增强检索质量的额外上下文
3. **灵活性**：系统可以处理各种类型的内容，包括纯文本、图像和混合文档
4. **可扩展性**：该方法可以扩展到包括其他模态，如音频或视频

## 实现细节

1. 视觉语言模型处理可能会产生 API 成本，应考虑批量处理
2. 图像标注质量直接影响检索效果
3. 系统可以在仅有文本或具有不同视觉内容密度的文档上运行

## 结论

多模态 RAG 与标注代表了 RAG 技术的重要进步，使系统能够利用视觉和文本信息。通过为图像生成文本描述并将它们纳入检索管道，该系统提供了更全面、更上下文感知的信息检索方法。这种方法在文档包含重要视觉元素的各个领域都有潜在应用，例如教育材料、技术文档或内容丰富的报告。

<div style="text-align: center;">

<img src="../images/multi_model_rag.svg" alt="multi-model RAG" style="width:100%; height:auto;">
</div>

# 包安装和导入

下面的单元格安装运行此笔记本所需的所有必要包。

```python
# 安装所需的包
!pip install langchain langchain-openai python-dotenv Pillow
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
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from typing import List
from PIL import Image
import base64
import io

# 原始路径追加已替换为 Colab 兼容性
from helper_functions import *
from evaluation.evalute_rag import *

# 从 .env 文件加载环境变量
load_dotenv()

# 设置 OpenAI API 密钥环境变量
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
```

### 定义文档和图像路径

```python
# 下载所需的数据文件
import os
os.makedirs('data', exist_ok=True)

# 下载此笔记本中使用的示例图像
!wget -O data/sample_image1.jpg https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/sample_image1.jpg
!wget -O data/sample_image2.jpg https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/sample_image2.jpg
```

```python
image_folder = "data"
text_content = "This is some sample text that provides context about the images."
```

### 图像加载和标注函数

```python
def load_images_from_folder(folder_path: str) -> List[Image.Image]:
    """
    从文件夹加载所有图像。

    Args:
        folder_path (str): 包含图像的文件夹路径。

    Returns:
        List[Image.Image]: 加载的图像列表。
    """
    images = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(valid_extensions):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
            images.append(img)

    return images

def encode_image_to_base64(image: Image.Image) -> str:
    """
    将 PIL 图像编码为 base64 字符串。

    Args:
        image (Image.Image): 要编码的 PIL 图像。

    Returns:
        str: Base64 编码的图像字符串。
    """
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def generate_caption(image: Image.Image, llm) -> str:
    """
    使用多模态 LLM 为图像生成描述性标注。

    Args:
        image (Image.Image): 要标注的图像。
        llm: 用于生成标注的视觉语言模型。

    Returns:
        str: 生成的图像标注。
    """
    # 将图像转换为 base64
    base64_image = encode_image_to_base64(image)

    # 创建标注提示
    caption_prompt = PromptTemplate(
        input_variables=["image"],
        template="请详细描述这张图像。包括重要元素、对象、颜色和任何显著特征。提供全面但简洁的标注，捕获图像的要点。"
    )

    # 使用视觉语言模型生成标注
    # 注意：实际实现取决于特定的视觉语言模型 API
    # 这是一个概念示例
    response = llm.generate_caption(image=base64_image, prompt=caption_prompt)

    return response
```

### 主管道

```python
def process_multimodal_content(image_folder: str, text_content: str, llm) -> VectorStore:
    """
    处理图像和文本内容以创建多模态向量存储。

    Args:
        image_folder (str): 包含图像的文件夹路径。
        text_content (str): 要包含的附加文本内容。
        llm: 用于生成图像标注的视觉语言模型。

    Returns:
        VectorStore: 包含多模态内容的 FAISS 向量存储。
    """
    # 加载图像
    images = load_images_from_folder(image_folder)
    print(f"从文件夹加载了{len(images)}张图像。")

    # 为每个图像生成标注
    captions = []
    for i, image in enumerate(images, 1):
        print(f"为图像{i}/{len(images)}生成标注...")
        caption = generate_caption(image, llm)
        captions.append(f"图像{i}标注：{caption}")

    # 将图像标注与文本内容组合
    all_content = "\n\n".join(captions) + "\n\n" + text_content

    # 将组合的内容分割为块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(all_content)

    # 为块创建 embeddings
    embeddings = OpenAIEmbeddings()

    # 创建向量存储
    vectorstore = FAISS.from_texts(chunks, embeddings)

    return vectorstore
```

### 示例用法

```python
# 初始化视觉语言模型（注意：实际实现取决于特定的 LLM）
# 这是一个概念示例
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

# 处理多模态内容并创建向量存储
vectorstore = process_multimodal_content(image_folder, text_content, llm)

# 设置检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 示例查询
query = "图像中显示什么内容？"
relevant_docs = retriever.get_relevant_documents(query)

print(f"\n查询：{query}")
print("\n检索到的相关文档:")
for i, doc in enumerate(relevant_docs, 1):
    print(f"\n文档{i}:")
    print(doc.page_content)
```

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--multi-model-rag-with-captioning)
