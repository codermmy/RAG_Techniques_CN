#!/usr/bin/env python3
"""
翻译修复脚本 - 检查和修复 Jupyter Notebook 中的英文内容
- 翻译 Markdown 单元格中的英文
- 翻译代码注释中的英文
"""

import json
import re
import os
from pathlib import Path

# 翻译目录
CN_DIR = Path("/Users/maoyuan/code/opensource-project/RAG_Techniques/all_rag_techniques_cn")

# 技术术语列表（保留不翻译）
TECH_TERMS = [
    'RAG', 'LLM', 'API', 'FAISS', 'BM25', 'OpenAI', 'LangChain', 'GPT',
    'Embedding', 'Vector', 'Chunk', 'Index', 'Notebook', 'PDF', 'URL',
    'HTTP', 'AI', 'GPT-4', 'GPT-3.5', 'LLM', 'Settings', 'Model',
    'Temperature', 'ServiceContext', 'from_defaults', 'ColPali', 'Milvus',
    'LlamaIndex', 'Chroma', 'Cohere', 'Gemini', 'PyMuPDF', 'BM25',
    'FAISS', 'HuggingFace', 'DSPy', 'GraphRAG', 'CRAG', 'RAPTOR', 'HyDe',
    'HyPE', 'CCH', 'RSE', 'KITE', 'BLEU', 'ROUGE', 'GLM', 'LMUnit',
    'ContextualAI', 'V3', 'V2', 'Top-K', 'Reranker', 'Cross-Encoder',
    'Multi-faceted', 'DIVERSITY_WEIGHT', 'RELEVANCE_WEIGHT', 'SIGMA',
    'JSON', 'CSV', 'RAM', 'GPU', 'CPU', 'IDE', 'VSCode', 'Python',
    'JavaScript', 'TypeScript', 'React', 'Node.js', 'pip', 'conda',
    'GitHub', 'GitLab', 'CI/CD', 'Docker', 'Kubernetes', 'AWS', 'Azure',
    'GCP', 'S3', 'EC2', 'Lambda', 'REST', 'GraphQL', 'SQL', 'NoSQL',
    'MongoDB', 'PostgreSQL', 'MySQL', 'Redis', 'Elasticsearch', 'Kafka',
    'Spark', 'Hadoop', 'TensorFlow', 'PyTorch', 'Scikit-learn', 'Pandas',
    'NumPy', 'Matplotlib', 'Seaborn', 'Plotly', 'Jupyter', 'Colab',
    'Markdown', 'HTML', 'CSS', 'SCSS', 'Sass', 'Webpack', 'Vite',
    'npm', 'yarn', 'pnpm', 'Node.js', 'Deno', 'Bun', 'Rust', 'Go',
    'Java', 'Kotlin', 'Swift', 'Ruby', 'PHP', 'C++', 'C#', '.NET',
    'Spring', 'Django', 'Flask', 'FastAPI', 'Express', 'Next.js',
    'Nuxt.js', 'Svelte', 'Vue', 'Angular', 'Bootstrap', 'Tailwind',
    'Material', 'Ant Design', 'Chakra UI', 'Radix', 'shadcn/ui'
]

def is_english_text(text):
    """检查文本是否包含需要翻译的英文内容"""
    if not text or len(text.strip()) < 3:
        return False

    # 移除技术术语后检查是否还有英文
    temp = text
    for term in TECH_TERMS:
        temp = re.sub(re.escape(term), '', temp, flags=re.IGNORECASE)

    # 检查是否还有连续的英文字母（3 个以上）
    return bool(re.search(r'[A-Za-z]{3,}', temp))

def translate_short_text(text):
    """翻译短文本（简单规则替换）"""
    # 这是一个简单的翻译函数，实际应该调用翻译 API
    # 这里只做示例
    translations = {
        'Overview': '概述',
        'Motivation': '动机',
        'Key Components': '关键组件',
        'Method Details': '方法细节',
        'Advantages': '优势',
        'Conclusion': '结论',
        'Implementation': '实现',
        'Evaluation': '评估',
        'Results': '结果',
        'Settings': '设置',
        'Configuration': '配置',
        'Parameters': '参数',
        'Example': '示例',
        'Examples': '示例',
        'Code': '代码',
        'Output': '输出',
        'Input': '输入',
        'Note': '注意',
        'Notes': '注意',
        'Tip': '提示',
        'Tips': '提示',
        'Warning': '警告',
        'Important': '重要',
        'Required': '必需',
        'Optional': '可选',
    }

    for en, zh in translations.items():
        if text.strip() == en:
            return zh

    return text  # 无法翻译则返回原文

def check_notebook(file_path):
    """检查 Notebook 文件中的英文内容"""
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    issues = []

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            content = ''.join(cell['source'])
            if is_english_text(content):
                issues.append({
                    'cell_index': i,
                    'type': 'markdown',
                    'content': content[:100]
                })
        elif cell['cell_type'] == 'code':
            # 检查代码注释
            content = ''.join(cell['source'])
            lines = content.split('\n')
            for j, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith('#') and not stripped.startswith('#!'):
                    if is_english_text(stripped):
                        issues.append({
                            'cell_index': i,
                            'line': j + 1,
                            'type': 'code_comment',
                            'content': stripped[:100]
                        })

    return issues

def main():
    """主函数"""
    print("=" * 60)
    print("翻译修复脚本")
    print("=" * 60)

    # 获取所有 Notebook 文件
    notebooks = sorted(CN_DIR.glob("*.ipynb"))
    print(f"\n找到 {len(notebooks)} 个 Notebook 文件\n")

    total_issues = 0

    for nb_path in notebooks:
        print(f"检查：{nb_path.name}")
        issues = check_notebook(nb_path)

        if issues:
            print(f"  发现 {len(issues)} 个问题:")
            for issue in issues[:5]:  # 只显示前 5 个
                if issue['type'] == 'markdown':
                    print(f"    Cell {issue['cell_index']} (Markdown): {issue['content']}...")
                else:
                    print(f"    Cell {issue['cell_index']} Line {issue['line']} (注释): {issue['content']}...")
            if len(issues) > 5:
                print(f"    ... 还有 {len(issues) - 5} 个问题")
            total_issues += len(issues)
        else:
            print("  ✓ 翻译完整")
        print()

    print("=" * 60)
    print(f"总共有 {total_issues} 个需要修复的英文内容")
    print("=" * 60)

if __name__ == '__main__':
    main()
