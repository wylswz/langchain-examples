from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize
import math


def compute_token_probs(chunks: List[Document]) -> dict[str, float]:
    """计算文档块中所有token的概率分布"""
    # 合并所有文档块的文本
    all_text = " ".join([chunk.page_content for chunk in chunks])
    
    # 对文本进行分词并转换为小写
    tokens = word_tokenize(all_text.lower())
    
    # 过滤掉非字母数字的token
    tokens = [token for token in tokens if token.isalnum()]
    
    # 计算每个token的频率
    token_freq = Counter(tokens)
    
    # 计算总token数
    total_tokens = len(tokens)
    
    # 计算每个token的概率
    token_probs = {token: count/total_tokens for token, count in token_freq.items()}
    
    return token_probs


def compute_token_tfidf(chunks: List[Document]) -> dict[str, float]:
    """计算文档块中token的信息熵"""
    # 合并所有文档块的文本
    all_text = " ".join([chunk.page_content for chunk in chunks])
    
    # 对文本进行分词
    tokens = word_tokenize(all_text.lower())
    
    # 计算每个token的频率
    token_freq = Counter(tokens)
    total_tokens = len(tokens)
    
    # 计算每个token的概率
    token_probs = {token: count/total_tokens for token, count in token_freq.items()}
    # 计算每个文档的token频率
    doc_token_freqs = []
    for chunk in chunks:
        # 对文本进行分词并移除特殊字符
        tokens = [token.lower() for token in word_tokenize(chunk.page_content) 
                 if token.isalnum()]
        # 计算当前文档的token频率
        token_freq = Counter(tokens)
        doc_token_freqs.append(token_freq)
    
    # 计算所有不同的tokens
    all_tokens = set()
    for freq in doc_token_freqs:
        all_tokens.update(freq.keys())
    
    # 计算每个token的TF-IDF
    token_tfidf = {}
    num_docs = len(chunks)
    
    for token in all_tokens:
        # 计算文档频率(DF)
        doc_freq = sum(1 for freq in doc_token_freqs if token in freq)
        # 计算逆文档频率(IDF)
        idf = math.log(num_docs / (1 + doc_freq))
        
        # 计算所有文档的平均TF
        tf_sum = sum(freq[token] / sum(freq.values()) 
                    for freq in doc_token_freqs if token in freq)
        avg_tf = tf_sum / num_docs
        
        # 计算最终的TF-IDF值
        token_tfidf[token] = avg_tf * idf
    return token_tfidf

def filter_by_entropy(chunks: List[Document], threshold: float) -> List[tuple[Document, float]]:
    """根据信息熵过滤文档块"""
    # 计算token概率分布
    token_probs = compute_token_probs(chunks)
    
    filtered_chunks = []
    for chunk in chunks:
        # 对文档块进行分词
        tokens = [token.lower() for token in word_tokenize(chunk.page_content) if token.isalnum()]
        
        if not tokens:
            continue
            
        # 计算文档块的平均信息熵
        chunk_entropy = 0
        for token in tokens:
            if token in token_probs:
                p = token_probs[token]
                chunk_entropy -= p * math.log2(p)
        
        # 归一化信息熵
        chunk_entropy /= len(tokens)
        
        # 如果信息熵超过阈值，保留该文档块
        if chunk_entropy >= threshold:
            filtered_chunks.append((chunk, chunk_entropy))
            
    return filtered_chunks

def filter_by_tf_idf(chunks: List[Document], threshold: float) -> List[tuple[Document, float]]:
    """过滤掉TF-IDF值低于阈值的文档块"""
    token_tfidf = compute_token_tfidf(chunks)
    filtered_chunks = []
    for chunk in chunks:
        tokens = [token.lower() for token in word_tokenize(chunk.page_content) if token.isalnum()]
        # 计算文档块的总TF-IDF值并根据文档长度进行归一化
        chunk_tfidf = sum(token_tfidf[token] for token in tokens if token in token_tfidf) / len(tokens) if tokens else 0
        if chunk_tfidf >= threshold:
            filtered_chunks.append((chunk, chunk_tfidf))
    return filtered_chunks

def read_and_chunk_pdfs() -> List[Document]:
    """读取PDF文件并分块"""
    # 设置PDF文件夹路径
    pdf_dir = Path("./library")
    
    # 存储所有文档块
    all_chunks = []
    
    # 文本分割器配置
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    # 遍历文件夹中的所有PDF文件
    for pdf_path in pdf_dir.glob("*.pdf"):
        try:
            # 加载PDF文件
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            
            # 对每页进行分块
            chunks = text_splitter.split_documents(pages)
            all_chunks.extend(chunks)
            
        except Exception as e:
            print(f"处理PDF文件 {pdf_path} 时出错: {str(e)}")
            continue
    
    return all_chunks




if __name__ == "__main__":
    docs = read_and_chunk_pdfs()
    threshold = 0  # Example threshold value
    filtered_docs = filter_by_tf_idf(docs, threshold)
    filtered_by_entropy = filter_by_entropy(docs, 0)
    sorted_docs = sorted(filtered_docs, key=lambda x: x[1], reverse=True)
    sorted_entropy = sorted(filtered_by_entropy, key=lambda x: x[1], reverse=True)
    # 打印TF-IDF过滤结果
    PREVIEW_LENGTH = 400
    TOPN = 10
    print("\n=== TF-IDF过滤结果 ===")
    print("\n--- 最高分数文档 ---")
    for i, (doc, score) in enumerate(sorted_docs[:TOPN], 1):
        print(f"\n文档 {i}")
        print(f"TF-IDF分数: {score:.4f}")
        print("内容预览:")
        # 截取前200个字符作为预览
        ELLIPSIS = "..."
        preview = doc.page_content[:PREVIEW_LENGTH] + ELLIPSIS if len(doc.page_content) > PREVIEW_LENGTH else doc.page_content
        print(preview)
        print("-" * 80)
    
    print("\n--- 最低分数文档 ---")
    for i, (doc, score) in enumerate(sorted_docs[-TOPN:], 1):
        print(f"\n文档 {i}")
        print(f"TF-IDF分数: {score:.4f}")
        print("内容预览:")
        preview = doc.page_content[:PREVIEW_LENGTH] + "..." if len(doc.page_content) > PREVIEW_LENGTH else doc.page_content
        print(preview)
        print("-" * 80)

    # 打印熵过滤结果
    print("\n=== 熵过滤结果 ===")
    print("\n--- 最高熵值文档 ---")
    for i, (doc, entropy) in enumerate(sorted_entropy[:TOPN], 1):
        print(f"\n文档 {i}")
        print(f"熵值: {entropy:.4f}")
        print("内容预览:")
        preview = doc.page_content[:PREVIEW_LENGTH] + "..." if len(doc.page_content) > PREVIEW_LENGTH else doc.page_content
        print(preview)
        print("-" * 80)
        
    print("\n--- 最低熵值文档 ---")
    for i, (doc, entropy) in enumerate(sorted_entropy[-TOPN:], 1):
        print(f"\n文档 {i}")
        print(f"熵值: {entropy:.4f}")
        print("内容预览:")
        preview = doc.page_content[:PREVIEW_LENGTH] + "..." if len(doc.page_content) > PREVIEW_LENGTH else doc.page_content
        print(preview)
        print("-" * 80)
