from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.document_loaders import PDFPlumberLoader, CSVLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_qdrant import Qdrant

from langgraph.checkpoint.memory import MemorySaver
from unstructured.partition.pdf import partition_pdf
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from diskcache import Cache
import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from functools import cache as fc
from PIL import Image
import io

cache = Cache("tmp")
VECTOR_CACHE_ROOT = 'cache'

if not os.path.exists(VECTOR_CACHE_ROOT):
    os.mkdir(VECTOR_CACHE_ROOT)

# be careful, this is related to embedder
VECTOR_SIZE = 768

def get_image_summarizer():
    SUMMARISE_MODEL = "gemma3:4b"
    summarizer = OllamaLLM(model=SUMMARISE_MODEL)
    return summarizer


@fc
def get_embedder():
    EMBED_MODEL = 'nomic-embed-text'
    embedder = OllamaEmbeddings(model=EMBED_MODEL)
    return embedder

@fc
def get_text_splitter():
    text_splitter = SemanticChunker(embeddings=get_embedder())
    return text_splitter

def md5(filename):
    import hashlib
    import codecs
    return hashlib.md5(codecs.encode(filename)).hexdigest()

def gen_text_and_table_summaries(summarizer, text_splitter, path):
    """
    从 PDF 中提取文本和表格, 使用大模型总结
    """
    # Extract elements from PDF
    def extract_pdf_elements(path):
        """
        Extract images, tables, and chunk text from a PDF file.
        path: File path, which is used to dump images (.jpg)
        fname: File name
        """
        return partition_pdf(
            filename=path,
            extract_images_in_pdf=False,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            image_output_dir_path=path,
            )


    # Categorize elements by type
    def categorize_elements(raw_pdf_elements):
        """
        Categorize extracted elements from a PDF into tables and texts.
        raw_pdf_elements: List of unstructured.documents.elements
        """
        tables = []
        texts = []
        for element in raw_pdf_elements:
            if "unstructured.documents.elements.Table" in str(type(element)):
                tables.append(str(element))
            elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
                texts.append(str(element))
        return texts, tables
    
    def generate_text_summaries(texts, tables, summarize_texts=False):
        """
        Summarize text elements
        texts: List of str
        tables: List of str
        summarize_texts: Bool to summarize texts
        """

        # Prompt
        prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval. \
        These summaries will be embedded and used to retrieve the raw text or table elements. \
        Give a concise summary of the table or text that is well optimized for retrieval. Table or text: {element} """
        prompt = ChatPromptTemplate.from_template(prompt_text)

        # Text summary chain
        summarize_chain = {"element": lambda x: x} | prompt | summarizer | StrOutputParser()

        # Initialize empty summaries
        text_summaries = []
        table_summaries = []

        # Apply to text if texts are provided and summarization is requested
        if texts and summarize_texts:
            text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
        elif texts:
            text_summaries = texts

        # Apply to tables if tables are provided
        if tables:
            table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

        return text_summaries, table_summaries
    raw = extract_pdf_elements(path)
    texts, tables = categorize_elements(raw)
    texts, tables = text_splitter.split_text(" ".join(texts)), tables
    return (texts, tables), generate_text_summaries(texts, tables, True)


def init_vector_store(collection_name):
    """
    把所有 collection_name 目录下的文档一起构建一个向量数据库
    """
    @cache.memoize()
    def to_doc(path: str):
        """
        把给定文件转换成文档格式，主要是一个加载和分段的过程.
        适用于简单 RAG
        """
        from typing import List
        from pydantic import TypeAdapter
        adapter = TypeAdapter(List[Document])
        ret: 'list[Document]'
        cache_file = os.path.join(VECTOR_CACHE_ROOT, md5(path))
        # 分段的过程比较慢（取决于 embedding 实现），因此我们用文件进行缓存
        if os.path.exists(cache_file):
            print(f'cache for {path} exists, will use cache')
            with open(cache_file, 'rb') as c:
                content = c.read()
                return adapter.validate_json(content)
        loader = None
        if path.endswith('pdf'):
            loader = PDFPlumberLoader(path)
        if path.endswith('csv'):
            loader = CSVLoader(path, encoding='utf-8')
        print(f'loading doc {path}')
        docs = loader.load()
        print(f'spliting {path}')
        ret = get_text_splitter().split_documents(docs)

        with open(cache_file, 'wb') as f:
            f.write(adapter.dump_json(ret))
        return ret
    docfiles = []
    for file in os.listdir(collection_name):
        docfiles.append(os.path.join(collection_name, file))
    
    docs = map(to_doc, docfiles)
    documents = [d for ds in docs if ds is not None for d in ds]
    print('building vector store')
    
    client = QdrantClient()
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    vector = Qdrant(client=client, embeddings=get_embedder(), collection_name=collection_name)
    vector.add_documents(documents)
    return vector


def get_vector_store(collection_name):
    client = QdrantClient()
    vector = Qdrant(client=client, embeddings=get_embedder(), collection_name=collection_name)
    return vector


def get_img_summaries_llm(image_path, additional_args: str) -> tuple[str, str]:
    import base64
    """
    summarise image using multi-modal llm
    """
    def encode_image(path):
        with Image.open(path) as img:
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=10)
            image_data = buffer.getvalue()
            encoded_image = base64.b64encode(image_data).decode('utf-8')
            return encoded_image
    
    def image_summarize(img_base64, prompt):
        """Make image summary"""
        msg = get_image_summarizer().invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                        },
                    ]
                )
            ]
        )
        return msg
        # Store base64 encoded images

    # Prompt
    prompt = f"you will summarise the content of given image base64. {additional_args}"
    # Apply to images
    base64_image = encode_image(image_path)
    return base64_image, image_summarize(base64_image, prompt)

@fc
def get_pipeline(task, model):
    from transformers import pipeline

    return pipeline(task, model)

def img_to_txt(path):
    """
    summarise image using img-to-txt models
    """
    from transformers import pipeline
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    ret = captioner(path, max_new_tokens=100)
    return ret[-1]['generated_text']


if __name__ == '__main__':
    print(img_to_txt('assets/rappers/eazy.jpg'))