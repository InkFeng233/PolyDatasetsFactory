# 文档切片器
from .loader import DocumentLoader
from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter,SemanticSplitterNodeParser,MarkdownNodeParser
from llama_index.embeddings.dashscope import DashScopeEmbedding
import os
from dotenv import load_dotenv

import re
def Punctuation_Chinese(text: str) -> list[str]:
    # 支持中文标点的句子分割器，保留标点
    parts = re.split(r'([。！？.!?]+)', text)
    sentences = []
    # 文本与标点两两组合，组合结果为一条字符串
    for i in range(0, len(parts) - 1, 2):
        sentence = parts[i] + parts[i + 1]
        if sentence.strip():
            sentences.append(sentence)
    # 末尾元素是一个字符串
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1])
    return sentences


# 文本切分类
class Chunker:
    _embed = None


    def __init__(self, batch_size: int = 1):
        if self._embed is None:
            batch_size = int(batch_size)
            if batch_size in range(1,11):
                load_dotenv()
                api_key = os.getenv("DASHSCOPE_API_KEY")
                self._embed = DashScopeEmbedding(api_key=api_key, model_name="text-embedding-v4",embed_batch_size=batch_size)


    # 语义切分器
    def SemanticSplitter(self, path: str, non_similarity: int = 70):
        embed = self._embed
        source = os.path.basename(path)
        non_similarity = int(non_similarity)
        if non_similarity not in range(1,101):
            return []

        # 加载文档
        loader = DocumentLoader().load(path=path)       # 返回dict
        document = loader["origin"]     # 返回 文件解析出资源 - Document类
        flag = loader["flag"]

        # 切分文档
        if flag:
            parser = SemanticSplitterNodeParser(
                embed_model=embed,
                buffer_size=1,
                breakpoint_percentile_threshold=non_similarity,
                sentence_splitter=Punctuation_Chinese
            )
        else:
            parser = MarkdownNodeParser()
        nodes = parser.get_nodes_from_documents(documents=[document])

        data = []
        for index,node in enumerate(nodes):
            text = node.text
            vector = embed.get_text_embedding(text)
            data.append({
                "text":text,
                "vector":vector,
                "source":source,
                "chunk_index":index
            })
        return data



    # 文本块切分器 - Token
    def BlockSplitter(self, path:str):
        embed = self._embed
        loader = DocumentLoader()
        document = Document(text=loader.load(path=path)["content"])

        parser = TokenTextSplitter(chunk_size=1024,chunk_overlap=20)
        nodes = parser.get_nodes_from_documents(documents=[document])
        source = os.path.basename(path)

        data = []
        for index,node in enumerate(nodes):
            text = node.text
            vector = embed.get_text_embedding(text)
            data.append({
                "text":text,
                "vector":vector,
                "source":source,
                "chunk_index":index
            })
        return data
