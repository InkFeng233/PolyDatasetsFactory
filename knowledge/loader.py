# 文档加载器 - 仅支持docx / xlsx / md / txt 等文档类资源
from llama_index.core import SimpleDirectoryReader,Document
from llama_index.readers.file import MarkdownReader
from pypdf import PdfReader
import os

class DocumentLoader:
    # 添加单个文件
    """
    --- Document对象解析 ---
    id_: str,                                # 文档唯一标识符
    embedding: Optional[List[float]] = None, # 文档级向量，通常为 None
    metadata: Dict[str, Any],                # 文件元信息字典
    excluded_embed_metadata_keys: List[str], # 不参与 embedding 的元数据键列表
    excluded_llm_metadata_keys: List[str],   # 不传给 LLM 的元数据键列表
    relationships: Dict[str, Any],           # 文档间关系，通常为空字典
    metadata_template: str,                  # 元数据格式化模板
    metadata_separator: str,                 # 元数据分隔符
    text_resource: MediaResource,            # 正文资源对象
    image_resource: Optional[MediaResource] = None,  # 图片资源
    audio_resource: Optional[MediaResource] = None,  # 音频资源
    video_resource: Optional[MediaResource] = None,  # 视频资源
    text_template: str                       # 文本拼接模板

    --- MediaResource 对象格式 ---
    embeddings: Optional[List[float]] = None, # 资源级向量
    data: Optional[bytes] = None,             # 原始二进制数据
    text: Optional[str] = None,               # 文本内容
    path: Optional[str] = None,               # 资源路径
    url: Optional[str] = None,                # 资源 URL
    mimetype: Optional[str] = None            # MIME 类型
    """

    # 需要新增功能 - 适配多类型文件 .md / .pdf / .txt + .docx / .xlsx + .csv
    def load(self,path:str):
        # 取文件后缀
        file_ext = os.path.splitext(path)[-1].lower()
        # 标识符 ： True 为 文档 ， False 为 Markdown文档
        flag = True

        if file_ext == ".md":
            flag = False
            file_extractor = {".md":MarkdownReader()}
            document = SimpleDirectoryReader(input_files=[path],file_extractor=file_extractor).load_data()[0]

        elif file_ext == ".pdf":
            texts = []
            reader = PdfReader(path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    texts.append(text)
            document = Document(text="\n".join(texts))

        elif file_ext in [".txt",".docx"]:
            # 返回Document列表
            document = SimpleDirectoryReader(input_files=[path]).load_data()[0]
            
        else:
            raise ValueError(f"不支持的文件类型: {file_ext}")
        
        return {
            # 提取信息
            "id":document.id_,
            "metadata":document.metadata,
            "content":document.text_resource.text,
            "flag":flag,
            # 本源信息
            "origin":document
        }