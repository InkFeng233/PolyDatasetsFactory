from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
)
from llama_index.core.llms import ChatMessage,MessageRole,ImageBlock, DocumentBlock, TextBlock,AudioBlock
from typing import Literal
from pydantic import BaseModel
import os
from dotenv import load_dotenv

from llama_index.llms.deepseek import DeepSeek
from llama_index.llms.dashscope import DashScope
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.prompts import RichPromptTemplate
from knowledge.knowledge import KnowledgeBase
from knowledge.response import Code


# 附件数据
## 附件数据格式
"""
ImageBlock参数格式：
    block_type: Literal["image"] = "image"
    image: bytes | IOBase | None = None
    path: FilePath | None = None
    url: AnyUrl | str | None = None
    image_mimetype: str | None = None
    detail: str | None = None

DocumentBlock参数格式：
    block_type: Literal["document"] = "document"
    data: bytes | IOBase | None = None
    path: Optional[Union[FilePath | str]] = None
    url: Optional[str] = None
    title: Optional[str] = None
    document_mimetype: Optional[str] = None

AudioBlock参数格式：
    block_type: Literal["audio"] = "audio"
    audio: bytes | IOBase | None = None
    path: FilePath | None = None
    url: AnyUrl | str | None = None
    format: str | None = None

TextBlock参数格式：
    block_type: Literal["text"] = "text"
    text: str
"""
class Attachment(BaseModel):
    url: str | None = None
    filename: str | None = None
    file_type: Literal["image", "document", "audio", "text"]
    mimetype: str | None = None
    image: bytes | None = None     # ImageBlock 专用
    data: bytes | None = None      # DocumentBlock 专用
    audio: bytes | None = None     # AudioBlock 专用
    path: str | None = None
    detail: str | None = None     # ImageBlock 专用

    def to_block(self):
        if self.file_type == "image":
            return ImageBlock(
                url=self.url,
                image=self.image,
                path=self.path,
                image_mimetype=self.mimetype,
                detail=self.detail
            )
        elif self.file_type == "document":
            return DocumentBlock(
                url=self.url,
                data=self.data,
                path=self.path,
                title=self.filename,
                document_mimetype=self.mimetype
            )
        elif self.file_type == "audio":
            return AudioBlock(
                url=self.url,
                audio=self.audio,
                path=self.path,
                format=self.mimetype
            )
        elif self.file_type == "text":
            with open(self.path, "r", encoding="utf-8") as f:
                content = f.read()
            return TextBlock(text=content)

# 模型加载
class LlmProvier:
        ## 供应商列表
        PROVIDERS = {
            "deepseek": {
                "api_key_env": "DEEPSEEK_API_KEY",
                "class": DeepSeek,
            },
            "dashscope":{
                "api_key_env": "DASHSCOPE_API_KEY",
                "class":DashScope,
            },
            "glm": {
                "api_key_env": "GLM_API_KEY",
                "class": OpenAILike,
                "api_base": "https://open.bigmodel.cn/api/paas/v4",
                "is_chat_model": True,
            }
        }
        def __init__(self, provider: str, model: str):
            load_dotenv()
            self.provider = provider
            config = self.PROVIDERS[provider]
            api_key = os.getenv(config["api_key_env"])
            # DashScope 使用 model_name，GLM 需要额外参数
            if provider == "dashscope":
                self.instance = config["class"](api_key=api_key, model_name=model)
            elif provider == "glm":
                self.instance = config["class"](
                    model=model,
                    api_key=api_key,
                    api_base=config["api_base"],
                    is_chat_model=config.get("is_chat_model", True),
                )
            else:
                self.instance = config["class"](api_key=api_key, model=model)

        def chat(self, prompt: ChatMessage):
            return self.instance.chat(prompt)



# 事件定义
## AI对话组件 - 事件
class AiChatInput(Event):
    # 模型配置
    model_provider: str = "deepseek"     # 模型提供商
    model_name: str         # 模型名称
    temperature: float = 0.7             # 温度参数
    max_tokens: int | None = None       # 最大 token 数
    attachments: list[Attachment] = []   # 附件列表

    # 提示词
    query: str = ""                          # 用户输入
    system_prompt: str = ""              # 系统提示词
    prompt_template:str | None = None      # 提示词模板
    template_filters: dict = None      # 占位符
    context: str |None = None        # 检索知识片段

class AiChatOutput(Event):
    response: str                        # LLM 回复

class KnowledgeInput(Event):
    query: str
    collections: list[str]
    top_k: int = 3

class KnowledgeOutput(Event):
    response: str



# 组件定义
## AI对话组件
class AiChatComponent:
    def run(self, ev: AiChatInput) -> AiChatOutput:
        blocks = [] 
        # 组装信息
        if ev.prompt_template:
            template = RichPromptTemplate(ev.prompt_template)
            variables = ev.template_filters or {}
            variables["query"] = ev.query
            variables["system_prompt"] = ev.system_prompt
            variables["context"] = ev.context
            messages = template.format_messages(**variables)
            chat_msg = messages[0] if messages else ChatMessage(role=MessageRole.USER.value, blocks=[])
        else:
            blocks.append(TextBlock(text=ev.system_prompt))
            if ev.context:
                blocks.append(TextBlock(text=ev.context))
            if ev.query:
                blocks.append(TextBlock(text=ev.query))
            chat_msg = ChatMessage(role=MessageRole.USER.value, blocks=blocks)

        for att in ev.attachments:
            chat_msg.blocks.append(att.to_block())

        # 设置大模型
        llm = LlmProvier(provider=ev.model_provider,model=ev.model_name)
        response =llm.chat([chat_msg])

        # 结果入库（处理 None 的情况）
        content = response.message.content or ""
        return AiChatOutput(response=content)

## 知识库检索组件
class KnowledgeDatabaseComponent:
    def run(self, ev: KnowledgeInput) -> KnowledgeOutput:
        query = ev.query
        kb = KnowledgeBase()
        result = kb.query(ev.collections, query, ev.top_k)

        # 检索结果审查
        if result["code"] == Code.SUCCESS.value and result["data"]["data"]:
            # 有结果 - 整理数据
            chunks = result["data"]["data"]
            lines = ["# 参考资料\n"]
            for index, chunk in enumerate(chunks, 1):
                formatted = f"## 片段 {index}（来源：{chunk['source']}）\n{chunk['text']}"
                lines.append(formatted)
            return KnowledgeOutput(response="\n".join(lines))
        else:
            # 出错或者无结果 - 返回空串
            return KnowledgeOutput(response="")