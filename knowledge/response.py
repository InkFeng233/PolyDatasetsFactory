# 接口响应格式定义
from enum import Enum


class Code(Enum):
    """响应状态码"""
    SUCCESS = 200           # 成功
    BAD_REQUEST = 400       # 参数错误
    NOT_FOUND = 404         # 资源不存在
    CONFLICT = 409          # 资源冲突（如已存在）
    SERVER_ERROR = 500      # 服务器错误


class Message:
    """响应消息常量"""
    # 通用
    SUCCESS = "操作成功"

    # 知识库操作
    KB_CREATE_SUCCESS = "知识库创建成功"
    KB_CREATE_EXISTS = "知识库已存在"
    KB_DELETE_SUCCESS = "知识库已删除"
    KB_NOT_FOUND = "知识库不存在"

    # 文档操作
    DOC_ADD_SUCCESS = "文档插入成功"
    DOC_ADD_EXISTS = "文档已存在，请先删除后再上传"
    DOC_ADD_FAILED = "文件处理失败，未获取有效内容"
    DOC_NOT_FOUND = "文档不存在"

    # 片段操作
    CHUNK_DELETE_SUCCESS = "片段已删除"
    CHUNK_NOT_FOUND = "片段不存在"

    # 系统错误
    INVALID_PARAMS = "传入参数错误"
    UNKNOWN_ERROR = "未知错误"


def success(data=None, message=Message.SUCCESS, code=Code.SUCCESS):
    """成功响应"""
    return {
        "code": code.value if isinstance(code, Code) else code,
        "message": message,
        "data": data
    }


def error(message, code=Code.BAD_REQUEST, data=None):
    """错误响应"""
    return {
        "code": code.value if isinstance(code, Code) else code,
        "message": message,
        "data": data
    }


# 数据格式规范
class DataFormat:
    """数据字段命名规范"""
    # 列表类返回
    LIST_FORMAT = {
        "items": [],    # 数据列表
        "total": 0      # 总数
    }

    # 单个对象返回
    OBJECT_FORMAT = {
        "item": None    # 单个对象
    }

    # 操作结果返回
    ACTION_FORMAT = {
        "id": None      # 操作的对象标识
    }
