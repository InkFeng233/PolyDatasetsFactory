# 知识库业务
## input = 用户提问
## 处理 = input -> 向量 -> 向量搜索向量数据库
## output = 检索结果的片段
from .chunker import Chunker
from .config import MilvusDB
from .response import success, error, Code, Message
from dotenv import load_dotenv
import os
from datetime import datetime

class KnowledgeBase:
    # 初始化句柄
    def __init__(self):
        self._db = MilvusDB()
        self._client = self._db.client
        self._client.use_database("poly_dataset_factory")

    # 创建知识库
    def create(self, collection_name: str):
        try:
            uuid = self._db.CreateCollection(collection_name)
            return success(
                data={
                    "name": collection_name, 
                    "uuid": uuid
                    },
                message=Message.KB_CREATE_SUCCESS
            )
        except RuntimeError as e:
            if "已存在" in str(e):
                return error(Message.KB_CREATE_EXISTS, Code.CONFLICT)
            return error(str(e), Code.SERVER_ERROR)

    # 查询知识库
    def search(self, collection_name: str):
        result = self._client.query(
            collection_name="reflection",
            output_fields=["uuid"],
            filter=f'display_name == "{collection_name}"'
        )
        if not result:
            raise ValueError(f"知识库 [{collection_name}] 不存在")
        return result[0]["uuid"]
    
    # 删除知识库
    def delete(self, collection_name: str):
        try:
            uuid = self.search(collection_name=collection_name)
            # 删除 collection
            self._db.ClearCollection(collection_name=uuid)
            # 删除 reflection的映射记录
            self._client.delete(
                collection_name="reflection",
                filter=f'display_name == "{collection_name}"'
            )
            return success(message=Message.KB_DELETE_SUCCESS)
        except ValueError:
            return error(Message.KB_NOT_FOUND, Code.NOT_FOUND)

    # 展示所有知识库
    def list_databases(self):
        knowledge_bases = self._client.query(
            collection_name="reflection",
            output_fields=["display_name", "uuid", "created_time"]
        )
        return success(
            data={"databases": knowledge_bases, "total": len(knowledge_bases)}
        )

    # 展示某个知识库所有知识文档
    def list_sources(self, collection_name: str):
        try:
            uuid = self.search(collection_name=collection_name)

            results = self._client.query(
                collection_name=uuid,
                output_fields=["source", "chunk_index", "created_time"]
            )

            # 分组统计，每组即为一个知识文档
            sources = {}
            for item in results:
                source = item["source"]
                if source not in sources:
                    sources[source] = {
                        "source": source,
                        "created_time": item["created_time"],
                        "count": 0
                    }
                sources[source]["count"] += 1

            data = list(sources.values())
            return success(data={"sources": data, "total": len(data)})
        except ValueError:
            return error(Message.KB_NOT_FOUND, Code.NOT_FOUND)
        
    # 向知识库增加内容
    def add(self, collection_name: str, path: str):
        try:
            # 获取实际collection & 知识文档名
            uuid = self.search(collection_name=collection_name)
            source = os.path.basename(path)

            # 检验文档是否存在
            existing = self._client.query(
                collection_name=uuid,
                filter=f'source == "{source}"',
                output_fields=["source"]
            )
            if existing:
                return error(Message.DOC_ADD_EXISTS, Code.CONFLICT)

            # 文档分片
            chunker = Chunker()
            chunks = chunker.SemanticSplitter(path=path)

            # 整合数据
            if not chunks:
                return error(Message.DOC_ADD_FAILED, Code.BAD_REQUEST)

            data = []
            created_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for chunk in chunks:
                data.append({
                    "text": chunk["text"],
                    "vector": chunk["vector"],
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                    "created_time": created_time
                })

            # 插入数据
            self._client.insert(collection_name=uuid, data=data)

            return success(
                data={"source": source, "count": len(chunks)},
                message=Message.DOC_ADD_SUCCESS
            )
        except ValueError:
            return error(Message.KB_NOT_FOUND, Code.NOT_FOUND)

    # 删除知识库的某个chunk
    def delete_chunk(self, collection_name: str, source: str, chunk_index: int):
        try:
            uuid = self.search(collection_name=collection_name)

            self._client.delete(
                collection_name=uuid,
                filter=f'source == "{source}" and chunk_index == {chunk_index}'
            )

            return success(
                data={"source": source, "chunk_index": chunk_index},
                message=Message.CHUNK_DELETE_SUCCESS
            )
        except ValueError:
            return error(Message.KB_NOT_FOUND, Code.NOT_FOUND)
        
    # 列出某个知识库的所有chunk
    def list_chunks(self, collection_name: str, source: str):
        try:
            uuid = self.search(collection_name=collection_name)

            results = self._client.query(
                collection_name=uuid,
                filter=f'source == "{source}"',
                output_fields=["text", "chunk_index", "created_time"]
            )

            return success(data={"chunks": results, "total": len(results)})
        except ValueError:
            return error(Message.KB_NOT_FOUND, Code.NOT_FOUND)

    # 指定知识库并查询
    def query(self, collections: list[str], text: str, top_k: int = 5):
        try:
            # 文本向量化
            embed = Chunker()._embed
            query_vector = embed.get_text_embedding(text)

            # 计算向量方式
            results = []        # 相关片段
            search_params={"metric_type": "L2"}

            # 检索结果
            for col in collections:
                uuid = self.search(collection_name=col)
                search_result = self._client.search(
                    collection_name=uuid,
                    output_fields=["text","source","chunk_index"],
                    search_params=search_params,        # 搜索参数
                    data=[query_vector],        # 查询向量
                    limit=top_k         # 返回片段数
                )

                # 整理结果，记录片段
                for result in search_result[0]:
                    results.append({
                        "text":result["entity"]["text"],
                        "source":result["entity"]["source"],
                        "chunk_index":result["entity"]["chunk_index"],
                        "score":result["distance"],
                        "origin":col
                    })

            # 检索结果得分排序（L2距离越小越相似，从小到大排序）
            results.sort(key=lambda x:x["score"],reverse=False)

            final_results = results[:top_k]
            return success(data={
                "data":final_results,
                "count":len(final_results)
            })
        
        except ValueError as e:
            return error(str(e),Code.NOT_FOUND)
        except Exception as e:
            return error(str(e),Code.SERVER_ERROR)
