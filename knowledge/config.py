# 配置文件
from pymilvus import MilvusClient, DataType
import os
from dotenv import load_dotenv
import uuid
from datetime import datetime

class MilvusDB:
    _client = None

    # 赋予句柄
    @property
    def client(self):
        return MilvusDB._client


    # 初始化 - 获取句柄
    @classmethod
    def Setup(cls):
        load_dotenv()

        # 提取环境变量
        uri = os.getenv("MILVUS_URI")
        role_name = os.getenv("MILVUS_ROLE")
        user_name= os.getenv("MILVUS_USER")
        user_password = os.getenv("MILVUS_PASSWORD")

        # 初始化角色
        root_client = MilvusClient(
            uri=uri,
            token="root:Milvus"
        )
        # 建立 访问角色（忽略已存在的错误）
        try:
            root_client.create_user(user_name=user_name,password=user_password)
        except Exception:
            pass
        try:
            root_client.create_role(role_name=role_name)
        except Exception:
            pass
        try:
            root_client.grant_privilege_v2(
                role_name=role_name,
                privilege="*",
                collection_name="*"
            )
        except Exception:
            pass
        try:
            root_client.grant_role(user_name=user_name,role_name=role_name)
        except Exception:
            pass
        # 返回 访问角色
        MilvusDB._client = MilvusClient(
            uri=uri,
            token=f"{user_name}:{user_password}"
        )

    # 初始化 - 建库
    @classmethod
    def Initeralizer(cls):
        client = cls._client
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="L2")

        # 建库（已存在则检查是否为本系统的库）
        try:
            client.create_database(db_name="poly_dataset_factory")
        except Exception as e:
            if "already exist" in str(e):
                # 数据库已存在，检查是否有 reflection 表（本系统的标记）
                client.use_database(db_name="poly_dataset_factory")
                collections = client.list_collections()
                if "reflection" not in collections:
                    raise RuntimeError("数据库 [poly_dataset_factory] 已存在，但不是本系统创建的，请更换数据库名")
                # 有 reflection 表，说明是本系统的，继续使用
            else:
                raise  # 其他错误继续抛出
        else:
            client.use_database(db_name="poly_dataset_factory")

        # 系统映射表（已存在则跳过）
        collections = client.list_collections()
        if "reflection" not in collections:
            schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="display_name", datatype=DataType.VARCHAR, max_length=256)
            schema.add_field(field_name="uuid", datatype=DataType.VARCHAR, max_length=256)
            schema.add_field(field_name="created_time", datatype=DataType.VARCHAR, max_length=64)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT16_VECTOR, dim=2)  # 占位字段
            client.create_collection(collection_name="reflection", schema=schema)
            client.create_index(collection_name="reflection", index_params=index_params)
            client.load_collection(collection_name="reflection")

    # 管理 - 建表
    def CreateCollection(self,collection_name:str):
        client = self.client
        name = collection_name
        # 表名的映射名
        ref = f"kd_{uuid.uuid5(uuid.NAMESPACE_URL, name).hex}"

        # 建立新表
        schema = MilvusClient.create_schema(
            auto_id=True,
            enable_dynamic_field=True
        )
        schema.add_field(field_name="id",datatype=DataType.INT64,is_primary=True)
        schema.add_field(field_name="text",datatype=DataType.VARCHAR,max_length=4096)
        schema.add_field(field_name="vector",datatype=DataType.FLOAT16_VECTOR,dim=1024)
        schema.add_field(field_name="source",datatype=DataType.VARCHAR,max_length=512)
        schema.add_field(field_name="chunk_index",datatype=DataType.INT64)
        schema.add_field(field_name="created_time", datatype=DataType.VARCHAR, max_length=64)

        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="L2")

        client.use_database(db_name="poly_dataset_factory")
        existing = client.query(
            collection_name="reflection",
            filter=f'display_name == "{name}"',
            output_fields=["display_name"]
        )
        if existing:
            raise RuntimeError(f"知识库[{name}]已存在，请更换知识库名称！")

        data = {
            "display_name":name,
            "uuid":str(ref),
            "vector":[0.0,0.0],
            "created_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        client.insert(collection_name="reflection",data=[data])
        client.create_collection(collection_name=ref, schema=schema)
        client.create_index(collection_name=ref, index_params=index_params)
        client.load_collection(collection_name=ref)  # 加载到内存

        return ref


    # 管理 - 删除指定表
    @classmethod
    def ClearCollection(cls, collection_name: str):
        if cls._client is None:
            cls.Setup()
        client = cls._client
        client.use_database(db_name="poly_dataset_factory")
        client.drop_collection(collection_name=collection_name)

    # 管理 - 监控表状态
    @classmethod
    def CollectionGuard(cls, collection_name: str):
        client = cls._client
        client.use_database(db_name="poly_dataset_factory")
        result = client.get_collection_stats(collection_name=collection_name)
        return result

    # 管理 - 删库
    @classmethod
    def ClearDatabase(cls, db_name: str = "poly_dataset_factory"):
        if cls._client is None:
            cls.Setup()
        client = cls._client
        client.drop_database(db_name=db_name)


    # 初始化
    def __init__(self):
        if MilvusDB._client is None:
            MilvusDB.Setup()
            MilvusDB.Initeralizer()