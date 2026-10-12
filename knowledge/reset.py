# 数据库重置脚本
# 用于删除 poly_dataset_factory 数据库及其下所有 collection

import os
from dotenv import load_dotenv
from pymilvus import MilvusClient


def reset_database():
    """重置数据库：删除所有 collection 和数据库"""
    load_dotenv()
    uri = os.getenv("MILVUS_URI")

    print("正在连接 Milvus...")
    client = MilvusClient(uri=uri, token="root:Milvus")

    db_name = "poly_dataset_factory"

    # 检查数据库是否存在
    try:
        client.use_database(db_name)
    except Exception:
        print(f"数据库 [{db_name}] 不存在，无需重置")
        return

    # 获取所有 collection
    collections = client.list_collections()
    print(f"\n找到 {len(collections)} 个 collection:")
    for col in collections:
        print(f"  - {col}")

    # 删除所有 collection
    print("\n正在删除 collection...")
    for col in collections:
        client.drop_collection(col)
        print(f"  已删除: {col}")

    # 删除数据库
    print(f"\n正在删除数据库 [{db_name}]...")
    client.drop_database(db_name)
    print("数据库已删除")

    print("\n重置完成！")


if __name__ == "__main__":
    reset_database()
