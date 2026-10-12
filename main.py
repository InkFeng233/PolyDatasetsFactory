from workflow.poly_datasets_factory import PolyDatasetsFactory
from workflow.module import KnowledgeInput
import asyncio


async def dataset_factory(prompt: str, path: str = None, collections: list[str] = None, top_k: int = 3) -> str:
    """
    运行数据集生成工作流

    Args:
        prompt: 用户问题或指令
        path: 文件路径（可选，支持 .pdf/.png/.jpg/.jpeg/.md/.txt）
        collections: 知识库列表（可选）
        top_k: 知识库检索数量（默认3）

    Returns:
        生成的问答对 JSON 字符串
    """

    # 构建知识库输入
    knowledge = None
    if collections:
        knowledge = KnowledgeInput(
            query=prompt,
            collections=collections,
            top_k=top_k
        )

    # 创建工作流
    workflow = PolyDatasetsFactory(timeout=120, verbose=False)

    # 运行工作流
    result = await workflow.run(
        prompt=prompt,
        path=path,
        knowledge=knowledge
    )

    return result


def main():
    prompt = input("请输入问题: ")
    path_input = input("文件路径（回车跳过）: ").strip()
    path = path_input if path_input else None

    collections = input("知识库（逗号分隔，回车跳过）: ")
    collections = [c.strip() for c in collections.split(",")] if collections else None
    top_k = int(input("检索数量（默认3）: ") or 3)

    result = asyncio.run(dataset_factory(prompt, path, collections, top_k))
    print("\n结果：\n", result)


if __name__ == "__main__":
    main()
