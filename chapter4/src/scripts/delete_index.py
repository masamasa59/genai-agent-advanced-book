from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient


def delete_es_index(es: Elasticsearch, index_name: str) -> None:
    # インデックスの削除
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print(f"Index '{index_name}' has been deleted.")
    else:
        print(f"Index '{index_name}' does not exist.")


def delete_qdrant_index(qdrant_client: QdrantClient, collection_name: str) -> None:

    if qdrant_client.collection_exists(collection_name=collection_name):
        # qdrantでインデックスを削除
        qdrant_client.delete_collection("documents")
        print(f"Collection '{collection_name}' has been deleted.")
    else:
        print(f"Collection '{collection_name}' does not exist.")


if __name__ == "__main__":
    es = Elasticsearch("http://localhost:9200")
    qdrant_client = QdrantClient("http://localhost:6333")

    index_name = "documents"

    delete_es_index(es=es, index_name=index_name)

    delete_qdrant_index(qdrant_client=qdrant_client, collection_name=index_name)
