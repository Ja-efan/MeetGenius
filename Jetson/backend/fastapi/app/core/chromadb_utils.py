import chromadb
from chromadb import Settings
import os
import platform
from dotenv import load_dotenv
from typing import Any, Dict, List

# 환경 변수 로드
load_dotenv()


class ChromaCollection:
    """ChromaDB Collection 관리 클래스"""

    def __init__(self, collection_name: str):
        """ChromaCollection 생성자

        Args:
            collection_name (str): 사용할 ChromaDB 컬렉션 이름
        """
        self.collection_name = collection_name
        self.client = self._get_chroma_client()

        self.collection = self.client.get_or_create_collection(name=collection_name)


    def _get_chroma_client(self):
        """운영 체제에 따라 적절한 ChromaDB 클라이언트를 선택"""

        system_name = platform.system()
        
        if system_name in ["Windows", "Darwin"]:  # Windows & MacOS (Darwin)
            print(f"[INFO] Running on {system_name} - Using Local ChromaDB Client")
            return chromadb.PersistentClient(path="./chroma_db")  # 로컬에서 ChromaDB 사용
        else:  # Jetson (Linux 기반)
            print(f"[INFO] Running on {system_name} - Using Remote ChromaDB Server")
            return chromadb.HttpClient(host="chromadb-server", port=8001, ssl=False)  # Jetson에서 ChromaDB 서버에 연결


    def insert_data(self, data: List[Dict[str, Any]]) -> None:
        """데이터(문서)를 ChromaDB 컬렉션에 삽입"""
        for doc in data:
            self.collection.add(
                ids=[doc["id"]],
                metadatas=[doc.get("metadata", {})],
                documents=[doc["text"]],
            )
            print(f"Document({doc['id']}) saved successfully!")
        print(f"All documents saved successfully!!")


    def search_data(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """검색어(query)와 유사한 문서 검색"""
        results = self.collection.query(query_texts=[query], n_results=top_k)
        return [
            {"id": doc_id, "text": text, "metadata": metadata}
            for doc_id, text, metadata in zip(results["ids"][0], results["documents"][0], results["metadatas"][0])
        ]


    def update_data(self, doc_id: str, new_text: str, new_metadata: Dict[str, Any] = None) -> None:
        """특정 문서를 수정"""
        self.collection.update(
            ids=[doc_id],
            documents=[new_text],
            metadatas=[new_metadata or {}],
        )


    def delete_data(self, doc_id: str) -> None:
        """특정 문서 삭제"""
        self.collection.delete(ids=[doc_id])


    def remove_collection(self) -> None:
        """컬렉션 삭제"""
        self.client.delete_collection(self.collection.name)


# FastAPI와 연동하는 Dependency Injection 함수
def get_chroma_collection(app, collection_name: str) -> ChromaCollection:
    """FastAPI에서 ChromaDB Collection을 관리하도록 하는 함수
    
    - 프로젝트 관련 collection 생성 및 app.state에 저장
    Args:
        app (FastAPI): FastAPI 인스턴스
        collection_name (str): 사용할 ChromaDB 컬렉션 이름

    Returns:
        ChromaCollection: ChromaDB Collection 인스턴스
    """
    if not hasattr(app.state, "chromadb_collections"):
        app.state.chromadb_collections = {}

    if collection_name not in app.state.chromadb_collections:
        app.state.chromadb_collections[collection_name] = ChromaCollection(collection_name=collection_name)

    return app.state.chromadb_collections[collection_name]
