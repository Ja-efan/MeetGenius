import chromadb
from chromadb import Settings
import os
import platform
from dotenv import load_dotenv
from typing import Any, Dict, List
from models import EmbeddingDocument
from fastapi import FastAPI, Depends

# 환경 변수 로드
load_dotenv()


class ChromaCollection:
    """ChromaDB Collection 관리 클래스"""

    def __init__(self, collection_name: str, app: FastAPI = Depends()):
        """ChromaCollection 생성자

        Args:
            collection_name (str): 사용할 ChromaDB 컬렉션 이름
            app (FastAPI): FastAPI 인스턴스
        """
        self.collection_name = collection_name
        self.client = self._get_chroma_client()

        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.app = app

    def _get_chroma_client(self):
        """운영 체제에 따라 적절한 ChromaDB 클라이언트를 선택"""

        system_name = platform.system()
        
        if system_name in ["Windows", "Darwin"]:  # Windows & MacOS (Darwin)
            print(f"[INFO] Running on {system_name} - Using Local ChromaDB Client")
            return chromadb.PersistentClient(path="./chroma_db")  # 로컬에서 ChromaDB 사용
        else:  # Jetson (Linux 기반)
            print(f"[INFO] Running on {system_name} - Using Remote ChromaDB Server")
            return chromadb.HttpClient(host="chromadb-server", port=8001, ssl=False)  # Jetson에서 ChromaDB 서버에 연결


    def insert_data(self, data: List[EmbeddingDocument]) -> None:
        """데이터(문서)를 ChromaDB 컬렉션에 삽입"""

        for embedding_document in data:
            self.collection.add(
                ids=[embedding_document.ids],
                metadatas=[embedding_document.metadatas],
                documents=[embedding_document.documents],
            )
            print(f"Document({embedding_document.ids}) saved successfully!")
            
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

    def search_documents_by_agenda(self, agenda: str, top_k: int = 3):
        """안건명(agenda)과 유사한 문서 검색"""
        # 임베딩 모델 
        model = self.app.state.embedding_model
        # 안건명 포맷팅 (KoE5 모델 사용)
        formatted_agenda = [f"query: {agenda}"]
        # 임베딩 임베딩 결과 
        agenda_embedding = model.encode(formatted_agenda)
        
        # collection
        collection = self.collection
        
        # 안건명과 유사한 문서 검색
        results = collection.query(query_embeddings=agenda_embedding, n_results=top_k)
        
        return results
        
# FastAPI와 연동하는 Dependency Injection 함수
def get_project_collection(project_id: str, app: FastAPI = Depends()) -> ChromaCollection:
    """FastAPI에서 ChromaDB Collection을 관리하도록 하는 함수
    
    - 프로젝트 관련 collection 생성 및 app.state에 저장
    Args:
        app (FastAPI): FastAPI 인스턴스
        collection_name (str): 사용할 ChromaDB 컬렉션 이름

    Returns:
        ChromaCollection: ChromaDB Collection 인스턴스
    """

    # chromadb_collections: 프로젝트 관련 collection 목록
    # chromadb_collections[collection_name]: 프로젝트 관련 collection 인스턴스

    # 초기화 되지 않았다면 초기화
    if not hasattr(app.state, "chromadb_collections"): 
        app.state.chromadb_collections = {}

    # project_id 컬렉션이 초기화 되지 않았다면 초기화
    if project_id not in app.state.chromadb_collections:
        app.state.chromadb_collections[project_id] = ChromaCollection(collection_name=project_id)

    return app.state.chromadb_collections[project_id]  # 프로젝트 관련 collection 인스턴스 반환
