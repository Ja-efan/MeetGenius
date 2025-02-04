'''
chroma_db 연동  
'''

import chromadb 
from chromadb import Settings
import os
from dotenv import load_dotenv
from typing import Any, Dict, List

# 환경 변수 로드
load_dotenv()


class ChromaCollection:
    """ChromaDB Collection 관리 클래스 """


    def __init__(self, collection_name: str):
        """ChromaCollection 생성자

        Args:
            collection_name (str): 사용할 ChromaDB 컬렉션 이름
        """
        self.client = chromadb.HttpClient(
            host="chromadb-server",
            port=8001,
            ssl=False,
        )

        self.collection = self.client.get_or_create_collection(nemae=collection_name)
    
    def insert_data(self, data: List[Dict[str, Any]]) -> None:
        """데이터(문서)를 ChromaDB 컬렉션에 삽입

        Args:
            data (List[Dict[str, Any]]): 삽입할 문서 리스트 (id, metadata, text 포함) <- 문서 구조 정의되면 수정 필요 
        """
        for doc in data:
            self.collection.add(
                ids = [doc["id"]],
                metadatas = [doc.get("metadata", {})],
                documents = [doc["text"]],
            )
            print(f"Document({doc["id"]}) saved successfully!")
        
        print(f"All documents saved successfully!!")

    
    def search_data(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """검색어(query)와 유사한 문서 검색 함수 

        Args:
            query (str): 검색할 쿼리 텍스트 
            top_k (int, optional): 반환할 문서 개수. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: 검색된 문서 리스트 
        """
        results = self.collection.query(query_texts=[query], n_results=top_k)
        return [
            {"id": doc_id, "text": text, "metadata": metadata}
            for doc_id, text, metadata in zip(results["ids"][0], results["documents"][0], results["metadatas"][0])
        ]
    
    def update_data(self, doc_id: str, new_text: str, new_metadata: Dict[str, Any] = None) -> None:
        """ChromaDB 컬렉션에서 특정 문서를 수정

        Args:
            doc_id (str): 수정할 문서의 ID
            new_str (_type_): 새로운 텍스트 데이터
            new_metadata (Dict[str, Any], optional): 새로운 메타데이터터. Defaults to None.
        """
        self.collection.update(
            ids=[doc_id],
            documents=[new_text],
            metadatas=[new_metadata or {}],
        )

    def delete_data(self, doc_id: str) -> None:
        """특정 데이터(문서) 삭제 

        -> collection 내 저장된 doc_id 가 아닌 메타데이터 내의 문서 ID 로 삭제 가능하게 업데이트 필요 

        Args:
            doc_id (str): 삭제할 문서 ID
        """
        self.collecion.delete(ids=[doc_id])

    def remove_collection(self) -> None:
        """현재 컬렉션을 삭제 
        """

        self.client.delete_collection(self.collection.name)


# FastAPI와 연동하는 Dependency Injection 함수
def get_chroma_collection(app, collection_name: str) -> ChromaCollection:
    """FastAPI에서 ChromaDb Collection을 관리하도록 하는 함수 

    Args:
        app (FastAPI): FastAPI 애플리케이션 인스턴스 
        collection_name (str): 사용할 컬렉션 이름

    Returns:
        ChromaCollection: ChromaCollection 인스턴스 
    """
    # app.state.chromadb_collections: 
    # FastAPI 애플리케이션의 전역 상태(app.stete)에 저장되는 ChromaDB 컬렉션 객체들의 딕셔너리 
    if not hasattr(app.state, "chromadb_collections"):
        app.state.chromadb_collections = {}

    if collection_name not in app.state.chromadb_collections:
        app.state.chroamdb_collection[collection_name] = ChromaCollection(collection_name=collection_name)
    
    return app.state.chromadb_collections[collection_name]