from fastapi import FastAPI
from pathlib import Path
from chromadb import PersistentClient
from app.schemes.documents import DocumentList
from app.utils.llm_utils import load_embedding_model
from app.utils import logging_config
import numpy as np
import chromadb
import platform
from dotenv import load_dotenv
from typing import Any, Dict

# 로깅 설정
logger = logging_config.app_logger

# 환경 변수 로드
load_dotenv()


class ProjectCollection:
    def __init__(self, project_id: int, app: FastAPI):
        """ProjectCollection 생성자

        Args:
            project_id (int): 프로젝트 ID
            app (FastAPI): FastAPI 인스턴스
        """
        if len(str(project_id)) < 3 :
            self.project_id = "000" + str(project_id) # ChromaDB는 int 지원하지 않으므로 str으로 변환
        else:
            self.project_id = str(project_id)

        # self.project_id = str(project_id) # ChromaDB는 int 지원하지 않으므로 str으로 변환
        self.app = app  # FastAPI 인스턴스 저장
        
        self.client = self._get_client()
        self.collection = self.client.get_or_create_collection(self.project_id,
                                                               metadata={"hnsw:space": "cosine"})

    
    def _get_client(self):
        """운영 체제에 따라 적절한 ChromaDB 클라이언트를 선택"""
        system_name = platform.system()
        
        if system_name in ["Windows", "Darwin"]:  # Windows & MacOS (Darwin)
            logger.info(f"Running on {system_name} - Using Local ChromaDB Client")
            base_dir = Path(__file__).resolve().parent.parent  # 프로젝트 루트
            db_path = base_dir / "vector_db"
            logger.info(f"ChromaDB 데이터베이스 경로: {str(db_path)}")
            return PersistentClient(path=str(db_path))
    
        else:  # Jetson (Linux 기반)
            logger.info(f"Running on {system_name} - Using Remote ChromaDB Server")
            return chromadb.HttpClient(host="chromadb-server", port=8001, ssl=False)  # Jetson에서 ChromaDB 서버에 연결


    def insert_documents(self, documents: DocumentList):
        """문서 삽입"""
        
        # FastAPI의 최신 app.state 가져오기
        if not hasattr(self.app.state, "embedding_model"):
            self.app.state.embedding_model = load_embedding_model(self.app)

        model = self.app.state.embedding_model  # 최신 embedding_model 가져오기
        
        inserted_ids = []
        for document in documents.documents:  # documents는 DocumentList 객체
            sentence = [f"passage: {document.document_content}"]
            embeddings = model.encode(sentence)

            self.collection.add(
                ids=[str(document.document_id)],  # ChromaDB의 ID는 문자열이어야 함
                documents=[document.document_content],
                embeddings=embeddings,
                metadatas=[document.document_metadata.model_dump()]
            )

            inserted_ids.append(document.document_id)
        
        return inserted_ids


    def get_documents(self):
        """ 모든 문서 조회 """
        documents = self.collection.get(
            include=["documents", "metadatas"]
        )
        return documents 
    

    def delete_documents(self, doc_id: int) -> bool:
        """특정 문서 ID가 존재하는지 확인하고, 존재하면 삭제 후 True 반환, 없으면 False 반환"""
        
        # ChromaDB의 ID는 문자열이므로 변환
        doc_id_str = str(doc_id)

        logger.info(f"Deleting document: {doc_id_str}")
        # 현재 저장된 문서 목록 조회
        existing_docs = self.collection.get(ids=[doc_id_str], include=["documents"])
        
        # 문서가 존재하는지 확인
        if not existing_docs["documents"]:
            logger.info(f"Document {doc_id} not found.")    
            return False  # 문서가 존재하지 않음

        # 문서 삭제
        self.collection.delete(ids=[doc_id_str])
        logger.info(f"Deleted document: {doc_id}")
        
        return True  # 삭제 성공


    def search_documents(self, query_embedding: list, top_k: int = 1):
        """검색어 임베딩(query_embedding)과 유사한 문서 검색"""
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        return results


    def remove_collection(self) -> None:
        """컬렉션 삭제"""
        self.client.delete_collection(self.collection.name)


    def get_agenda_docs(self, agenda: str, top_k: int = 3):
        """안건명(agenda)과 유사한 문서 검색 및 문서 id 반환"""
        
        # 안건명 포맷팅 (KoE5 모델 사용)
        formatted_agenda = [f"query: {agenda}"]

        # 임베딩 임베딩 결과 
        agenda_embedding = self.app.state.embedding_model.encode(formatted_agenda)
        # 안건명과 유사한 문서 검색
        results = self.collection.query(query_embeddings=agenda_embedding, n_results=top_k)
        # 문서 id 반환 
        doc_ids = results["ids"][0]

        # 문서 id 반환 형식 변환
        doc_ids = [int(doc_id) for doc_id in doc_ids]
        
        return doc_ids