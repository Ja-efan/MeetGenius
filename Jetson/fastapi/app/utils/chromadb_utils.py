from curses.ascii import isdigit
from fastapi import FastAPI
import chromadb
from chromadb import PersistentClient, HttpClient
from numpy import where
from app.schemes.documents import DocumentList
from app.utils.llm_utils import load_embedding_model
from app.utils import logging_config
import platform
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# 로깅 설정
logger = logging_config.app_logger

def get_chromadb_client():
    """운영 체제에 따라 적절한 ChromaDB 클라이언트를 선택"""
    system_name = platform.system()
    
    if system_name in ["Windows", "Darwin"]:  # Windows & MacOS (Darwin)
        logger.info(f"Running on '{system_name}' - Using Local ChromaDB Client")
        persistent_client_path = os.getenv("CHROMADB_PERSISTENT_CLIENT_PATH")
        logger.info(f"ChromaDB 데이터베이스 경로: {persistent_client_path}")
        return PersistentClient(path=persistent_client_path)
    
    else:  # Jetson (Linux 기반)
        logger.info(f"Running on {system_name} - Using Remote ChromaDB Server")
        logger.info(f"ChromaDB HTTP Client Host: {os.getenv('CHROMADB_HTTP_CLIENT_HOST')}")
        logger.info(f"ChromaDB HTTP Client Port: {int(os.getenv('CHROMADB_HTTP_CLIENT_PORT'))}")
        logger.info(f"ChromaDB HTTP Client SSL: {bool(os.getenv('CHROMADB_HTTP_CLIENT_SSL'))}")

        return HttpClient(host= os.getenv("CHROMADB_HTTP_CLIENT_HOST"),  
                              port=int(os.getenv("CHROMADB_HTTP_CLIENT_PORT")), 
                              ssl=False)  # Jetson에서 ChromaDB 컨테이너 서버에 연결

        # return HttpClient(host='chromadb-server', port=8001, ssl=False)
class ProjectCollection:
    def __init__(self, client, project_id: int, app: FastAPI):
        """ProjectCollection 생성자

        Args:
            project_id (int): 프로젝트 ID
            app (FastAPI): FastAPI 인스턴스
        """
        if isinstance(project_id, int):
            self.project_id = "PJT-" + str(project_id) # ChromaDB는 int 지원하지 않으므로 str으로 변환
        else:
            self.project_id = project_id
        logger.info(f"Project ID: {self.project_id}")
        app.state.project_id = self.project_id
        
        self.app = app  # FastAPI 인스턴스 저장
        
        self.client = client
        self.collection = self.client.get_or_create_collection(self.project_id,
                                                               metadata={"hnsw:space": "cosine"})


    def insert_documents(self, embedding_model, documents_list: DocumentList):
        """문서 삽입"""
        
        # # FastAPI의 최신 app.state 가져오기
        # if not hasattr(self.app.state, "embedding_model"):
        #     self.app.state.embedding_model = load_embedding_model(self.app)

        # model = self.app.state.embedding_model  # 최신 embedding_model 가져오기
        
        # text splitter 생성 (chunk size 1000, overlap 100)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

        inserted_ids = []
        for document in documents_list.documents:  # documents는 DocumentList 객체

            chunks = text_splitter.split_text(document.document_content)

            for idx, chunk in enumerate(chunks):
                # 각 청크마다 ID 생성 (document.document_id + _ + idx)
                chunk_id = f"{document.document_id}_{idx}"
                # 청크 임베딩
                sentence = [f"passage: {chunk}"]
                embeddings = embedding_model.encode(sentence)

                # 청크 메타데이터 생성
                chunk_metadata = {
                    "document_id": document.document_id,
                    "chunk_id": chunk_id,
                    "chunk_index": idx,
                    "chunk_size": len(chunk),
                    "document_type": document.document_metadata.document_type,
                    "meeting_id": document.document_metadata.meeting_id,
                    "project_id": self.project_id
                }

                # 청크 추가
                self.collection.add(
                    ids=[chunk_id],
                    documents=[chunk],
                    embeddings=embeddings,
                    metadatas=[chunk_metadata]
                )
                
            inserted_ids.append(document.document_id)
        
        return inserted_ids
    
    def insert_meeting_transcript(self, embedding_model, meeting_id, document_id, transcript_text):
        """요약전 회의록 텍스트 삽입"""
        # 회의록 텍스트 임베딩

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

        chunks = text_splitter.split_text(transcript_text)
        for idx, chunk in enumerate(chunks):
            # 각 청크마다 ID 생성 (document_id + _ + idx)
            chunk_id = f"{document_id}_{idx}"
            # 청크 임베딩
            sentence = [f"passage: {chunk}"]
            embeddings = embedding_model.encode(sentence)

            # 청크 메타데이터 생성
            chunk_metadata = {
                "document_id": document_id,
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "chunk_size": len(chunk),
                "document_type": 0,  # 요약 전 회의록
                "meeting_id": meeting_id,
                "project_id": self.project_id
            }

            # 청크 추가
            self.collection.add(
                ids=[chunk_id],
                documents=[chunk],
                embeddings=embeddings,
                metadatas=[chunk_metadata]
            )
            
        inserted_id = document_id

        return inserted_id        
        
        


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
        """안건과 유사한 원본 문서를 검색하여 document_id를 반환합니다.
        
        원본 문서를 여러 청크로 나누어 저장하더라도, 쿼리 시에 top_k보다 여유 있게 검색한 후
        각 문서별 가장 유사한 청크(즉, cosine 유사도가 가장 높은 청크)의 점수를 대표값으로 하여
        정렬한 뒤 상위 top_k개의 원본 문서를 선택합니다.
        """
        # 안건 포맷팅 (예: KoE5 모델 사용)
        formatted_agenda = [f"query: {agenda}"]

        # 원본 문서 개수를 의미하는 top_k에 맞추기 위해, 청크 검색 수를 보수적으로 늘림.
        multiplier = 3
        n_results = top_k * multiplier

        # 안건 임베딩 생성
        agenda_embedding = self.app.state.embedding_model.encode(formatted_agenda)

        # 각 청크에 대한 메타데이터와 거리를 포함하여 쿼리 실행
        results = self.collection.query(
            query_embeddings=agenda_embedding,
            where={"project_id": self.project_id},
            n_results=n_results,
            include=["metadatas", "distances"]
        )
        
        logger.info(f"query: {agenda}")
        logger.info(f"results: {results}")

        # 결과는 리스트의 리스트 형식으로 반환되므로 첫 번째 요소 선택
        metadatas = results.get("metadatas", [])[0] if results.get("metadatas") else []
        distances = results.get("distances", [])[0] if results.get("distances") else []

        # 각 document_id별로 가장 높은 cosine similarity (1 - distance)를 선택
        doc_similarities = {}
        for metadata, distance in zip(metadatas, distances):
            # cosine distance => cosine similarity로 변환
            cosine_similarity = 1 - distance
            doc_id = metadata.get("document_id")
            if doc_id in doc_similarities:
                # 이미 등록된 값보다 높은 유사도를 가진 청크가 있다면 업데이트
                if cosine_similarity > doc_similarities[doc_id]:
                    doc_similarities[doc_id] = cosine_similarity
            else:
                doc_similarities[doc_id] = cosine_similarity

        # cosine similarity가 높은 순서대로 정렬 (내림차순)
        sorted_docs = sorted(doc_similarities.items(), key=lambda x: x[1], reverse=True)
        unique_doc_ids = [doc_id for doc_id, sim in sorted_docs][:top_k]

        # 원래 document_id가 숫자라면 int 타입으로 변환한 후 반환
        return [int(doc_id) for doc_id in unique_doc_ids]