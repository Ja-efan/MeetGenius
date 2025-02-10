from fastapi import FastAPI
from pathlib import Path
from chromadb import PersistentClient
from app.models.documents import DocumentList
from app.utils.llm_utils import load_embedding_model
import numpy as np
import chromadb
import platform
from dotenv import load_dotenv
from typing import Any, Dict, List

# 환경 변수 로드
load_dotenv()

class ProjectCollection:
    def __init__(self, project_id: int, app: FastAPI):
        """ProjectCollection 생성자

        Args:
            project_id (int): 프로젝트 ID
            app (FastAPI): FastAPI 인스턴스
        """
        self.project_id = str(project_id) # ChromaDB는 int 지원하지 않으므로 str으로 변환
        self.app = app  # FastAPI 인스턴스 저장
        
        self.client = self._get_client()
        self.collection = self.client.get_or_create_collection(self.project_id)

    
    def _get_client(self):
        """운영 체제에 따라 적절한 ChromaDB 클라이언트를 선택"""
        system_name = platform.system()
        
        if system_name in ["Windows", "Darwin"]:  # Windows & MacOS (Darwin)
            print(f"[INFO] Running on {system_name} - Using Local ChromaDB Client")
            base_dir = Path(__file__).resolve().parent.parent  # 프로젝트 루트
            db_path = base_dir / "vector_db"
            print(f"[ChromaDB] 데이터베이스 경로: {str(db_path)}")
            return PersistentClient(path=str(db_path))
    
        else:  # Jetson (Linux 기반)
            print(f"[INFO] Running on {system_name} - Using Remote ChromaDB Server")
            return chromadb.HttpClient(host="chromadb-server", port=8001, ssl=False)  # Jetson에서 ChromaDB 서버에 연결


    def insert_documents(self, documents: DocumentList):
        """문서 삽입"""
        
        # FastAPI의 최신 app.state 가져오기
        if not hasattr(self.app.state, "embedding_model"):
            load_embedding_model(self.app.state)

        model = self.app.state.embedding_model  # 최신 embedding_model 가져오기
        
        for document in documents.documents:  # documents는 DocumentList 객체
            sentence = [f"passage: {document.document_content}"]
            embeddings = model.encode(sentence)

            self.collection.add(
                ids=[str(document.document_id)],  # ChromaDB의 ID는 문자열이어야 함
                documents=[document.document_content],
                embeddings=embeddings,
                metadatas=[document.document_metadata.model_dump()]
            )
        
        documents = self.collection.get(
            include=["documents", "embeddings", "metadatas"]
        )
        print(f"documents: {documents}")
        return documents


    def get_documents(self, project_id: str):
        """ 모든 문서 조회 """
        documents = self.collection.get(
            where={"project_id": str(project_id)},
            include=["documents", "embeddings", "metadatas"]
        )

        # 🔥 numpy array → list 변환 (JSON 직렬화 가능하도록 변환)
        if "embeddings" in documents and isinstance(documents["embeddings"], np.ndarray):
            documents["embeddings"] = documents["embeddings"].tolist()

        print(f"✅ [DEBUG] Retrieved documents: {documents}")  # 디버깅 출력

        return documents  # ✅ JSON 변환 가능
    

    def del_documents(self, document_id: int):
        """
        문서 삭제
        """
        # 삭제하려는 문서 존재 확인하는 건 projects/del_documents에서 진행함
        # 문서 삭제 진행
        self.collection.delete(ids=[str(document_id)])

        # 정상 삭제 여부 확인
        updated_documents = self.collection.get(include=["documents"])
        updated_document_ids = [str(doc["id"]) for doc in updated_documents.get("documents", [])]
        if str(document_id) in updated_document_ids:
            print(f"❌ [ERROR] Document {document_id} deletion failed.")
            return False

        print(f"✅ [INFO] Document {document_id} deleted successfully.")
        return True

# =======


    def search_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """검색어(query)와 유사한 문서 검색"""
        results = self.collection.query(query_texts=[query], n_results=top_k)
        return [
            {"id": int(doc_id), "text": text, "metadata": metadata}
            for doc_id, text, metadata in zip(results["ids"][0], results["documents"][0], results["metadatas"][0])
        ]


    def update_documents(self, doc_id: str, new_text: str, new_metadata: Dict[str, Any] = None) -> None:
        """특정 문서를 수정"""
        self.collection.update(
            ids=[doc_id],
            documents=[new_text],
            metadatas=[new_metadata if new_metadata is not None else {}]
        )

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


        return doc_ids