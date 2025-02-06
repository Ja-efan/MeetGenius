"""
프로젝트 관련 문서 관리 엔드포인트

1. 프로젝트 문서 삽입  [POST]  /api/v1/projects/{project_id}/documents

    - 프로젝트 문서 삽입 
    - 문서 정보 수신 
    - 문서 내용 임베딩 변환
    - 문서 정보 저장 
    - 문서 삽입 완료 시 완료 처리 
    
    body:
    {
        "document_id": str,
        "project_id": str,
        "document_name": str,
        "document_content": str,
        "document_type": str
    }

    response:
    {
        "result": bool,
        "message": str
    }

2. 프로젝트 문서 삭제   [DELETE]  /api/v1/projects/{project_id}/documents/{document_id}

    - 프로젝트 문서 삭제 
    - 문서 삭제 완료 시 완료 처리 
    
    body:
    {
        "document_id": str,
        "project_id": str
    }

    response:
    {
        "result": bool,
"""

from fastapi import FastAPI, APIRouter, Depends
from pydantic import BaseModel
import json

from core import chromadb_utils, llm_utils

router = APIRouter(
    prefix="/api/v1/projects",
)


class DocumentBase(BaseModel):
    """문서 기본 모델"""
    document_id: str  # 문서 id
    project_id: str  # 프로젝트 id
    document_type: str  # 문서 타입
    department_name: str | None  # 부서 이름
    content: list[str]  # 문서 내용 (문장으로 구성된 리스트)
    
class Document(DocumentBase):
    """벡터 DB에 저장할 문서 원본 모델"""
    document_name: str  # 문서 이름
    project_name: str  # 프로젝트 이름
    meeting_name: str | None  # 회의 이름 (회의록 문서인 경우)
    agenda_name: str | None  # 안건 이름 (회의록 문서인 경우)

class Metadata(BaseModel):
    """벡터 DB에 저장할 문서 메타데이터 모델"""
    project_id: str | None # 프로젝트 id
    project_name: str | None  # 프로젝트 이름
    document_id: str  # 문서 id
    document_name: str  # 문서 이름
    document_type: str  # 문서 타입
    meeting_name: str | None  # 회의 이름 (회의록 문서인 경우)
    agenda_name: str | None  # 안건 이름 (회의록 문서인 경우)

class EmbeddingDocument(BaseModel):
    """벡터 DB에 저장할 문서 임베딩 모델"""
    ids: str  # 각 벡터의 고유 ID -> 문서 id
    documents: str  # 문서 내용  -> 텍스트 데이터 or list[str]
    embeddings: list[list[float]]  # 임베딩 변환 된 문서 내용
    metadatas: Metadata  # 문서 메타데이터

@router.post("/{project_id}/documents")
async def insert_project_document(document: Document, app: FastAPI = Depends()) -> json:
    """ 프로젝트 문서 삽입 엔드포인트

    Args:
        document (Document): 문서 정보
        app (FastAPI, optional): FastAPI 인스턴스. Defaults to Depends().

    Returns:
        json: 문서 삽입 여부 및 메시지
    """
    # 프로젝트 컬렉션 
    project_collection = chromadb_utils.get_project_collection(app=app, project_id=document.project_id)

    # 컬렉션에 문서가 존재하는지 확인
    if project_collection.get_document(document.document_id):  # get_document 메서드 추가 필요
        return {"result": False, "message": f"문서({document.document_id})가 이미 존재합니다."}

    # 임베딩 모델 로드 
    model = llm_utils.load_embedding_model(app=app)

    # 문서 내용 임베딩 변환 -> 현재 한 문장씩 임베딩 -> 배치 단위로 임베딩 할 수 있게 디벨롬
    embeddings = []
    for sentence in document.document_content:
        embedding = model.encode([f"passage: {sentence}"])
        embeddings.append(embedding)
    
    embedding_document = EmbeddingDocument(
        ids=[document_id],
        metadatas=[
            {
                "document_id": document_id,  # 이걸 ids로 사용할 수 있을 것 같음
                "project_id": project_id,  # 프로젝트 id
                "project_name": document.project_name,  # 프로젝트 이름
                "document_name": document.document_name,  # 문서 이름
                "document_type": document.document_type,  # 문서 타입

            }
        ],
        documents=document.document_content,
        embeddings=embeddings
    )
    
    # 문서 정보 저장
    project_collection.insert_data(
        ids=[document_id],
        metadatas=[{"source": document.document_name}],
        documents=[document.document_content],
        embeddings=embeddings
    )

    # 문서 삽입
    app.state.project_collection.insert_data(document)
    return {"result": True, "message": "문서 삽입 완료"}


@router.delete("/{project_id}/documents/{document_id}")
async def delete_project_document(project_id: str, document_id: str, app: FastAPI = Depends()) -> json:
    """ 프로젝트 문서 삭제 엔드포인트

    Args:
        project_id (str): 프로젝트 id
        document_id (str): 문서 id
        app (FastAPI, optional): FastAPI 인스턴스. Defaults to Depends().

    Returns:
        json: 문서 삭제 여부 및 메시지
    """
    # 문서 삭제
    app.state.project_collection.delete_data(document_id)