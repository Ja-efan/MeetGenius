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
        "document_name": str,   
        "project_id": str,
        "project_name": str,
        "document_type": str,
        "department_name": str | None,
        "content": list[str],
        "meeting_name": str | None,
        "agenda_name": str | None
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
from models import Document, EmbeddingDocument, Metadata
import json

from core import chromadb_utils, llm_utils

router = APIRouter(
    prefix="/api/v1/projects",
)



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

    # 문서 내용 임베딩 변환 -> 현재 한 문장씩 임베딩 -> 배치 단위로 임베딩 할 수 있게 디벨롭되면 좋을 것 같다.
    embeddings = []
    for sentence in document.content:
        embedding = model.encode([f"passage: {sentence}"])
        embeddings.append(embedding)
    
    embedding_document = EmbeddingDocument(
        ids=[document.document_id],
        documents=document.content,
        embeddings=embeddings,
        metadatas=[
            {
                "document_id": document.document_id,  # 이걸 ids로 사용할 수 있을 것 같음
                "project_id": document.project_id,  # 프로젝트 id
                "project_name": document.project_name,  # 프로젝트 이름
                "document_name": document.document_name,  # 문서 이름
                "document_type": document.document_type,  # 문서 타입

            }
        ]
    )
    
    # 문서 정보 저장
    project_collection.insert_data([embedding_document])

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