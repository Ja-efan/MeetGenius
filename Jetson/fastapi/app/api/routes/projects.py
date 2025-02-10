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

from fastapi import APIRouter, HTTPException, Depends
from app.models.documents import DocumentList
from app.dependencies import get_app_state, get_app
from app.utils import chromadb_utils, llm_utils


router = APIRouter(
    prefix="/api/v1/projects",
    tags=["projects"]
)


@router.post("/{project_id}/documents")
async def insert_documents(project_id: int, documents:DocumentList, app=Depends(get_app)):
    """
    문서 삽입 
    """
    # 프로젝트 컬렉션 초기화 
    if not hasattr(app.state, "project_collection"):
        print(f"🔄 [INFO] Project collection not found in app.state, creating new one...")
        app.state.project_collection = chromadb_utils.ProjectCollection(str(project_id), app)
        print(f"✅ [INFO] Project collection created successfully!")

    # 문서 삽입 
    print(f"🔄 [INFO] Inserting documents into project collection...")
    app.state.project_collection.insert_documents(documents)
    print(f"✅ [INFO] Documents inserted successfully!")

    return {"message": "문서 삽입 완료", "documents": documents}


@router.get("/{project_id}/documents")
async def get_documents(project_id: int, app=Depends(get_app)):
    """
    문서 조회 
    """
    if not hasattr(app.state, "project_collection"):
        print(f"🔄 [INFO] Project collection not found in app.state, creating new one...")
        app.state.project_collection = chromadb_utils.ProjectCollection(str(project_id), app)
        print(f"✅ [INFO] Project collection created successfully!")

    documents = app.state.project_collection.get_documents(project_id)
    return {"message": "문서 조회 완료", "documents": documents}   


@router.delete("/{project_id}/documents/{document_id}")
async def del_document(project_id: int, document_id: int, app=Depends(get_app)):
    """
    문서 삭제
    """
    if not hasattr(app.state, "project_collection"):
        print(f"⚠️ [WARNING] Project collection not found for project {project_id}. Creating new one...")
        app.state.project_collection = chromadb_utils.ProjectCollection(str(project_id), app)
    
    documents = app.state.project_collection.get_documents(project_id) # 삭제하려는 문서 존재 확인(get_documents)
    document_ids = [doc["id"] for doc in documents]
    if document_id not in document_ids:
        raise HTTPException(status_code=404, detail=f"문서 {document_id}를 찾을 수 없습니다.")
    
    print(f"🔄 [INFO] Deleting document {document_id} from project collection...")
    app.state.project_collection.del_documents(document_id)
    print(f"✅ [INFO] Document {document_id} deleted successfully!")

    return {"message": f"문서 {document_id} 삭제 완료"}