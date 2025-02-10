""" 프로젝트 문서 관련 라우트 """

import fastapi
from fastapi import FastAPI, APIRouter, Depends
from app.schemes import DocumentList
from typing import Any
from app.utils import chromadb_utils, llm_utils
from app.dependencies import get_app_state, get_app
from app.schemes import DocumentInsertResponse, DocumentDeleteResponse

router = APIRouter(
    prefix="/api/v1/projects",
)   



@router.post("/{project_id}/documents")
async def insert_document(documents: DocumentList, project_id: str, app: FastAPI = Depends(get_app)):
    """ 프로젝트 문서 삽입 엔드포인트


    Args:
        document (Document): 문서 정보
        app (FastAPI, optional): FastAPI 인스턴스. Defaults to Depends().

    Returns:
        json: 문서 삽입 여부 및 메시지
    """
    
    # 프로젝트 컬렉션 초기화 
    if not hasattr(app.state, "project_collection"):
        print(f"🔄 [INFO] Project collection not found in app.state, creating new one...")
        app.state.project_collection = chromadb_utils.ProjectCollection(str(project_id), app)
        print(f"✅ [INFO] Project collection created successfully!")

    # 문서 삽입 
    print(f"🔄 [INFO] Inserting documents into project collection...")
    inserted_ids = app.state.project_collection.insert_documents(documents)
    print(f"✅ [INFO] Documents inserted successfully!")

    return DocumentInsertResponse(success=True,
                                  message="문서 삽입 완료", 
                                  num_inserted=len(documents), 
                                  inserted_ids=inserted_ids)


@router.delete("/{project_id}/documents/{document_id}")
async def delete_project_document(project_id: str, document_id: str, app: FastAPI = Depends(get_app)):

    """ 프로젝트 문서 삭제 엔드포인트

    Args:
        project_id (str): 프로젝트 id
        document_id (str): 문서 id
        app (FastAPI, optional): FastAPI 인스턴스. Defaults to Depends(get_app).

    Returns:
        _type_: _description_
    """
    # 프로젝트 컬렉션 
    if not hasattr(app.state, "project_collection"):
        print(f"🔄 [INFO] Project collection not found in app.state, creating new one...")
        app.state.project_collection = chromadb_utils.ProjectCollection(str(project_id), app)
        print(f"✅ [INFO] Project collection created successfully!")
        
    # 문서 삭제 
    print(f"🔄 [INFO] Deleting document from project collection...")
    app.state.project_collection.delete_document(document_id)
    print(f"✅ [INFO] Document deleted successfully!")
    
    return DocumentDeleteResponse(success=True,
                                  message="문서 삭제 완료", 
                                  num_deleted=1, 
                                  deleted_ids=[document_id])