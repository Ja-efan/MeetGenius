from fastapi import APIRouter, HTTPException, Depends
from app.schemes.documents import DocumentList, DocumentInsertResponse, DocumentDeleteResponse
from app.utils import chromadb_utils
from app.dependencies import get_app

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
    inserted_ids = app.state.project_collection.insert_documents(documents)
    print(f"✅ [INFO] Documents inserted successfully!")

    return DocumentInsertResponse(success=True, message="문서 삽입 완료", num_inserted=len(inserted_ids), inserted_ids=inserted_ids)


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
async def delete_document(project_id: int, document_id: int, app=Depends(get_app)):
    """
    문서 삭제
    """
    # 프로젝트 컬렉션 초기화 
    if not hasattr(app.state, "project_collection"):
        print(f"⚠️ [WARNING] Project collection not found for project {project_id}. Creating new one...")
        app.state.project_collection = chromadb_utils.ProjectCollection(str(project_id), app)
    else:
        if app.state.project_collection.project_id != str(project_id):
            print(f"⚠️ [WARNING] Project collection found for project {project_id}, but it's not the correct one. Creating new one...")
            app.state.project_collection = chromadb_utils.ProjectCollection(str(project_id), app)

    documents = app.state.project_collection.get_documents(project_id) # 삭제하려는 문서 존재 확인(get_documents)
    
    document_ids = documents.get('ids', [])
    
    # 프로젝트에 문서가 존재하지 않는 경우
    if not document_ids:
        print(f"⚠️ [WARNING] No documents found for project {project_id}")
        raise HTTPException(status_code=404, detail=f"프로젝트 {project_id}에 문서가 존재하지 않습니다.")
    
    # 삭제하려는 문서가 존재하지 않는 경우
    if str(document_id) not in document_ids:
        print(f"⚠️ [WARNING] Document {document_id} not found in project {project_id}")
        raise HTTPException(status_code=404, detail=f"문서 {document_id}를 찾을 수 없습니다.")
    
    print(f"🔄 [INFO] Deleting document {document_id} from project collection...")
    result = app.state.project_collection.delete_documents(document_id)
    print(f"✅ [INFO] Document {document_id} deleted successfully!")

    return DocumentDeleteResponse(success=result, message=f"문서 {document_id} 삭제 완료", num_deleted=1, deleted_ids=[document_id])
