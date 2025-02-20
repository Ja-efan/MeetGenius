from fastapi import APIRouter, HTTPException, Depends
from app.schemes.documents import DocumentList, DocumentInsertResponse, DocumentDeleteResponse
from app.utils import chromadb_utils, llm_utils, logging_config
from app.dependencies import get_app

router = APIRouter(
    prefix="/api/v1/projects",
    tags=["projects"]
)

# 로깅 설정
logger = logging_config.app_logger


@router.post("/{project_id}/documents/")
def insert_documents(project_id: int, documents:DocumentList, app=Depends(get_app)):
    """
    문서 삽입 
    """

    # chromdb client 생성
    logger.info(f"Loading chromadb client...")
    chromadb_client = chromadb_utils.get_chromadb_client()
    logger.info(f"Chromadb client loaded successfully!")

    # 프로젝트 컬렉션 초기화 
    logger.info(f"Loading project collection...")
    project_collection = chromadb_utils.ProjectCollection(client=chromadb_client, project_id=project_id, app=app)
    logger.info(f"Project collection loaded successfully!")

    if not hasattr(app.state, "embedding_model"):
        app.state.embedding_model = llm_utils.load_embedding_model()

    # 문서 삽입 
    logger.info(f"Inserting documents into project collection...")
    inserted_ids = project_collection.insert_documents(app.state.embedding_model, documents)
    logger.info(f"Documents inserted successfully!")
    
    # 모델 언로드 
    # llm_utils.unload_models(app=app, embedding_model=True)
    # 변수 제거
    del chromadb_client
    del project_collection

    return DocumentInsertResponse(success=True, message="문서 삽입 완료", num_inserted=len(inserted_ids), inserted_ids=inserted_ids)


@router.get("/{project_id}/documents/")
async def get_documents(project_id: int, app=Depends(get_app)):
    """
    문서 조회 
    """
    
    project_id_str = "PJT-" + str(project_id)
    # chromadb client 생성
    if not hasattr(app.state, "chromadb_client"):
        app.state.chromadb_client = chromadb_utils.get_chromadb_client()
        logger.info(f"Chromadb client created successfully!")

    # 프로젝트 컬렉션 초기화 
    if not hasattr(app.state, "project_collection"):
        logger.info(f"Project collection not found in app.state, creating new one...")
        app.state.project_collection = chromadb_utils.ProjectCollection(client=app.state.chromadb_client, project_id=project_id, app=app)
        logger.info(f"Project collection created successfully!")

    if project_id_str not in app.state.chromadb_client.list_collections():
        logger.info(f"Project {project_id_str} not found in chromadb client")
        return {"message": f"{project_id_str}이 존재하지 않습니다."}

    documents = app.state.project_collection.get_documents()
    
    if not documents:
        logger.info(f"No documents found for {project_id_str}")
        raise HTTPException(status_code=404, detail=f"{project_id_str}에 문서가 존재하지 않습니다.")
    
    logger.info(f"{project_id_str} documents: {documents['ids']}")
    
    return {"message": "문서 조회 완료", "documents": documents}   


@router.delete("/{project_id}/documents/{document_id}/")
async def delete_document(project_id: int, document_id: int, app=Depends(get_app)):
    """
    문서 삭제
    """
    # chromadb client 생성
    if not hasattr(app.state, "chromadb_client"):
        app.state.chromadb_client = chromadb_utils.get_chromadb_client()
        logger.info(f"Chromadb client created successfully!")

    # 프로젝트 컬렉션 초기화 
    if not hasattr(app.state, "project_collection"):
        logger.info(f"Project collection not found for project {project_id}. Creating new one...")
        app.state.project_collection = chromadb_utils.ProjectCollection(client=app.state.chromadb_client, project_id=project_id, app=app)
        
    if app.state.project_collection.project_id != "PJT-" + str(project_id):
        logger.info(f"Project collection found for project {project_id}, but it's not the correct one. Creating new one...")
        app.state.project_collection = chromadb_utils.ProjectCollection(client=app.state.chromadb_client, project_id=project_id, app=app)

    documents = app.state.project_collection.get_documents() # 삭제하려는 문서 존재 확인(get_documents)
    
    
    # logger.info(f"documents: {documents}")
    
    metadatas = documents['metadatas']
    document_ids = [metadata['document_id'] for metadata in metadatas]
    
    
    # 프로젝트에 문서가 존재하지 않는 경우
    if not document_ids:
        logger.info(f"No documents found for project {project_id}")
        raise HTTPException(status_code=404, detail=f"프로젝트 {project_id}에 문서가 존재하지 않습니다.")
    
    # 삭제하려는 문서가 존재하지 않는 경우
    if document_id not in document_ids:
        logger.info(f"Document {document_id} not found in project {project_id}")
        raise HTTPException(status_code=404, detail=f"문서 {document_id}를 찾을 수 없습니다.")
    
    logger.info(f"Deleting document {document_id} from project collection...")
    result = app.state.project_collection.delete_document(document_id)
    logger.info(f"Document {document_id} deleted successfully!")

    return DocumentDeleteResponse(success=result, message=f"문서 {document_id} 삭제 완료", num_deleted=1, deleted_ids=[document_id])
