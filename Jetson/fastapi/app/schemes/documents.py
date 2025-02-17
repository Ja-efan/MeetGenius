from pydantic import BaseModel

class DocumentMetadata(BaseModel):
    project_id: int
    document_type: int
    meeting_id: int  # 보고서인 경우 미팅 아이디 -1

class Document(BaseModel):
    document_id: int
    document_content: str
    document_metadata: DocumentMetadata
    
class DocumentList(BaseModel):
    documents: list[Document]
    
class DocumentInsertResponse(BaseModel):
    success: bool
    message: str
    num_inserted: int
    inserted_ids: list[int] | None

class DocumentDeleteResponse(BaseModel):
    success: bool
    message: str
    num_deleted: int
    deleted_ids: list[int] | None
