from pydantic import BaseModel

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
