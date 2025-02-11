from pydantic import BaseModel, Field
from typing import List

class STTMessage(BaseModel):
    """
    STT(음성 인식) 메시지 모델.
    메시지 타입은 'plain', 'query', 'rag' 등으로 지정하며,
    음성 인식 결과나 질의/응답 메시지를 포함합니다.
    """
    type: str = Field(..., description="메시지 타입 (예: 'plain', 'query', 'rag')")
    content: str = Field(..., description="메시지 내용")

class Agenda(BaseModel):
    """
    기본 안건 모델.
    회의에서 다루어질 안건의 고유 식별자와 제목을 포함합니다.
    """
    id: int = Field(..., description="안건 고유 식별자")
    title: str = Field(..., description="안건 제목")

class MeetingAgendas(BaseModel):
    """
    회의 준비 시 입력으로 사용되는 안건 목록 모델.
    프로젝트 식별자와 해당 프로젝트 내 회의에 사용할 안건들을 포함합니다.
    """
    project_id: int = Field(..., description="프로젝트 식별자")
    agendas: List[Agenda] = Field(..., description="회의 안건 목록")

class AgendaDetail(Agenda):
    """
    상세 안건 모델.
    기본 안건 모델에 회의 진행 중 기록된 상세 내용을 추가합니다.
    """
    content: str = Field(..., description="안건에 대한 상세 내용 또는 논의 결과")

class MeetingAgendaDetails(BaseModel):
    """
    회의 진행 중 또는 결과로 반환할 상세 안건 목록 모델.
    """
    agendas: List[AgendaDetail] = Field(..., description="상세 안건 목록")

class AgendaSummary(BaseModel):
    """
    안건별 회의록 요약 모델.
    각 안건의 제목, 원본 내용, 그리고 요약 내용을 포함합니다.
    """
    title: str = Field(..., description="안건 제목")
    original_content: str = Field(..., description="안건 원본 내용")
    summary: str = Field(..., description="안건 요약 내용")
