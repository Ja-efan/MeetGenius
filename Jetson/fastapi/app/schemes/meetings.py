""" 회의 및 안건 관련 모델 정의 """

from pydantic import BaseModel, Field

class Message(BaseModel):
    """메시지 모델"""
    type: str = Field(..., description="메시지 타입(plain, query, rag)")
    content: str = Field(..., description="메시지 내용")

class AgendaBase(BaseModel):
    """안건 기본 모델"""
    agenda_id: int = Field(..., description="안건 id")
    agenda_title: str = Field(..., description="안건명")
    
    
class AgendaList(BaseModel):
    project_id: str = Field(..., description="프로젝트 id")
    agenda_list: list[AgendaBase] = Field(..., description="안건 목록")


class PrepareMeetingOut(BaseModel):
    """회의 준비 완료 모델"""
    result: bool = Field(..., description="결과")
    message: str = Field(..., description="메시지")


class NextAgendaOut(BaseModel):
    """다음 안건 모델"""
    stt_running: bool = Field(..., description="STT 실행 상태")
    agenda_docs: list = Field(..., description="안건과 유사한 문서 id 목록")