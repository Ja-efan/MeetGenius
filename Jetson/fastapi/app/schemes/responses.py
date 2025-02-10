"""
    엔드포인트 응답 모델 정의
"""

from pydantic import BaseModel, Field
from meetings import AgendaSummary


class PrepareMeetingResponse(BaseModel):

    """회의 준비 완료 모델"""
    result: bool = Field(..., description="결과")
    message: str = Field(..., description="메시지")


class NextAgendaResponse(BaseModel):
    """다음 안건 모델"""
    stt_running: bool = Field(..., description="STT 실행 상태")
    agenda_docs: list = Field(..., description="안건과 유사한 문서 id 목록")
    

class EndMeetingResponse(BaseModel):
    """회의 종료 모델"""
    meeting_id: int = Field(..., description="회의 id")
    result: bool = Field(..., description="결과(True: 회의 종료, False: 회의 진행중)")


class SummaryResponse(BaseModel):
    """회의 요약 모델"""
    meeting_id: int = Field(..., description="회의 id")
    summary: list[AgendaSummary] = Field(..., description="회의 요약 목록(안건별 요약)")

