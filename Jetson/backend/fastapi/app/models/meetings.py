import datetime
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class AgendaBase(BaseModel):
    """MariaDB Agenda Table"""
    agenda_id: int = Field(..., description="안건 id")
    meeting_id: int = Field(..., description="회의 id")
    title: str = Field(..., min_length=1, max_length=200, description="안건명")

class Agenda(AgendaBase):
    project_id: str = Field(..., description="프로젝트 id")
    
class MeetingBase(BaseModel):
    """MariaDB Meeting Table"""
    meeting_id: id = Field(..., description="회의 id")
    meeting_room: int | None = Field(None, description="회의실 번호")
    meeting_title: str = Field(..., min_length=1, max_length=100, description="회의 제목")
    meeting_date: datetime = Field(..., description="회의 날짜")
    meeting_start_time: datetime = Field(..., description="회의 시작 시간")
    meeting_end_time: datetime = Field(..., description="회의 종료 시간")
    booker_id: str = Field(..., description="예약자 id")
    project_id: str = Field(..., description="프로젝트 id")
    
class Meeting(MeetingBase):
    agendas: list[AgendaBase] | None = Field(default_factory=list, description="회의 안건 목록")


