from fastapi import APIRouter, HTTPException, Request
from core.summary import process_query
from models.agendas import AgendaData, AgendaItem

router = APIRouter(
    prefix="/api/v1/meetings",
)


@router.get("/")
def test():
    return {"message":"reports page is running"}


# 회의ID(meeting_id)에 따른 회의록 수정 페이지(edit_document)
# 요약이 안건별로 되어야 하는데, 어떻게 안건별로 나눔..?
# 웹에서 가져올 때 안건 제목(agenda_title) + 안건 내용(agenda_result) 같이 가져오기.
# 안건 제목 인식될 때마다 하나의 안건이라는 뜻 => 이걸로 안건별로 나누어 요약해서 보내면 될듯?
# [정리] 요약 실행 -> 안건 제목 + 내용 가져오기 -> 내용만 가지고 요약 -> 보낼 때는 그 제목이랑 요약 결과 보내기
@router.post("/{meeting_id}/summary")
async def summarize_meetings(meeting_id: int, item: AgendaData)-> dict | HTTPException:
    """회의록 요약 함수

    Args:
        meeting_id (int): 회의 ID
        item (AgendaData): [안건 제목, 요약 내용]

    Returns:
        dict | HTTPException: 회의 ID, [안건 제목, 요약 내용] 반환 및 예외 처리
    """
    if item.items:
        agenda_items = [agenda.model_dump() for agenda in item.items]
        summaries = await process_query(agenda_items)
        return {"meeting_id": meeting_id, "summaries": summaries}
    else:
        raise HTTPException(status_code=400, detail="안건이 없습니다.")