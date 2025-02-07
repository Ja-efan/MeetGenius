"""
회의 관련 엔드포인트
"""

from fastapi import FastAPI, APIRouter, BackgroundTasks, HTTPException, Depends
import httpx 
import os 
from dotenv import load_dotenv
import gc
import torch 
import json 

from core import rag, llm_utils, chromadb_utils, summary
from models.agendas import AgendaData, AgendaItem


load_dotenv()


DJANGO_URL = os.getenv('DJANGO_URL') # 장고 url 
STT_MODEL = os.getenv('STT_MODEL')
EMB_MODEL = os.getenv('EMB_MODEL')
RAG_MODEL = os.getenv('RAG_MODEL')

router = APIRouter(
    prefix="/api/v1/meetings",
)

stt_running = False # STT 실행 상태 ( 백그라운드 실행 )

trigger_keywords = ["젯슨", "젯슨아"]  # RAG 트리거 

"""
1. 회의전/회의준비(참가) [POST]  /api/v1/meetings/{meeting_id}/prepare

    - 회의 참가 버튼 클릭 시 호출 
    - 회의 정보 수신 
    - 회의 정보 저장 (?)
    - 프로젝트 관련 collection 생성
    - STT, Emb, RAG 모델 메모리 로드 
    - chromadb 및 모델 로드 완료 시 회의 준비 완료 처리 

    body:
    {
        "user_id": str,
        "project_id": str,
        "meeting_name": str,
        "meeting_date": str,
        "meeting_type": str,
        "meeting_participants": list[str],
        "meeting_agenda": list[str]
    }

    response:
    {
        "result": bool,
        "message": str
    }
"""

"""
2. 회의 시작 [POST]  /api/v1/meetings/{meeting_id}/start

    - 회의 시작 버튼 클릭 시 호출 
    - 회의 첫 안건 수신 (from DJango)
    - 안건과 관련된 문서 검색 
    - 검색된 문서 정보 전달 (DJango)
    

    body:
    {
        "user_id": str,
        "project_id": str,
        "agenda": str
    }

    response:
    {
        "result": bool,
        "agenda_docs": list[str]
    }
"""
"""
3. 회의중/다음안건 [POST]  /api/v1/meetings/{meeting_id}/next-agenda

    body:
    {
        "user_id": str,
        "project_id": str,
        "agenda": str
    }

    response:
    {
        "result": bool,
        "agenda_docs": list[str]
    }
"""

"""
4. 회의중/회의종료 [POST]  /api/v1/meetings/{meeting_id}/end

    body:
    {
        "user_id": str
    }

    response:
    {
        "result": bool
    }
"""

"""
5. 회의후/회의록요약 [POST]  /api/v1/meetings/{meeting_id}/summary

    body:
    {
        "user_id": str,
        "project_id": str,
        "mom_origin": json  # 안건 별로 수정된 회의록을 전달 받는다.
    }

    response:
    {
        "result": bool,
        "mom_summary": json  # 안건 별로 요약된 회의록을 전달한다.
    }
"""


@router.post("/{meeting_id}/prepare")
async def prepare_meeting(meeting_id: str, meeting_info: json, app: FastAPI = Depends()) -> json:
    """회의 준비(회의 참가) 엔드포인트 

        - 회의 참가 버튼 클릭 시 호출 
        - 회의 정보 수신 
        - 회의 정보 저장 (?)
        - 프로젝트 관련 collection 생성
        - STT, Emb, RAG 모델 메모리 로드 
        - chromadb 및 모델 로드 완료 시 회의 준비 완료 처리 
    
    Args:
        meeting_id (str): 회의 id
        meeting_info (json): 회의 정보
            {
                "user_id": str,
                "project_id": str,
                "meeting_name": str,
                "meeting_date": str,
                "meeting_type": str,
            }

    Returns:
        json: 회의 준비 완료 여부 및 메시지 
    """
    
    # 프로젝트 관련 collection 생성 및 app.state에 저장
    app.state.project_collection = chromadb_utils.get_project_collection(app=app,
                                                                        project_id=meeting_info["project_id"])
    
    # STT, Emb, RAG 모델 메모리 로드 
    app.state.stt_model = llm_utils.load_stt_model(app=app)
    app.state.embedding_model = llm_utils.load_embedding_model(app=app) 
    app.state.rag_model = llm_utils.load_rag_model(app=app)

    # chromadb 및 모델 로드 완료 시 회의 준비 완료 처리 
    app.state.is_meeting_ready = True

    return {"result": app.state.is_meeting_ready, "message": "회의 준비 완료"}

@router.post("/{meeting_id}/start")
async def start_meeting(meeting_id: str, agenda_info: dict, app: FastAPI = Depends()):
    # 회의 시작 로직
    # 첫 안건 수신 및 안건 관련 문서 검색
    # ...
    return {"result": True, "agenda_docs": []}  # 검색된 문서 목록 반환

@router.post("/{meeting_id}/next-agenda")
async def next_agenda(meeting_id: str, agenda_info: dict, app: FastAPI = Depends()):
    # 다음 안건 처리 로직
    # ...
    return {"result": True, "agenda_docs": []}  # 검색된 문서 목록 반환


@router.post("/{meeting_id}/end")
async def end_meeting(meeting_id: int, end_request: bool, app: FastAPI = Depends()):
    try:
        # 1. 백으로부터 종료 flag 받기
        if end_request:
            # 2. stt 종료 처리
            if stt_running:
                stt_running = False # 이 상태를 변수로 관리할지, app 상태로 관리할지 생각해보자 !!!

                # 3. 모델 unload
                llm_utils.unload_models(app)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                gc.collect()

            # 4. 종료 완료 flag 반환
            return {"meeting_id": meeting_id, "result": True}
        
        else:
            raise HTTPException(status_code=400, detail="회의 종료 요청 오류")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"회의 종료 중 오류 발생: {str(e)}")


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
        summaries = await summary.process_query(agenda_items)
        return {"meeting_id": meeting_id, "summaries": summaries}
    else:
        raise HTTPException(status_code=400, detail="안건이 없습니다.")
