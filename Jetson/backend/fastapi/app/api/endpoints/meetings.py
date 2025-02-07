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

from core import rag, llm_utils, chromadb_utils


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
async def end_meeting(meeting_id: str, user_info: dict, app: FastAPI = Depends()):
    # 회의 종료 로직
    # ...
    return {"result": True}

@router.post("/{meeting_id}/summary")
async def summarize_meeting(meeting_id: str, summary_info: dict, app: FastAPI = Depends()):
    # 회의록 요약 로직
    # ...
    return {"result": True, "mom_summary": {}}  # 요약된 회의록 반환






