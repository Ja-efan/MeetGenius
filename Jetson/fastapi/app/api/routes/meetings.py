"""
    회의 관련 엔드포인트
"""

import os
import gc
import torch 
import json 
import httpx
import logging
import asyncio
from typing import Any

from fastapi import FastAPI, APIRouter, BackgroundTasks, HTTPException, Depends, status
from app.schemes.meetings import STTMessage, MeetingAgendas, Agenda, MeetingAgendaDetails, AgendaDetail
from app.schemes.responses import PrepareMeetingResponse, NextAgendaResponse, EndMeetingResponse, SummaryResponse
from app.dependencies import get_app, get_app_state
from app.services import rag, summary
from app.utils import llm_utils, chromadb_utils, logging_config
from dotenv import load_dotenv

# 로깅 설정
logger = logging_config.app_logger

load_dotenv()

DJANGO_URL = os.getenv('DJANGO_URL')  # 장고 stt url 
STT_MODEL = os.getenv('STT_MODEL')
EMB_MODEL = os.getenv('EMB_MODEL')
RAG_MODEL = os.getenv('RAG_MODEL')

router = APIRouter(
    prefix="/api/v1/meetings",
    tags=["meetings"]
)

def is_stt_running(app: FastAPI):
    """STT 실행 상태 확인"""
    return getattr(app.state, "stt_running", False)

def set_stt_running(app: FastAPI, status: bool):
    """STT 실행 상태 설정"""
    app.state.stt_running = status

trigger_keywords = ["젯슨", "젯슨아"]  # RAG 트리거


async def send_message(message: STTMessage):
    """Jetson Orin Nano(FastAPI) -> Web(Django) 메시지 전송 함수

    Args:
        message (STTMessage): 메시지 모델

    Raises:
        HTTPException: 예외 발생 시 예외 처리 
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{DJANGO_URL}", json=message.model_dump())
            logger.info(f"Message sent to Django: {message}")
            logger.info(f"Response: {response.json()}")
        except Exception as e:
            logger.exception(f"Exception occured in send_message.\n{str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail="메시지 전송 중 오류가 발생했습니다."
            )

async def stt_task(app: FastAPI):
    """STT 백그라운드 작업"""
    while app.state.stt_running:
        # 실제 음성인식 로직 부분 (예시)
        logger.info("STT is running ... (waiting for audio input)")
        
        # 음성인식 로직 부분
        transcript = "음성인식 테스트 중 입니다."  # 실제 로직 적용 필요

        # 트리거 키워드 확인
        if any(keyword in transcript for keyword in trigger_keywords):
            logger.info(f"Trigger keyword detected: {transcript}")
            # 사용자 질문 전송
            message = STTMessage(type="query", content=transcript)
            logger.info(f"Message sent to Django: {message}")
            await send_message(message)
            # RAG 질문 응답
            answer = await rag.rag_process(app=app, query=transcript)
            message = STTMessage(type="rag", content=answer)
            await send_message(message)
        else:
            logger.info(f"No trigger keyword detected: {transcript}")
            message = STTMessage(type="plain", content=transcript)
            logger.info(f"Message sent to Django: {message}")
            await send_message(message)

        await asyncio.sleep(10)


@router.post("/{meeting_id}/prepare", status_code=status.HTTP_200_OK)
async def prepare_meeting(
    meeting_info: MeetingAgendas, 
    meeting_id: str, 
    background_tasks: BackgroundTasks, 
    app: FastAPI = Depends(get_app)
):
    """회의 준비 엔드포인트

    Args:
        meeting_info (MeetingAgendas): 회의 정보 (프로젝트 ID 및 안건 목록)
        meeting_id (str): 회의 ID
        background_tasks (BackgroundTasks): 백그라운드 작업 관리 객체
        app_state (Any): 앱 상태

    Returns:
        PrepareMeetingResponse: 회의 준비 완료 모델
    """
    try:
        # 필수 키 존재 여부 확인
        if not meeting_info.project_id:
            msg = "Missing 'project_id' in meeting_info"
            logger.error(msg)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)

        # 현재 회의 프로젝트 ID 저장
        app.state.project_id = meeting_info.project_id 
        logger.info(f"Project ID: {app.state.project_id}")
        
        # 프로젝트 관련 collection 생성 및 app_state에 저장
        app.state.project_collection = chromadb_utils.ProjectCollection(
            project_id=meeting_info.project_id,
            app=app
        )
        
        app.state.stt_model = llm_utils.load_stt_model(app=app)
        app.state.embedding_model = llm_utils.load_embedding_model(app=app)
        app.state.rag_model = llm_utils.load_rag_model(app=app)
        
        # chromadb 및 모델 로드 완료 시 회의 준비 완료 처리
        app.state.is_meeting_ready = True
        logger.info(f"is_meeting_ready: {app.state.is_meeting_ready}")
        app.state.stt_running = True  # STT 실행 상태 업데이트
        logger.info(f"stt_running: {app.state.stt_running}")
        
        # 안건 관련 문서 상태 초기화 및 저장
        app.state.agenda_docs = {}
        logger.info(f"agenda_docs: {app.state.agenda_docs}")
        # meeting_info.agendas 로 변경 (기존 agenda_list → agendas)
        app.state.agenda_list = meeting_info.agendas
        logger.info(f"agenda_list: {app.state.agenda_list}")
        for agenda in meeting_info.agendas:
            # 각 안건의 식별자와 제목은 각각 agenda.id, agenda.title 로 접근
            app.state.agenda_docs[agenda.id] = app.state.project_collection.get_agenda_docs(
                agenda=agenda.title, top_k=3
            )

        # 백그라운드 작업 시작
        # background_tasks.add_task(stt_task, app=app)  # test/dev/jetson 주석 처리

        logger.info(f"Meeting '{meeting_id}' preparation completed.")
        logger.info(f"Agenda docs: {app.state.agenda_docs}")
        
        return PrepareMeetingResponse(result=app.state.is_meeting_ready, message="회의 준비 완료")

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception(f"Exception occured in prepare_meeting.\n{str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="회의 준비 중 오류가 발생했습니다."
        )


@router.post("/{meeting_id}/next-agenda", status_code=status.HTTP_200_OK)
async def next_agenda(agenda: Agenda, app: FastAPI = Depends(get_app)):
    """회의 시작 / 다음 안건 엔드포인트

    Args:
        agenda (Agenda): 안건 정보
        app (FastAPI): 앱 상태

    Returns:
        NextAgendaResponse: 회의 시작 또는 다음 안건 응답
    """
    try:
        # 필요한 상태 값 초기화
        if not hasattr(app.state, "stt_running"):
            app.state.stt_running = False
        if not hasattr(app.state, "agenda_docs"):
            app.state.agenda_docs = {}

        agenda_docs = app.state.agenda_docs

        # <회의 시작>: STT가 아직 실행되지 않은 경우
        if not app.state.stt_running:
            app.state.stt_running = True
            # 현재 안건 id에 해당하는 관련 문서 반환
            docs = agenda_docs.get(agenda.id, [])
            logger.info(f"Start STT for meeting agenda '{agenda.id}'")
            return NextAgendaResponse(stt_running=app.state.stt_running, agenda_docs=docs)
        else:
            docs = agenda_docs.get(agenda.id, [])
            # <다음 안건>: 기존 안건인 경우
            if docs:
                logger.info(f"Processing existing agenda '{agenda.id}'")
                logger.info(f"Documents for agenda '{agenda.id}': {docs}")
                return NextAgendaResponse(stt_running=app.state.stt_running, agenda_docs=docs)
            # <안건 추가>: 신규 안건인 경우
            else:
                logger.info(f"Processing new agenda '{agenda.id}'")
                new_agenda_title = agenda.title  # 신규 안건 제목
                docs = app.state.project_collection.get_agenda_docs(agenda=new_agenda_title, top_k=3)
                logger.info(f"Documents for agenda '{agenda.id}': {docs}")
                return NextAgendaResponse(stt_running=app.state.stt_running, agenda_docs=docs)

    except Exception as e:
        logger.exception(f"Exception occured in next_agenda.\n{str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="다음 안건 처리 중 오류가 발생했습니다."
        )


@router.post("/{meeting_id}/end", status_code=status.HTTP_200_OK)
async def end_meeting(meeting_id: int, app: FastAPI = Depends(get_app)):
    """회의 종료 엔드포인트
    STT 중지, 관련 모델 언로드, 앱 상태에서 필요없는 것들 삭제, 메모리 정리 후 회의 종료 처리
    
    Args:
        meeting_id (int): 회의 ID
        app (FastAPI, optional): FastAPI 앱 인스턴스. Defaults to Depends(get_app).

    Raises:
        HTTPException: 예외 발생 시 예외 처리 

    Returns:
        EndMeetingResponse: 회의 종료 응답
    """
    try:
        # STT 종료 처리
        if hasattr(app.state, "stt_running") and is_stt_running(app):
            set_stt_running(app, False)

            # 관련 모델 언로드: app.state 에 저장된 모델들에 대한 참조 삭제 
            llm_utils.unload_models(app=app)
            
            # 추가로 필요 없는 상태 값들도 삭제 
            for attr in ["project_collection", "is_meeting_ready", "agenda_docs", "agenda_list"]:
                if hasattr(app.state, attr):
                    logger.info(f"🔄 [INFO] Deleting attribute: {attr}")
                    delattr(app.state, attr)

            # 메모리 정리
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.ipc_collect()  # IPC 캐시 정리 
                torch.cuda.empty_cache()  # VRAM  메모리 캐시 정리 
                
            logger.info("Memory cleaned up.")
            return EndMeetingResponse(meeting_id=meeting_id, stt_running=False)
        
        else:
            logger.info("STT is not running.")
            raise HTTPException(
                status_code=400,
                detail="STT가 실행되지 않았습니다."
            )

    except Exception as e:
        logger.exception(f"Exception occured in end_meeting.\n{str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"회의 종료 중 오류 발생: {str(e)}"
        )
    

@router.post("/{meeting_id}/summary", status_code=status.HTTP_200_OK)
async def summarize_meetings(
    meeting_id: int, 
    item: MeetingAgendaDetails, 
    app_state: Any = Depends(get_app_state)
):
    """회의록 요약 함수

    Args:
        meeting_id (int): 회의 ID
        item (MeetingAgendaDetails): 안건 및 안건 상세 내용 (각 안건의 제목과 내용)
        app_state (Any): 앱 상태

    Returns:
        SummaryResponse: 안건별 요약 내용 반환
    """
    if item.agendas:
        # 각 안건 정보를 dict로 변환하여 요약 처리 함수에 전달
        agenda_items = [
            AgendaDetail(id=agenda.id, title=agenda.title, content=agenda.content) 
            for agenda in item.agendas
        ]        
        summaries = await summary.process_query(agenda_items, app_state)
        # 요약 결과는 각 안건에 대해 "title", "original_content", "summary" 형태로 구성
        return SummaryResponse(meeting_id=meeting_id, summary=summaries)
    else:
        raise HTTPException(status_code=400, detail="안건이 없습니다.")

@router.post("/rag-test", status_code=status.HTTP_200_OK)
async def rag_test(query: str, app: FastAPI = Depends(get_app)):
    answer = await rag.rag_process(app=app, query=query)
    return answer