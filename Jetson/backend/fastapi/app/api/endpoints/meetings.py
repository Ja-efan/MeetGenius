"""
회의 관련 엔드포인트
"""

from fastapi import FastAPI, APIRouter, BackgroundTasks, HTTPException, Depends, status
import httpx 
import os 
import asyncio
from dotenv import load_dotenv
import json
import logging
from core import rag, llm_utils, chromadb_utils 
from dependencies import get_app_state
from models.meetings import AgendaBase, AgendaList, PrepareMeetingOut, NextAgendaOut, Message

router = APIRouter(
    prefix="/api/v1/meetings",
)

load_dotenv()


DJANGO_URL = os.getenv('DJANGO_URL') # 장고 url 

######################################################### 로깅 설정 #########################################################
logger = logging.getLogger(__name__)  # 현재 모듈에 대한 로거 인스턴스 생성 
logger.setLevel(logging.INFO)  # 로그 레벨 설정 
handler = logging.StreamHandler()  # 콘솔에 로그 출력 
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # 로그 형식 설정 
handler.setFormatter(formatter)  # 형식 설정 적용 
logger.addHandler(handler)  # 로거에 핸들러 추가 
###########################################################################################################################    


trigger_keywords = ["젯슨", "젯슨아"]  # RAG 트리거

async def send_message(message: Message):  
    """메시지 전송 함수"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{DJANGO_URL}", json=message.model_dump())
            logger.info(f"Message sent to Django: {message}")
            logger.info(f"Response: {response.json()}")
        except Exception as e:
            logger.exception(f"Exception occured in send_message.\n{str(e)}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                                detail=f"메시지 전송 중 오류가 발생했습니다.")

async def stt_task(app_state: FastAPI.state):
    """STT 백그라운드 작업"""

    # 마이크 선언 
    while app_state.stt_running:
        """
            mic를 통해 듣고(읽고), STT 진행 및 회의 도중 RAG 
        """
        print(f"STT is runnign ... (waiting for audio input)")

        transcript = "음성인식 테스트 중 입니다."  # 실제 음성인식 로직 

        # 트리거 키워드 확인 
        if any(keyword in transcript for keyword in trigger_keywords):
            logger.info(f"Trigger keyword detected: {transcript}")
            
            # 사용자 질문 전송 
            await send_message(Message(type="query", content=transcript))
            
            # RAG 질문 응답 
            answer = await rag.process_query(query=transcript)
            message = Message(type="answer", content=answer)
            # 응답 전송 
            await send_message(message)

        # 트리거 키워드 확인 안된 경우 
        else:
            logger.info(f"No trigger keyword detected: {transcript}")
            await send_message(Message(type="plain", content=transcript))

        await asyncio.sleep(10)


@router.post("/{meeting_id}/prepare",status_code=status.HTTP_200_OK)
async def prepare_meeting(meeting_info: AgendaList, meeting_id: str, 
                          background_tasks: BackgroundTasks, app_state: FastAPI = Depends(get_app_state)):
    """회의 준비 엔드포인트

    Args:
        meeting_info (AgendaList): 회의 정보
        meeting_id (str): 회의 id
        app (FastAPI, optional): 앱 상태. Defaults to Depends().

    Raises:
        HTTPException: 예외 발생 시 예외 처리 

    Returns:
        PrepareMeetingOut: 회의 준비 완료 모델
    """
    try:
        # 필수 키 존재 여부 확인 
        if "project_id" not in meeting_info:
            msg = "Missing 'project_id' in meeting_info"
            logger.error(msg)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)

        # 프로젝트 관련 collection 생성 및 app.state에 저장
        app_state.project_collection = chromadb_utils.get_project_collection(project_id=meeting_info["project_id"],
                                                                              app_state=app_state)  
        
        # STT, Emb, RAG 모델 로드 
        llm_utils.load_stt_model(app_state=app_state)
        llm_utils.load_embedding_model(app_state=app_state) 
        llm_utils.load_rag_model(app_state=app_state)
        
        # chromadb 및 모델 로드 완료 시 회의 준비 완료 처리 
        app_state.is_meeting_ready = True   

        app_state.stt_running = True # STT 실행 상태 업데이트 

        # 백그라운드 작업 시작 
        background_tasks.add_task(stt_task, app_state=app_state) 

        logger.info(f"Meeting {meeting_id} preparation completed.")
        return PrepareMeetingOut(result=app_state.is_meeting_ready, message="회의 준비 완료")

    except HTTPException as he:
        raise he
        
    except Exception as e:
        logger.exception(f"Exception occured in prepare_meeting.\n{str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                            detail=f"회의 준비 중 오류가 발생했습니다.")


@router.post("/{meeting_id}/next-agenda", status_code=status.HTTP_200_OK)
async def next_agenda(agenda: AgendaBase, app_state: FastAPI = Depends(get_app_state)):
    """회의 시작 / 다음 안건 엔드포인트

    Args:
        agenda (agendas.AgendaBase): 안건 정보
        app (FastAPI, optional): 앱 상태. Defaults to Depends().

    Raises:
        HTTPException: 예외 발생 시 예외 처리 

    Returns:
        NextAgendaOut: 회의 시작 / 다음 안건 엔드포인트 응답 
    """
    try:
        # 필요한 상태 값이 없으면 초기화 
        if not hasattr(app_state.state, "stt_running"):
            app_state.state.stt_running = False  # STT 실행 상태 초기화 
        if not hasattr(app_state.state, "agenda_docs"):
            app_state.state.agenda_docs = {}  # 안건 문서 저장 초기화 

        
        agenda_docs = app_state.agenda_docs

        # <회의 시작> : STT가 아직 실행되지 않은 경우 
        if not app_state.stt_running:
            # STT 시작 
            app_state.stt_running = True
            
            # 현재 안건 id 에 해당하는 관련 문서 반환 
            docs = agenda_docs.get(agenda.agenda_id, [])  # 없으면 빈 리스트 반환 (KeyError 방지)

            logger.info(f"Start STT for meeting agenda '{agenda.agenda_id}'")  # STT 시작 로그 출력 
            return NextAgendaOut(stt_running=app_state.stt_running, agenda_docs=docs)
        
        # <다음 안건 / 안건 추가> : STT가 실행 중인 경우 
        else :
            docs = agenda_docs.get(agenda.agenda_id, [])
            # <다음 안건> : 기존 안건인 경우 (agenda_docs에 안건 id가 존재하는 경우)
            if docs:
                logger.info(f"Return existing agenda docs for agenda '{agenda.agenda_id}'")  # 기존 안건 로그 출력 
                return NextAgendaOut(stt_running=app_state.stt_running, agenda_docs=docs)
            
            # <안건 추가> : 신규 안건인 경우 (agenda_docs에 안건 id가 존재하지 않는 경우)
            else: 
                new_agenda_title = agenda.agenda_title  # 신규 안건 제목    
                collection = app_state.project_collection  # 프로젝트 컬렉션 
                docs = collection.get_agenda_docs(agenda=new_agenda_title, top_k=3)  # 유사 문서 검색 
                logger.info(f"New agenda processed: '{agenda.agenda_id}'")  # 신규 안건 처리 완료 로그 출력 
                return NextAgendaOut(stt_running=app_state.stt_running, agenda_docs=docs)
                
            
    except Exception as e:
        logger.exception(f"Exception occured in next_agenda.\n{str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                            detail=f"다음 안건 처리 중 오류가 발생했습니다.")
    

@router.post("/{meeting_id}/end")
async def end_meeting(meeting_id: str, user_info: dict, app_state: FastAPI = Depends(get_app_state)):
    # 회의 종료 로직
    # ...   
    return {"result": True}

@router.post("/{meeting_id}/summary")
async def summarize_meeting(meeting_id: str, summary_info: dict, app_state: FastAPI = Depends(get_app_state )):
    # 회의록 요약 로직
    # ...
    return {"result": True, "mom_summary": {}}  # 요약된 회의록 반환






