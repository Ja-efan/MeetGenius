from fastapi import FastAPI, APIRouter, BackgroundTasks
import httpx # FastAPI에서 http 요청 처리 
import asyncio # 테스트용.
from dotenv import load_dotenv
from core.embedding_utils import get_tokenizer, get_embedding_model
from core import rag, llm_utils
import torch
import gc


load_dotenv()

# 장고 url 
django_url = load_dotenv('DJANGO_URL')

router = APIRouter(
    prefix="/api/stt",
)

# STT 실행 상태 확인
is_listening = False

# RAG 트리거 
trigger_keywords = ["젯슨", "젯슨아"]

"""
send_data_to_django()

# Django 서버로 데이터 전송 함수 (비동기 HTTPX 요청) 
# -> 일반 회의 내용인지 RAG 응답인지 구분하면 좋을 것 같음. 
# -> data 파라미터에 대한 값으로 json을 받고, json을 넘겨주는게 어떤지
# ex. {"text": "일반 회의 내용"} / {"question": "RAG 질문 내용"} / {"response": "RAG 생성 응답답" } 
"""
async def send_data_to_django(data):
    # httpx.AsyncClient : httpx 비동기 버전..
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(django_url, json={"content":data})
            print(f'sent data:{data}')
        except Exception as e:
            print(f'error sending data : {e}')


# 음성 인식 & Django로 전송하는 함수 ( 백그라운드 실행 )
async def listen_and_recognize(app):
    global is_listening

    # mic 선언

    while is_listening:
        '''
            mic 통해 읽고, STT 진행 및 회의 도중 RAG
        '''
        print(f"STT is running ... (waiting for audio input)")
        
        transcript = '안녕하시렵니까. 기록중이렵니까'  # 오디오 스트림을 받아야 함 (찬호님 부탁해요)
        
        # STT 완료된 데이터 Django로 전송
        await send_data_to_django(transcript)

        if any(keyword in transcript for keyword in trigger_keywords):
            print(f"Trigger keyword in: {transcript}")

            # RAG를 통해 응답 생성
            answer = await rag.process_query(app, transcript)
            # 응답 출력
            print(f"RAG Response: {answer}")
            await send_data_to_django(answer)
        
        await asyncio.sleep(0) # CPU 과하게 점유 방지.


def init_app(app: FastAPI):

    # STT 시작 엔드포인트
    @router.get("/start/")
    async def start_voice_dectection(background_tasks: BackgroundTasks):
        """
            회의 시작 -> 모델 로드 -> STT 시작.
        """
        global is_listening 

        # 이미 STT가 진행중이라면면
        if is_listening:    
            return {"message": "STT is already running"}


        # STT 모델 로드 확인 및 로드 
        if not hasattr(app.state, "stt_model"):
            llm_utils.load_stt_model(app=app)

        # Embedding 모델 로드 확인 및 로드 
        if not hasattr(app.state, "embedding_model"):
            llm_utils.load_embedding_model(app=app)
        
        # LLM 로드 확인 및 로드 
        if not hasattr(app.state, "llm"):
            llm_utils.load_llm_model(app=app)

        print(f"All models loaded successfully.")

        
        is_listening = True
        background_tasks.add_task(listen_and_recognize, app) # STT 백그라운드 실행 -> 함수 결과가 아닌 함수 자체를 넘겨야 함
        return {"message":"Meeting started, models loaded, STT running."}


    # STT 종료 엔드포인트
    @router.get("/stop/")
    async def stop_voice_detection():
        global is_listening
        is_listening = False

        try:
            if hasattr(app.state, "stt_model"):
                del app.state.stt_model
            if hasattr(app.state, "embedding_model"):
                del app.state.embedding_model
            if hasattr(app.state, "llm"):
                del app.state.llm
            
             # ✅ 강제 Garbage Collection 실행 (CPU 메모리 해제)
            gc.collect()

            if torch.cuda.is_available():
                print(f"CUDA: {torch.cuda.is_available()}")
                torch.cuda.empty_cache()

            print("All models unloaded successfully.")
            
            return {"message": "Meeting ended, models unloaded, STT stopped."}
        
        except Exception as e:
            return {"error": f"Failed to unload models: {str(e)}"}

        
