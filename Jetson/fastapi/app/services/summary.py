import time
from app.schemes.meetings import AgendaSummary
from fastapi import FastAPI, HTTPException
from app.utils import llm_utils, logging_config
from typing import List
from app.schemes.meetings import AgendaDetail

# 로깅 설정
logger = logging_config.app_logger


async def summary_process(agenda_items: List[AgendaDetail], app: FastAPI):
    """안건별 요약 처리 함수
    Args:
        agenda_items (List[AgendaDetail]): 안건 제목, 내용 리스트
        app (FastAPI): FastAPI 애플리케이션 인스턴스 

    Returns:
        List[AgendaSummary]: 안건별 요약된 응답
    """
    logger.info(f"{len(agenda_items)} agendas in summary_process")
    logger.info(f"Agenda Items in summary_process: {agenda_items}")  # 안건 아이템 로그

    # Summary 모델 로드 
    if not hasattr(app.state, "summary_model"):
        logger.info(f"Summary model not found in app.state, loading...")
        app.state.summary_model = llm_utils.load_summary_model()
        app.state.summary_model = llm_utils.load_summary_model()
    
    # 모델이 정상적으로 로드되었는지 확인 
    if not app.state.summary_model:
        logger.error("Summary is not loaded properly. Please check the model.")
        raise HTTPException(status_code=500, detail="Summary 시스템이 완전히 로드되지 않았습니다. 다시 시도해주세요.")
    
    summaries = [] # 안건별 요약 결과 저장 리스트
    
    try:
        for i, item in enumerate(agenda_items):
            logger.info(f"Processing agenda {i+1} of {len(agenda_items)}")
            agenda_id = item.id
            agenda_title = item.title
            agenda_content = item.content

            if not agenda_content:
                continue
            
            prompt = f"""
            당신은 회의 내용을 간결히 정리하는 AI 어시스턴트입니다. 다음 지침을 지켜 요약하세요:

            [지침]
            1. 모든 주요 논의를 포함하세요. (기술 비교, 일정, 리스크 등)
            2. 여러 대안이 있다면, 각 장단점을 모두 포함하세요.
            3. 문서에 등장하는 질문과 관련된 수치(정확도, 속도, % 향상, 시간 등)는 누락 없이 그대로 답변에 포함하세요.
            4. 문서에 언급되지 않은 정보(예: 구체적 팀 인원 수, 예산 등)는 추측하거나 임의로 생성하지 마세요.
            5. 요약 내용은 간결하고 핵심 정보만 담아야 합니다. 불필요하게 길어지지 않도록 주의하세요.
            
            #안건 제목
            {agenda_title}

            #안건 내용
            {agenda_content} 

            #요약 내용
            """
            # 요약 모델 호출
            start_time = time.time()
            result = app.state.summary_model(
                prompt, 
                max_tokens=2000, 
                temperature=0.0
            )
            end_time = time.time()
            logger.info(f"회의록 요약 시간: {end_time - start_time:.2f}초")
            answer = result["choices"][0]["text"]

            # 모델 답변 포매팅
            summaries.append(AgendaSummary(
                id = agenda_id,
                title=agenda_title,
                summary=answer
            ))
        
        logger.info(f"Summary process completed. {len(summaries)} summaries generated.")
        
        # 모델 언로드
        llm_utils.unload_models(app=app)
        
        return summaries
    
    
    except Exception as e:
        logger.error(f"Summary process failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"요약 과정 중 오류 발생: {str(e)}")