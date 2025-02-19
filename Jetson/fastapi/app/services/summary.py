from app.schemes.meetings import AgendaSummary
import time
import re
import json
from fastapi import FastAPI, HTTPException
from app.utils import llm_utils, logging_config
from typing import List
from app.schemes.meetings import AgendaDetail

# 로깅 설정
logger = logging_config.app_logger

def extract_json_from_string(input_str: str) -> dict:
    """
    'json' 키워드 이후 첫 번째 JSON 객체를 추출하고 파싱하여 반환하는 함수
    """
    pattern = r"json\s*({.*?})"  # 가장 가까운 }까지 매칭
    match = re.search(pattern, input_str, re.DOTALL)

    if not match:
        logger.error("Cannot find JSON object in the input string.")
        return None  # JSON을 찾지 못했을 때 None 반환

    json_str = match.group(1).strip()

    try:
        parsed_data = json.loads(json_str)  # JSON 파싱
        return parsed_data
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 실패: {e}\n문제의 문자열: {json_str}")
        return None


async def summary_process(agenda_items: List[AgendaDetail], app: FastAPI):
    """안건별 요약 처리 함수
    Args:
        agenda_items (List[AgendaDetail]): 안건 제목, 내용 리스트
        app (FastAPI): FastAPI 애플리케이션 인스턴스 

    Returns:
        List[AgendaSummary]: 안건별 요약된 응답
    """
    
    logger.info(f"Agenda Items in summary_process: {agenda_items}")  # 안건 아이템 로그

    # Summary 모델 로드 
    if not hasattr(app.state, "summary_model"):
        logger.info(f"Summary model not found in app.state, loading...")
        app.state.summary_model = llm_utils.load_summary_model()
    
    # 필요한 모델 가져오기
    summary_model = app.state.summary_model
    
    # 모델 로드되었는지 확인 
    if not summary_model:
        logger.error("Summary is not loaded properly. Please check the model.")
        raise HTTPException(status_code=500, detail="Summary 시스템이 완전히 로드되지 않았습니다. 다시 시도해주세요.")
    
    summaries = [] # 안건별 요약 결과 저장 리스트
    
    try:
        for item in agenda_items:
            agenda_id = item.id
            agenda_title = item.title
            agenda_result = item.content

            if not agenda_result:
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
            {agenda_result} 

            #요약 내용
            """

            
            # 요약 모델 호출
            start_time = time.time()
            result = summary_model(
                prompt, 
                max_tokens=2000, 
                temperature=0.0
            )
            end_time = time.time()
            logger.info(f"Summary 응답 생성 시간: {end_time - start_time:.2f}초")
            answer = result["choices"][0]["text"]
            # answer = extract_json_from_string(answer)

            if answer is None:
                logger.error(f"agenda_id {agenda_id} 요약이 실패했습니다. 응답 내용: {answer}")
                continue  

            # **최종 결과 저장**
            summaries.append(AgendaSummary(
                id = agenda_id,
                title=agenda_title,
                summary=answer
            ))

        return summaries
    
    
    except Exception as e:
        logger.error(f"Summary process failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"요약 과정 중 오류 발생: {str(e)}")