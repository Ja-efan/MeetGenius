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
    
    logger.info("Agenda Items in summary_process:", agenda_items)  # 안건 아이템 로그

    # Summary 모델 로드 
    if not hasattr(app.state, "summary_model"):
        logger.info(f"Summary model not found in app.state, loading...")
        app.state.summary_model = llm_utils.load_summary_model(app=app)
    
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
            당신은 회의 내용을 요약하는 AI 어시스턴트입니다.
            각 회의에는 여러 개의 안건이 있으며, 안건별로 중요한 내용을 빠짐없이 요약해야 합니다.
            각 안건은 agenda_id로 식별됩니다.

            [출력 규칙]
            1. 반드시 JSON 형식으로 출력해야 합니다.
            2. 출력은 **"json "**으로 시작해야 하며, 뒤이어 바로 JSON 객체를 출력하세요.
            3. **그 외의 설명, 마크다운, 코드 블록, 추가 텍스트, 불필요한 줄바꿈, 공백을 포함하지 마세요.**
            4. JSON 형식은 다음과 같습니다.

            [출력 예시]
            json {{"안건 id": {agenda_id}, "안건 제목": "{agenda_title}", "요약 내용": "<안건 요약>"}}

            [요약 지침]
            1. 모든 논의된 주요 내용을 포함하여 요약하세요. (기술 비교, 협업 도구 활용, 일정 조정, 리스크 분석 등)
            2. 회의에서 논의된 "결정 사항"을 강조하여 명확하게 기술하세요. (예: "백엔드는 FastAPI로 결정")
            3. 여러 대안이 논의된 경우, 각각의 장단점을 간략히 포함하세요. (예: "Whisper는 성능 우수, Azure STT는 비용 절감 가능")
            4. 구체적인 수치(성능 개선율, 일정, 기술, 목표 KPI 등)가 포함된 경우 그대로 유지하세요.
            5. 출력은 반드시 하나의 문자열이어야 하며, **"json "**으로 시작한 후 바로 JSON 객체가 이어져야 합니다.
            마크다운, 코드 블록, 추가 텍스트, 불필요한 줄바꿈이나 공백은 포함하지 마세요.

            [안건 정보]
            안건 ID: {agenda_id}
            안건 제목: {agenda_title}
            안건 내용:
            {agenda_result}

            위 내용을 요약하고 JSON 형식으로 출력하세요.
            """

            
            # 요약 모델 호출
            start_time = time.time()
            result = summary_model(
                prompt, 
                max_tokens=2000, 
                temperature=0.3
            )
            end_time = time.time()
            logger.info(f"Summary 응답 생성 시간: {end_time - start_time:.2f}초")
            answer = result["choices"][0]["text"]
            summary_data = extract_json_from_string(answer)

            if summary_data is None:
                logger.error(f"agenda_id {agenda_id} 요약이 실패했습니다. 응답 내용: {answer}")
                continue  

            # **최종 결과 저장**
            summaries.append(AgendaSummary(
                title=agenda_title,
                original_content=agenda_result,
                summary=summary_data
            ))
        
        # 모델 언로드
        llm_utils.unload_models(app=app)
        
        return summaries
    
    
    except Exception as e:
        logger.error(f"Summary process failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"요약 과정 중 오류 발생: {str(e)}")