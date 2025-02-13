"""
    요약 관련 함수 모듈 
"""

import time
import re
import json
from typing import List, Dict
from fastapi import FastAPI, HTTPException, Depends
import torch
from app.utils.llm_utils import load_summary_model
from typing import Any
from app.schemes.meetings import AgendaSummary, AgendaDetail
from app.utils import logging_config

# 로깅 설정
logger = logging_config.app_logger


async def process_query(agenda_items: List[AgendaDetail], app_state: Any) -> List[AgendaSummary]:
    """
    안건별로 요약을 수행하는 함수

    Args:
        agenda_items (List[AgendaDetail]): 안건 제목, 내용 리스트

    Returns:
        List[AgendaSummary]: 안건별 요약된 응답
    """
    logger.info(f"Agenda Items in process_query: {agenda_items}")

    summaries = []
    summary_model = load_summary_model(app_state)
    
    if not summary_model:
        raise HTTPException(status_code=500, detail="요약 모델 로드 실패")

    try:
        for item in agenda_items:
            agenda_title = item.title
            agenda_result = item.content

            if not agenda_result:
                logger.warning(f"Skipping empty agenda item: {agenda_title}")
                continue

            # 프롬프트 구성
            prompt = f"""
            [INST]
            아래의 회의 내용을 분석하여, 핵심 정보를 3문장 이내로 요약하세요.

            지침:
            1. 중요한 수치 및 기술 정보를 그대로 반영하고, 임의로 추정하지 마세요.
            2. 핵심 내용만 포함하여 간결하게 요약하세요.
            3. 최종 결과는 반드시 JSON 형식으로 출력하세요:
            {{"요약": "<요약된 내용>"}}

            회의 제목: {agenda_title}
            회의 내용:
            {agenda_result}
            [/INST]
            """


            start_time = time.time()
            result = summary_model(
                prompt,
                max_tokens=1000,
                temperature=0.3
            )
            end_time = time.time()
            logger.info(f"요약 시간: {end_time - start_time:.2f}초")

            summary_text = result["choices"][0]["text"].strip()
            
            # **최종 결과 저장**
            summaries.append(AgendaSummary(
                title=agenda_title,
                original_content=agenda_result,
                summary=summary_text
            ))

        return summaries

    except Exception as e:
        logger.error(f"Summary process failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"요약 과정 중 오류 발생: {str(e)}")
    