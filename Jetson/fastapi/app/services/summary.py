"""
    요약 관련 함수 모듈 
"""

from typing import List, Dict
from fastapi import FastAPI, HTTPException, Depends
import torch
from app.utils.llm_utils import load_summary_model
from typing import Any
from app.schemes.meetings import AgendaSummary, AgendaDetail
from app.utils import logging_config

# 로깅 설정
logger = logging_config.app_logger


async def process_query(agenda_items: List[AgendaDetail], app_state: Any) -> List[Dict[str, str]]:
    """
    안건별로 요약을 수행하는 함수

    Args:
        agenda_items (List[AgendaDetail]): 안건 제목, 내용 리스트

    Returns:
        List[AgendaSummary]: 안건별 요약된 응답
    """
    logger.info("Agenda Items in process_query:", agenda_items) # 안건 아이템 로그

    summaries = []
    # 요약 모델이 제대로 로드되었는지 확인
    summary_model = load_summary_model(app_state)

    tokenizer = summary_model["tokenizer"]
    model = summary_model["model"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    try:
        for item in agenda_items:
            agenda_title = item.title
            agenda_result = item.content

            if not agenda_result:
                continue

            # 토큰화 및 요약
            inputs = tokenizer(agenda_result, return_tensors="pt", max_length=512, truncation=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.no_grad():
                summary_ids = model.generate(
                    inputs["input_ids"],
                    num_beams=4,
                    max_length=150,
                    early_stopping=True
                )

            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
  
            summaries.append(AgendaSummary(
                title=agenda_title,
                original_content=agenda_result,
                summary=summary
            ))


        return summaries

    except Exception as e:
        logger.error(f"Summary process failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"요약 과정 중 오류 발생: {str(e)}")
