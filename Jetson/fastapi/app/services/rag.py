"""
    RAG 관련 함수 모듈 
"""
import time
import re
import json
from fastapi import FastAPI, HTTPException
from app.utils import chromadb_utils, llm_utils, logging_config

# 로깅 설정
logger = logging_config.app_logger

def extract_json_from_string(input_str: str) -> dict:
    """
    주어진 문자열에서 'json' 키워드 뒤에 있는 JSON 객체를 추출하여 파싱한 후,
    Python의 dict 객체로 반환합니다.
    
    매개변수:
        input_str (str): 파싱할 문자열.
        
    반환값:
        dict: 파싱된 JSON 데이터.
        
    예외:
        ValueError: 문자열 내에서 JSON 객체를 찾지 못하거나 JSON 파싱에 실패한 경우.
    """
    # 'json' 키워드 이후에 나오는 JSON 객체를 찾기 위한 정규표현식.
    # DOTALL 옵션을 사용하여 개행 문자도 포함하도록 합니다.
    pattern = r"json\s*(\{.*\})"
    match = re.search(pattern, input_str, re.DOTALL)
    
    if not match:
        raise ValueError("문자열에서 JSON 객체를 찾을 수 없습니다.")
    
    json_str = match.group(1)
    
    try:
        parsed_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 파싱에 실패했습니다: {e}")
    
    return parsed_data


async def rag_process(query: str, app: FastAPI):
    """사용자의 질문을 처리하여 RAG 기반의 답변을 생성하는 함수 

    Args:
        query (str): 사용자 질문
        app (FastAPI): FastAPI 애플리케이션 인스턴스 
    
    Returns:
        str: RAG 응답 (검색된 문서 기반 생성)
    """
    
    # 필요한 모델과 DB가 로드되었는지 확인 
    if not hasattr(app.state, "project_collection"):
        logger.info(f"Project collection not found in app.state, loading ... (project_id: '{app.state.project_id}')")
        app.state.project_collection = chromadb_utils.ProjectCollection(
            project_id=str(app.state.project_id),
            app=app
        )
        logger.info(f"Project collection loaded successfully!")
    elif app.state.project_collection.project_id != str(app.state.project_id):
        logger.info(f"Project collection project_id mismatch, reloading ... (project_id: '{app.state.project_id}')")
        app.state.project_collection = chromadb_utils.ProjectCollection(
            project_id=str(app.state.project_id),
            app=app
        )
        logger.info(f"Project collection reloaded successfully!")
        
    # 임베딩 모델 로드 
    if not hasattr(app.state, "embedding_model"):
        logger.info(f"Embedding model not found in app.state, loading...")
        app.state.embedding_model = llm_utils.load_embedding_model(app=app)
        
    # RAG 모델 로드 
    if not hasattr(app.state, "rag_model"):
        logger.info(f"RAG model not found in app.state, loading...")
        app.state.rag_model = llm_utils.load_rag_model(app=app)
        
    # 필요한 모델과 DB 가져오기 
    project_collection = app.state.project_collection
    embedding_model = app.state.embedding_model
    rag_model = app.state.rag_model
    
    # 모델과 DB가 로드되었는지 확인 
    if not project_collection or not embedding_model or not rag_model:
        logger.error("RAG is not loaded properly. Please check the models and database.")
        raise HTTPException(status_code=500, detail="RAG 시스템이 완전히 로드되지 않았습니다. 다시 시도해주세요.")
    
    # KoE5 모델 임베딩 프로세스
    formatted_query = [f"query: {query}"]

    # 사용자의 질문을 벡터로 변환 
    query_embedding = embedding_model.encode(formatted_query)

    # ChroaaDB에서 관련 문서 검색 (Top-K 검색)
    search_results = project_collection.search_documents(query_embedding=query_embedding, top_k=1)
    retrieved_content = search_results["documents"][0]
    retrieved_doc_ids = [int(id) for id in search_results["ids"][0]]
    
    
    # 프롬프트 구성 (EXAONE3.5)
    prompt = f"""
당신은 LG AI Research의 EXAONE 모델입니다. 현재 'Ondevice AI Meeting Copilot' 프로젝트의 일원으로, 제공된 문서에 기반해 정확하고 간결한 한글 답변을 작성해야 합니다.

[규칙]
1. 문서에 명시된 수치 외 임의 추정 수치는 사용하지 않습니다.
2. 문서에 없는 정보가 필요하면 '추가 정보가 필요합니다'와 "문서에서 구체적 수치는 제공되지 않았습니다"를 함께 기재하십시오.
3. 답변은 반드시 아래 세 필드로 구성합니다.
   - "답변 요약": 문서의 핵심을 간략히 요약.
   - "주요 수치 정보": 문서에 수치가 있으면 그대로 기재, 없으면 "추가 정보가 필요합니다. 문서에서 구체적 수치는 제공되지 않았습니다."라고 작성.
   - "세부 내용": 수치의 역할/효과 등 보충 설명(수치가 있다면 "구체적 수치는 제공되지 않았다"라는 문구는 사용하지 않습니다).
4. 출력은 오직 하나의 문자열이어야 하며, 반드시 **"json "**으로 시작한 후 바로 JSON 객체가 이어져야 합니다.  
   마크다운, 코드 블록, 추가 텍스트, 불필요한 줄바꿈이나 공백은 절대 포함하지 마십시오.

사용자 질문: {query}

관련 문서:
{retrieved_content}

출력 예시:
json {{"답변 요약": "<핵심 요약>", "주요 수치 정보": "<수치 정보 또는 '추가 정보가 필요합니다. 문서에서 구체적 수치는 제공되지 않았습니다.'>", "세부 내용": "<보충 설명>"}} 
"""





    start_time = time.time()
    result = rag_model(
        prompt,
        max_tokens=1000,  # 답변 생성 최대 토큰 수 제한 
        temperature=0.3  # 답변 생성 온도 조절 
    )
    end_time = time.time()
    logger.info(f"RAG 응답 생성 시간: {end_time - start_time:.2f}초")

    # 답변 형식 정리 
    answer = result["choices"][0]["text"]
    answer_json = extract_json_from_string(answer)

    return {"answer": answer_json, "doc_ids": retrieved_doc_ids}

