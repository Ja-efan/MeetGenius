"""
    RAG 관련 함수 모듈 
"""
import time
import re
from fastapi import FastAPI, HTTPException
from app.utils import chromadb_utils, llm_utils, logging_config

# 로깅 설정
logger = logging_config.app_logger

def extract_answer(answer_str: str) -> str:
    """
    주어진 문자열에서 "답변:"과 "**" 등의 토큰을 제거하고,
    문장 끝(., ?, !) 뒤의 개행은 보존하며, 그 외의 개행은 공백으로 치환합니다.
    또한, 괄호, 작은/큰 따옴표, dot(.), comma(,), 느낌표(!), 물음표(?)를 제외한
    모든 특수문자를 제거합니다.
    
    :param answer_str: 원본 RAG모델 응답 문자열
    :return: 정제된 답변 내용
    """
    # 1. "답변:"과 "**" 제거
    text = answer_str.replace("답변:", "").replace("**", "")
    text = text.strip()
    
    # 2. 문장 끝(., ?, !) 뒤에 오는 개행은 보존하고, 그 외의 개행은 공백으로 치환
    text = re.sub(r'(?<![.?!])\n+', ' ', text)
    
    # 3. 허용 문자: 영문, 한글, 숫자, 공백, 괄호, 작은/큰 따옴표, dot(.), comma(,), 느낌표(!), 물음표(?)
    text = re.sub(r'[^0-9A-Za-z가-힣\s\(\)\'",.!?%]+', '', text)
    
    # 4. 각 줄별로 불필요한 공백 제거 (개행은 유지)
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.splitlines()]
    cleaned_text = "\n".join(line for line in lines if line)
    
    return cleaned_text

async def rag_process(query: str, app: FastAPI, project_id: int):
    """사용자의 질문을 처리하여 RAG 기반의 답변을 생성하는 함수 

    Args:
        query (str): 사용자 질문
        app (FastAPI): FastAPI 애플리케이션 인스턴스 
    
    Returns:
        str: RAG 응답 (검색된 문서 기반 생성)
    """
    
    # project_id가 int가 아니라 PJT-xx로 넘어올 수도 있음
    if isinstance(project_id, str):
        project_id = int(project_id.replace("PJT-", ""))
    
    # chromadb client 생성
    if not hasattr(app.state, "chromadb_client"):
        app.state.chromadb_client = chromadb_utils.get_chromadb_client()
        logger.info(f"Chromadb client created successfully!")
    
    # 필요한 DB가 로드되었는지 확인 
    """
    현재 app.state에 project_collection이 없는 경우 
    -> 새로운 ProjectCollection 인스턴스 생성 

    현재 app.state에 project_collection이 있는 경우 
    -> 기존 project_collection의 project_id와 현재 project_id가 다른 경우 
    -> 새로운 ProjectCollection 인스턴스 생성 
    """
    if not hasattr(app.state, "project_collection"):
        # logger.info(f"Project collection not found in app.state, loading ... (project_id: '{app.state.project_id}')")
        app.state.project_collection = chromadb_utils.ProjectCollection(
            client=app.state.chromadb_client,
            project_id=project_id,
            app=app
        )
        logger.info(f"Project collection loaded successfully!")
        
    elif app.state.project_collection.project_id != "PJT-" + str(project_id):
        # logger.info(f"Project collection project_id mismatch, reloading ... (project_id: '{app.state.project_id}')")
        app.state.project_collection = chromadb_utils.ProjectCollection(
            client=app.state.chromadb_client,
            project_id=project_id,
            app=app
        )
        logger.info(f"Project collection reloaded successfully!")
        
    # 임베딩 모델 로드 
    if not hasattr(app.state, "embedding_model"):
        logger.info(f"Embedding model not found in app.state, loading...")
        app.state.embedding_model = llm_utils.load_embedding_model()
        
    # RAG 모델 로드 
    if not hasattr(app.state, "rag_model"):
        logger.info(f"RAG model not found in app.state, loading...")
        app.state.rag_model = llm_utils.load_rag_model()
        
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
    # logger.info(f"search_results: {search_results}")
    
    print(f"search_results['metadatas']: {search_results['metadatas']}")
    retrieved_content = search_results["documents"][0]
    retrieved_doc_id = search_results["metadatas"][0][0]["document_id"]
    
    if not retrieved_doc_id:
        return {"answer": "문서에서 해당 내용을 찾을 수 없습니다.", "agenda_docs": []}
    
    # 프롬프트 구성 (EXAONE3.5)
    prompt = f"""
당신은 질의응답 작업을 하는 유용한 AI 비서입니다.
아래의 문서를 참고하여, 회의 중 나온 사용자의 질문에 대해 간결하게 답변하세요.

[지침]
1. 질문과 관련된 내용이 문서에 없으면, 반드시 "문서에서 해당 내용을 찾을 수 없습니다."라고 답변하세요.
2. 문서에 등장하는 질문과 관련된 수치(정확도, 속도, % 향상, 시간 등)는 누락 없이 그대로 답변에 포함하세요.
3. 문서에 언급되지 않은 정보(예: 구체적 팀 인원 수, 예산 등)는 추측하거나 임의로 생성하지 마세요.
4. 답변은 간결하고 핵심 정보만 담아야 합니다. 불필요하게 길어지지 않도록 주의하세요.

#문서
{retrieved_content}

#질문
{query}

#답변
"""

    start_time = time.time()
    result = rag_model(
        prompt,
        max_tokens=1000,  # 답변 생성 최대 토큰 수 제한 
        temperature=0.0
    )
    end_time = time.time()
    logger.info(f"RAG 응답 생성 시간: {end_time - start_time:.2f}초")

    logger.info(f"result: {result}")
    
    # 답변 형식 정리 
    answer = result["choices"][0]["text"]
    
    answer = extract_answer(answer)
    return {"answer": answer, "agenda_docs": [retrieved_doc_id]}

