"""
    RAG 관련 함수 모듈 
"""
import time
from fastapi import FastAPI, HTTPException
from app.utils import chromadb_utils, llm_utils, logging_config

# 로깅 설정
logger = logging_config.app_logger


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
        logger.info(f"Project collection not found in app.state, loading...")
        app.state.project_collection = chromadb_utils.ProjectCollection(
            project_id=app.state.project_id,
            app=app
        )
        logger.info(f"Project collection loaded successfully!")
        
    # 임베딩 모델 로드 
    if not hasattr(app.state, "embedding_model"):
        logger.info(f"Embedding model not found in app.state, loading...")
        app.state.embedding_model = llm_utils.load_embedding_model(app_state=app)
        
    # RAG 모델 로드 
    if not hasattr(app.state, "rag_model"):
        logger.info(f"RAG model not found in app.state, loading...")
        app.state.rag_model = llm_utils.load_rag_model(app_state=app)
        
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
    retrieved_doc_ids = search_results["ids"][0]
    
    # 프롬프트 구성 (EXAONE3.5)
    prompt = [

    ]
    
    start_time = time.time()
    result = rag_model(
        prompt,
        max_new_tokens=1000,  # 답변 생성 최대 토큰 수 제한 
        temperature=0.2  # 답변 생성 온도 조절 
    )
    end_time = time.time()
    logger.info(f"RAG 응답 생성 시간: {end_time - start_time:.2f}초")

    # 답변 형식 정리 
    answer = result["choices"][0]["text"]

    return {"answer": answer, "doc_ids": retrieved_doc_ids}

