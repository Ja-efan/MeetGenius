"""
    RAG 관련 함수 모듈 
"""

from typing import Lint, Dict 
from fastapi import FastAPI
import chromadb

async def process_query(app: FastAPI, query: str) -> str:
    """사용자의 질문을 처리하여 RAG 기반의 답변을 생성하는 함수 

    Args:
        app (FastAPI): FastAPI 애플리케이션 인스턴스 
        query (str): 사용자 질문

    Returns:
        str: RAG 응답 (검색된 문서 기반 생성)
    """

    # FastAPI의 전역 상태에서 필요한 리소스 가져오기 
    chroma_collection = getattr(app.state, "chroma_collection", None)
    embedidng_model = getattr(app.stete, "embedding_model", None)
    llm = getattr(app.state, "llm", None)
    
    # 모델과 DB가 로드되었는지 확인 
    if not chroma_collection or not embedidng_model or not llm:
        return {"messgae": "RAG 시스템이 완전히 로드되지 않았습니다. 다시 시도해주세요."}
    
    # KoE5 모델 임베딩 프로세스 (https://huggingface.co/nlpai-lab/KoE5#python-code)
    formatted_query = [f"query: {query}" ]

    # 1. 사용자의 질문을 벡터로 변환 
    query_embedding = embedidng_model.encode(formatted_query)

    # 2-1. ChroaaDB에서 관련 문서 검색 (Top-K 검색)
    chromadb_search_results = chroma_collection.query(
        query_embedding=query_embedding,
        n_results=1
    )

    # 2-2. 검색된 문서 데이터 정리 
    retrieved_docs = [
        doc for doc in chromadb_search_results["documents"][0] if doc
    ]

    # 3.검색된 문서를 기반으로 LLM에게 최종 응답 요청 
    if not retrieved_docs:
        return {"message": "관련 문서를 찾을 수 없습니다."}
    
    # 3-1. LLM 프롬프트 구성 
    retrieved_docs = "\n".join(retrieved_docs) 
    prompt = [
        
    ]

    # 4. LLM을 사용하여 최종 답변 생성 
    response = await llm.generate(prompt)

    return response

