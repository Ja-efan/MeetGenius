import httpx
from fastapi import APIRouter, Depends, FastAPI, status, HTTPException
from app.services import rag, summary
from app.schemes.meetings import AgendaDetail
from app.dependencies import get_app
from app.utils.chromadb_utils import ProjectCollection, get_chromadb_client
from app.utils.llm_utils import load_embedding_model, load_rag_model, load_summary_model, unload_models, load_stt_model

router = APIRouter(
    prefix="/api/v1/tests",
    tags=["tests"]
)

@router.get("/")
def test():
    return {"message": "Test Endpoints"}

@router.post("/rag/{project_id}", status_code=status.HTTP_200_OK)
async def rag_test(project_id: int, app: FastAPI = Depends(get_app)):
    
    queries = [
        "온디바이스 환경에서 STT 모델을 경량화하기 위해 어떤 기술을 고려하고 있나요?",
        "2차 기획 회의에서 음성 인식 성능 개선이 어느 정도 이뤄졌나요?",
        "데이터베이스 인덱싱 방식은 어떻게 구성되어 있나요?",
        "데이터 파티셔닝은 어떤 기준으로 적용하나요?",
        "2월부터 3월까지 어떤 로드맵이 계획되어 있나요?",
        
        "프로젝트 예산 규모는 어느 정도이며, 각 팀에 배정된 예산은 얼마나 되나요?",
        "현재 STT 모델을 학습하기 위해 수집된 전체 음성 데이터 용량은 어느 정도인가요?",
        "회의록 자동 요약 모델로 어떤 자연어 처리 프레임워크(Transformers, RNN 등)를 사용하고 있나요?",
        "개발팀, 기획팀, QA팀 각각 몇 명으로 구성되어 있고, 팀별 세부 역할은 어떻게 나누어졌나요?",
        "실제 사용자의 피드백 결과(만족도)는 현재 어느 정도 수준인가요?"
    ]
    
    
    # app.state.project_id = project_id
    answers = []
    for query in queries:
        answer = await rag.rag_process(app=app, query=query, project_id=project_id)
        answers.append(answer)
    return answers





#==== 회의 요약 테스트 ===

DOCUMENTS_ENDPOINT = "http://localhost:8000/api/v1/projects/{project_id}/documents"  # 프로젝트 기반 문서 조회

@router.post("/summary/{meeting_id}", status_code=status.HTTP_200_OK)
async def test_summary(meeting_id: int, app: FastAPI = Depends(get_app)):
    """
    특정 meeting_id에 해당하는 문서를 조회하고 요약을 수행하는 테스트용 엔드포인트.
    """

    project_id = 1000  # 🔥 테스트할 프로젝트 ID (실제 환경에서는 요청 값 또는 설정에서 가져와야 함)

    # 프로젝트 문서 조회
    async with httpx.AsyncClient() as client:
        response = await client.get(DOCUMENTS_ENDPOINT.format(project_id=project_id))
    
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="문서 조회에 실패했습니다.")

    document_data = response.json()
    
    if "documents" not in document_data or "metadatas" not in document_data["documents"]:
        raise HTTPException(status_code=400, detail="올바른 문서 데이터가 아닙니다.")
    
    documents = document_data["documents"]["documents"]
    metadatas = document_data["documents"]["metadatas"]

    # meeting_id에 해당하는 문서 필터링
    agendas = []
    for idx, metadata in enumerate(metadatas):
        if metadata.get("meeting_id") == meeting_id:  # 특정 meeting_id에 해당하는 문서만 선택
            agendas.append(AgendaDetail(
                id=idx + 1,  # 안건 ID (임시)
                title=f"회의 안건 {idx + 1}",  # 안건 제목 (임시)
                content=documents[idx]  # 실제 회의 내용
            ))

    if not agendas:
        raise HTTPException(status_code=400, detail=f"회의 ID {meeting_id}에 해당하는 문서가 없습니다.")

    # 요약 수행
    summaries = await summary.summary_process(agendas, app)

    return {
        "meeting_id": meeting_id,
        "summaries": summaries
    }

@router.get("/{project_id}", status_code=status.HTTP_200_OK)
async def test_project_documents(project_id: int, app: FastAPI = Depends(get_app)):
    project_collection = ProjectCollection(project_id=project_id, app=app)
    if not project_collection.get_documents():
        return {"message": "No documents found"}
    else:
        return project_collection.get_documents()


@router.get("/projects", status_code=status.HTTP_200_OK)
async def test_project_list(app: FastAPI = Depends(get_app)):
    chromadb_client = get_chromadb_client()
    project_list = chromadb_client.list_collections()
    if not project_list:
        return {"message": "No collections found"}
    else:
        return project_list

@router.get("/stt", status_code=status.HTTP_200_OK)
async def test_load_stt_model(app: FastAPI = Depends(get_app)):
    app.state.stt_model= load_stt_model()
    return {"message": "STT model loaded successfully!"}


@router.get("/embedding", status_code=status.HTTP_200_OK)
async def test_load_embedding_model(app: FastAPI = Depends(get_app)):
    app.state.embedding_model = load_embedding_model()
    return {"message": "Embedding model loaded successfully!"}


@router.get("/rag", status_code=status.HTTP_200_OK)
async def test_load_rag_model(app: FastAPI = Depends(get_app)):
    app.state.rag_model = load_rag_model()
    return {"message": "RAG model loaded successfully!"}


@router.get("/summary", status_code=status.HTTP_200_OK)
async def test_load_summary_model(app: FastAPI = Depends(get_app)):
    app.state.summary_model = load_summary_model()
    return {"message": "Summary model loaded successfully!"}


@router.get("/unload", status_code=status.HTTP_200_OK)
async def test_unload_models(app: FastAPI = Depends(get_app)):
    unload_models(app)
    return {"message": "Models unloaded successfully!"}