from fastapi import APIRouter, Depends, FastAPI, status
from app.services import rag
from app.dependencies import get_app

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
        # "온디바이스 환경에서 STT 모델을 경량화하기 위해 어떤 기술을 고려하고 있나요?",
        # "2차 기획 회의에서 음성 인식 성능 개선이 어느 정도 이뤄졌나요?",
        # "데이터베이스 인덱싱 방식은 어떻게 구성되어 있나요?",
        # "데이터 파티셔닝은 어떤 기준으로 적용하나요?",
        # "2월부터 3월까지 어떤 로드맵이 계획되어 있나요?",
        
        "프로젝트 예산 규모는 어느 정도이며, 각 팀에 배정된 예산은 얼마나 되나요?",
        "현재 STT 모델을 학습하기 위해 수집된 전체 음성 데이터 용량은 어느 정도인가요?",
        "회의록 자동 요약 모델로 어떤 자연어 처리 프레임워크(Transformers, RNN 등)를 사용하고 있나요?",
        "개발팀, 기획팀, QA팀 각각 몇 명으로 구성되어 있고, 팀별 세부 역할은 어떻게 나누어졌나요?",
        "실제 사용자의 피드백 결과(만족도)는 현재 어느 정도 수준인가요?"
    ]
    
    
    app.state.project_id = project_id
    answers = []
    for query in queries:
        answer = await rag.rag_process(app=app, query=query)
        answers.append(answer)
    return answers