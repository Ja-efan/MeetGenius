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
async def rag_test(query: str, project_id: int, app: FastAPI = Depends(get_app)):
    app.state.project_id = project_id
    answer = await rag.rag_process(app=app, query=query)
    return answer