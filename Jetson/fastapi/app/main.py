from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Depends
from app.api.routes import meetings, projects, tests
import ctypes

# 라이브 시연을 위한 코드 
from app.utils import llm_utils
app = FastAPI()
from app.dependencies import get_app

ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(
    None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p
)
def py_error_handler(filename, line, function, err, fmt): # 에러 메시지를 무시함
    pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
try:
    asound = ctypes.cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)
except Exception:
    pass

@app.get("/")
def hello():
    return {"message": "Hello!"}

app.include_router(meetings.router)
app.include_router(projects.router)
app.include_router(tests.router) 

@app.on_event("startup")
async def startup_event():
    app.state.embedding_model = llm_utils.load_embedding_model()
    app.state.rag_model = llm_utils.load_rag_model()
    # app.state.stt_model = await llm_utils.load_stt_model(app=app)