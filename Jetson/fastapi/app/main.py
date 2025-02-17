from fastapi import FastAPI
from app.api.routes import meetings, projects, tests
import ctypes

app = FastAPI()

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
app.include_router(tests.router)  # 테스트 라우터 추가
