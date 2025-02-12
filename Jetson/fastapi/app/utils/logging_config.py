import logging

# 로그 설정 함수
def setup_logger(name: str):
    """FastAPI에서 사용할 로거를 설정"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 중복 방지를 위해 기존 핸들러 제거
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s - %(asctime)s - %(name)s -  %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

# 애플리케이션에서 공통적으로 사용할 로거 생성
app_logger = setup_logger("app")
