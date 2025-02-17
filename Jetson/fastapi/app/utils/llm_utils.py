import os, gc, asyncio, torch 
from llama_cpp import Llama
from fastapi import FastAPI
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, models
from dotenv import load_dotenv
from app.utils import logging_config
from app.services.audio import Custom_faster_whisper  # audio.py 경로에 맞게 수정하세요.


# SYSTEM OS & MODEL CACHE DIR
SYSTEM_OS = os.getenv("SYSTEM_OS")
LLAMACPP_CACHE_DIR = os.getenv("LLAMACPP_CACHE_DIR")
HUGGINGFACE_CACHE_DIR = os.getenv("HUGGINGFACE_CACHE_DIR")

# EMBEDDING MODEL CONFIG
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
EMBEDDING_MODEL_LOAD_IN_4BIT = os.getenv("EMBEDDING_MODEL_LOAD_IN_4BIT")
EMBEDDING_MODEL_BBN_4BIT_COMPUTE_DTYPE = os.getenv("EMBEDDING_MODEL_BBN_4BIT_COMPUTE_DTYPE")
EMBEDDING_MODEL_LOW_CPU_USAGE = os.getenv("EMBEDDING_MODEL_LOW_CPU_USAGE")
# EMBEDDING_MODEL_LOAD_IN_8BIT = os.getenv("EMBEDDING_MODEL_LOAD_IN_8BIT")
# EMBEDDING_MODEL_BBN_8BIT_COMPUTE_DTYPE = os.getenv("EMBEDDING_MODEL_BBN_8BIT_COMPUTE_DTYPE")

# RAG MODEL CONFIG
RAG_MODEL_NAME = os.getenv("RAG_MODEL_NAME")
RAG_MODEL_N_CTX = os.getenv("RAG_MODEL_N_CTX")
RAG_MODEL_TEMPERATURE = os.getenv("RAG_MODEL_TEMPERATURE")
RAG_MODEL_N_GPU_LAYERS = os.getenv("RAG_MODEL_N_GPU_LAYERS")

# SUMMARY MODEL CONFIG
SUMMARY_MODEL_NAME = os.getenv("SUMMARY_MODEL_NAME")
SUMMARY_MODEL_N_CTX = os.getenv("SUMMARY_MODEL_N_CTX")
SUMMARY_MODEL_TEMPERATURE = os.getenv("SUMMARY_MODEL_TEMPERATURE")
SUMMARY_MODEL_N_GPU_LAYERS = os.getenv("SUMMARY_MODEL_N_GPU_LAYERS")

# 로깅 설정
logger = logging_config.app_logger


async def load_stt_model(app: FastAPI):
    """
    STT 모델을 로드한 후 app.state에 저장하고, 인스턴스를 반환합니다.
    """
    if not hasattr(app.state, "stt_model"):
        STT_MODEL = 'base'
        logger.info("Loading STT model ...")
        stt_model_instance = Custom_faster_whisper()
        await asyncio.to_thread(stt_model_instance.set_model, STT_MODEL)
        app.state.stt_model = stt_model_instance
        logger.info("STT model loaded successfully!")
    return app.state.stt_model
# def load_stt_model(app: FastAPI):
#     """
#     STT 모델을 로드 후 반환환
#     """

#     if not hasattr(app.state, "stt_model"):
#         logger.info(f"Loading STT model ...")
#         stt_model_instance = Custom_faster_whisper()
#         await asyncio.to_thread(stt_model_instance.set_model, STT_MODEL)
#         stt_model = None  # STT 모델 로드 
#         logger.info(f"STT model loaded successfully!") 
        
#         return stt_model
###########################################################


def load_embedding_model():
    """
    Embedding 모델 로드 후 반환환
    """
    try:
        logger.info(f"Loading Embedding model ...")

        if SYSTEM_OS == "macos":
            logger.info(f"Running on {SYSTEM_OS} - Using non-quantized Embedding Model")
            device = 'mps' if torch.backends.mps.is_available() else 'cpu'
            logger.info(f"Using device: {device}")
            embeddings = SentenceTransformer(model_name_or_path=EMBEDDING_MODEL_NAME,
                                            cache_folder=HUGGINGFACE_CACHE_DIR,
                                            device=device)
            logger.info(f"Embedding model ({EMBEDDING_MODEL_NAME}) loaded successfully!")
            return embeddings
        
        elif SYSTEM_OS in {"linux", "windows"}:
            logger.info(f"Running on {SYSTEM_OS} - Using quantized Embedding Model")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {device}")

            # 양자화 설정
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=EMBEDDING_MODEL_LOAD_IN_4BIT, # 4-bit 양자화 
                bnb_4bit_compute_dtype=EMBEDDING_MODEL_BBN_4BIT_COMPUTE_DTYPE
            )

            # 양자화 모델 로드 
            quantized_model = AutoModel.from_pretrained(pretrained_model_name_or_path=EMBEDDING_MODEL_NAME,
                                                        quantization_config=quantization_config,
                                                        cache_dir=HUGGINGFACE_CACHE_DIR,
                                                        low_cpu_mem_usage=EMBEDDING_MODEL_LOW_CPU_USAGE)
            # 토크나이저 로드
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=EMBEDDING_MODEL_NAME,
                                                    cache_dir=HUGGINGFACE_CACHE_DIR)
            # SentenceTransformer 모듈 구성
            word_embedding_model = models.Transformer(model_name_or_path=EMBEDDING_MODEL_NAME)
            word_embedding_model.auto_model = quantized_model  # 기존 모델을 양자화된 모델로 교체 
            
            # Pooling 레이어 추가 (문장 수준의 임베딩을 위해 필수)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
            
            # SentenceTransformer 객체 생성
            embeddings = SentenceTransformer(modules=[word_embedding_model, pooling_model])

            # Tokenizer도 명시적으로 설정
            embeddings.tokenizer = tokenizer

            logger.info(f"Embedding model ({EMBEDDING_MODEL_NAME}) loaded successfully!")
            return embeddings
        
    except Exception as e:
        logger.error(f"Failed to load Embedding model {EMBEDDING_MODEL_NAME}: {e}")
        raise e


def load_rag_model():
    try:
        logger.info(f"Loading RAG model: {RAG_MODEL_NAME}...")
        logger.info(f"LLAMACPP_CACHE_DIR: {LLAMACPP_CACHE_DIR}")

        # 모델 로드
        rag_model = Llama(
            model_path=LLAMACPP_CACHE_DIR + RAG_MODEL_NAME,
            n_ctx=RAG_MODEL_N_CTX,  # 컨텍스트 윈도우 크기 (입력 프롬프트 최대 토큰 수)
            temperature=RAG_MODEL_TEMPERATURE,
            n_gpu_layers=RAG_MODEL_N_GPU_LAYERS,
            verbose=False)

        logger.info(f"RAG model successfully stored in app.state!")
        
        return rag_model

    except Exception as e:
        logger.error(f"Failed to load RAG model {RAG_MODEL_NAME}: {e}")
        raise e


def load_summary_model():
    """
    Summary 모델 로드 후 반환
    """
    try:
        logger.info(f"Loading Summary model: {SUMMARY_MODEL_NAME}...")
        logger.info(f"LLAMACPP_CACHE_DIR: {LLAMACPP_CACHE_DIR}")

        # 모델 로드
        summary_model = Llama(  
            model_path=LLAMACPP_CACHE_DIR + SUMMARY_MODEL_NAME, 
            n_ctx=SUMMARY_MODEL_N_CTX,  # 컨텍스트 윈도우 크기
            n_gpu_layers=SUMMARY_MODEL_N_GPU_LAYERS,
            temperature=SUMMARY_MODEL_TEMPERATURE,
            verbose=False)

        logger.info(f"Summary model successfully loaded!")

        return summary_model

    except Exception as e:
        logger.error(f"Failed to load Summary model {SUMMARY_MODEL_NAME}: {e}")
        return None


def unload_models(app: FastAPI):
    """
    FastAPI 상태(app.state)에서 모델을 제거하여 메모리 해제 
    """
    if hasattr(app.state, "stt_model"):
        logger.info(f"Unloading model: stt_model")
        del app.state.stt_model
        logger.info(f"STT model unloaded!")

    if hasattr(app.state, "embedding_model"):
        logger.info(f"Unloading model: embedding_model")
        del app.state.embedding_model
        logger.info(f"Embedding model unloaded!")

    if hasattr(app.state, "rag_model"):
        logger.info(f"Unloading model: rag_model")
        del app.state.rag_model
        logger.info(f"RAG model unloaded!")
    
    if hasattr(app.state, "summary_model"):
        logger.info(f"Unloading model: summary_model")
        del app.state.summary_model
        logger.info(f"Summary model unloaded!")
    
    # 메모리 정리
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()  # IPC 캐시 정리 
        torch.cuda.empty_cache()  # VRAM  메모리 캐시 정리 

    logger.info("All models unloaded successfully!")
