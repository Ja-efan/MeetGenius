import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedTokenizerFast, BartForConditionalGeneration
from sentence_transformers import SentenceTransformer, models
from llama_cpp import Llama
from fastapi import FastAPI
from pathlib import Path
from app.utils import logging_config

# 로깅 설정
logger = logging_config.setup_logger(__name__)

# 모델 캐시 디렉토리
BASE_DIR = Path(__file__).resolve().parent.parent.parent
LLM_MODELS_DIR = BASE_DIR / ".llm-model-caches"


def load_stt_model(app: FastAPI):
    """
    STT 모델을 로드 후 반환환
    """

    if not hasattr(app.state, "stt_model"):
        logger.info(f"🔄 [INFO] Loading STT model ...")
        stt_model = None  # STT 모델 로드 
        logger.info(f"✅ [INFO] STT model loaded successfully!") 
        
        return stt_model


def load_embedding_model(app: FastAPI):
    """
    Embedding 모델 로드 후 반환환
    """
    if not hasattr(app.state, "embedding_model"):

        model_name_or_path ="nlpai-lab/KoE5"
        
        logger.info(f"🔄 [INFO] Loading Embedding model ...")

        # 양자화 설정 
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, # 4-bit 양자화 
            # load_in_8bit=True, # 8-bit 양자화
        )

        # 양자화 모델 로드 
        quantized_model = AutoModel.from_pretrained(model_name_or_path,
                                          quantization_config=quantization_config,
                                          cache_dir=LLM_MODELS_DIR / "huggingface-caches")
        # Tokenizer 로드 
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # SentenceTransformer 모듈 구성
        word_embedding_model = models.Transformer(model_name_or_path)
        word_embedding_model.auto_model = quantized_model  # 기존 모델을 양자화된 모델로 교체 

        # Pooling 레이어 추가 (문장 수준의 임베딩을 위해 필수)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

        # SentenceTransformer 객체 생성 (Transformer와 Pooling을 결합)
        sentence_embedding_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        # Tokenizer도 명시적으로 설정
        sentence_embedding_model.tokenizer = tokenizer
        
        logger.info(f"✅ [INFO] Embedding model loaded successfully!")
        
        return sentence_embedding_model


def load_rag_model(app: FastAPI, 
                   rag_model_name="EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf",
                   n_gpu_layers: int=-1,
                   metadata: bool=True,
                   context_params: bool=True):
    try:
        print(f"✅ [INFO] Loading RAG model: {rag_model_name}...")
        llama_cpp_dir = LLM_MODELS_DIR / "llama-cpp"
        print(f"✅ [INFO] LLM_MODELS_DIR: {llama_cpp_dir}")

        # 모델 로드
        rag_model = Llama(
            model_path=str(llama_cpp_dir / rag_model_name), 
            n_ctx=2048,  # 컨텍스트 윈도우 크기 (입력 프롬프트 토큰 최대 2048)
            n_gpu_layers=n_gpu_layers)

        logger.info(f"✅ [INFO] RAG model successfully stored in app.state!")

        # # 반환할 데이터 구성
        # response_data = {}
        
        # if metadata:
        #     response_data["metadata"] = app_state.rag_model.metadata
        
        # if context_params:
        #     response_data["context_params"] = app_state.rag_model.context_params
        
        # # 반환할 데이터가 없다면 None을 반환
        # return response_data if response_data else None
        
        return rag_model

    except Exception as e:
        logger.error(f"❌ [ERROR] Failed to load RAG model {rag_model_name}: {e}")
        raise e


def load_summary_model(app: FastAPI):
    """
    Summary 모델 로드 후 반환
    """
    if not hasattr(app.state, "summary_model"):
        try:
            print(f"Loading summary model ...")

            # 요약 모델 로딩
            model_name = 'gangyeolkim/kobart-korean-summarizer-v2'
            tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
            model = BartForConditionalGeneration.from_pretrained(model_name)

            # 모델과 토크나이저를 app.state에 저장
            app.state.summary_model = {
                'tokenizer': tokenizer,
                'model': model
            }

            logger.info("✅ [INFO] Summary model loaded successfully!")

        except Exception as e:
            logger.error(f"❌ [ERROR] Failed to load summary model: {str(e)}")
            return None
        
    return app.state.summary_model


def unload_models(app: FastAPI):
    """
    FastAPI 상태(app.state)에서 모델을 제거하여 메모리 해제 
    """
    if hasattr(app.state, "stt_model"):
        logger.info(f"🔄 [INFO] Unloading model: stt_model")
        del app.state.stt_model
        print(f"STT model unloaded!")

    if hasattr(app.state, "embedding_model"):
        logger.info(f"🔄 [INFO] Unloading model: embedding_model")
        del app.state.embedding_model
        print(f"Embedding model unloaded!")

    if hasattr(app.state, "rag_model"):
        logger.info(f"🔄 [INFO] Unloading model: rag_model")
        del app.state.rag_model
        print(f"RAG model unloaded!")
    
    if hasattr(app.state, "summary_model"):
        logger.info(f"🔄 [INFO] Unloading model: summary_model")
        del app.state.summary_model
        print(f"Summary model unloaded!")
    
    logger.info("✅ [INFO] All models unloaded successfully!")
