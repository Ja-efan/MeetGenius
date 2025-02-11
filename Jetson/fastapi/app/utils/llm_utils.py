import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedTokenizerFast, BartForConditionalGeneration
from sentence_transformers import SentenceTransformer, models
import logging
# from llama_cpp import Llama


# 로그 설정
logging.basicConfig(level=logging.INFO)
 
def load_stt_model(app_state):
    """
    STT 모델을 로드 후 반환환
    """

    if not hasattr(app_state, "stt_model"):
        print(f"Loading STT model ...")
        stt_model = None  # STT 모델 로드 
        print(f"STT model loaded successfully!") 
        
        return stt_model


def load_embedding_model(app_state):
    """
    Embedding 모델 로드 후 반환환
    """
    if not hasattr(app_state, "embedding_model"):

        model_name_or_path ="nlpai-lab/KoE5"
        
        print(f"Loading Embedding model ...")

        # 양자화 설정 
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, # 4-bit 양자화 
            # load_in_8bit=True, # 8-bit 양자화
        )

        # 양자화 모델 로드 
        quantized_model = AutoModel.from_pretrained(model_name_or_path,
                                          quantization_config=quantization_config,
                                          cache_dir="../.huggingface-cache/")
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
        
        logging.info("Embedding model (KoE5) loaded successfully!")
        
        return sentence_embedding_model


def load_rag_model(app_state):        
    """
    RAG 모델 로드 후 반환
    """
    if not hasattr(app_state, "rag_model"):

        model_name_or_path = "Qwen/Qwen2.5-0.5B-Instruct"

        print(f"Loading RAG model ...")

        # 양자화 설정
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            # load_in_8bit=True,
        )

        # 양자화된 모델 로드 
        rag_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                                       quantization_config=quantization_config,
                                                                       cache_dir="../.huggingface-cache/")
    
        # app_state.state.rag_model = None 
        logging.info("RAG model loaded successfully!")

        return rag_model


def load_summary_model(app_state):
    """
    Summary 모델 로드 후 반환
    """
    if not hasattr(app_state, "summary_model"):
        try:
            print(f"Loading summary model ...")

            # 요약 모델 로딩
            model_name = 'gangyeolkim/kobart-korean-summarizer-v2'
            tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
            model = BartForConditionalGeneration.from_pretrained(model_name)

            # 모델과 토크나이저를 app.state에 저장
            app_state.summary_model = {
                'tokenizer': tokenizer,
                'model': model
            }

            logging.info("Summary model loaded successfully!")

        except Exception as e:
            logging.error(f"요약 모델 로드 실패: {str(e)}")
            return None
        
    return app_state.summary_model


def unload_models(app_state):
    """
    FastAPI 상태(app.state)에서 모델을 제거하여 메모리 해제 
    """
    if hasattr(app_state, "stt_model"):
        del app_state.stt_model
        print(f"STT model unloaded!")

    if hasattr(app_state, "embedding_model"):
        del app_state.embedding_model
        print(f"Embedding model unloaded!")
    
    if hasattr(app_state, "rag_model"):
        del app_state.rag_model
        print(f"RAG model unloaded!")
    
    if hasattr(app_state, "summary_model"):
        del app_state.summary_model
        print(f"Summary model unloaded!")
    

    print("All models unloaded successfully!!")
