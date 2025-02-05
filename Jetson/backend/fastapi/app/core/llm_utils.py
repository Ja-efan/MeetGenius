import requests
import torch
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from  sentence_transformers import SentenceTransformer, models
import json
import os 
from dotenv import load_dotenv
import logging


# 로그 설정
logging.basicConfig(level=logging.INFO)
 
def load_stt_model(app):
    """
    STT 모델을 로드하여 FastAPI의 상태 (app.state)에 저장 
    """

    if not hasattr(app.state, "stt_model"):
        print(f"Loading STT model ...")
        app.state.stt_model = None 
        print(f"STT model loaded successfully!")


def load_embedding_model(app):
    """
    Embedding 모델을 로드하여 FastAPI의 상태 (app.state)에 저장 
    """
    if not hasattr(app.state, "embedding_model"):

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

        # SentenceTransformer 객체 생성
        sentence_embedding_model = SentenceTransformer(modules=[word_embedding_model])

        # Tokenizer도 명시적으로 설정
        sentence_embedding_model.tokenizer = tokenizer

        app.state.embedding_model = sentence_embedding_model
        
        logging.info("Embedding model (KoE5) loaded successfully!")


def load_llm_model(app):
    """
    LLM을 로드하여 FastAPI의 상태 (app.state)에 저장 
    """
    if not hasattr(app.state, "llm_model"):
        print(f"Loading LLM model ...")
        app.state.llm_model = None 
        # print(f"LLM model loaded successfully!")
        logging.info("LLM model loaded successfully!")


def unload_models(app):
    """
    FastAPI 상태(app.state)에서 모델을 제거하여 메모리 해제 
    """
    if hasattr(app.state, "stt_model"):
        del app.state.stt_model
        print(f"STT model unloaded!")

    if hasattr(app.state, "embedding_model"):
        del app.state.embedding_model
        print(f"Embedding model unloaded!")
    
    if hasattr(app.state, "llm_model"):
        del app.state.llm_model
        print(f"LLM model unloaded!")
    
    print("All models unloaded successfully!!")
