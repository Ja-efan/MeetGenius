import requests
import torch
from transformers import AutoTokenizer, AutoModel
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
        print(f"Loading Embedding model ...")
        app.state.embedding_model = None 
        print(f"Embedding model loaded successfully!")


def load_llm_model(app):
    """
    LLM을 로드하여 FastAPI의 상태 (app.state)에 저장 
    """
    if not hasattr(app.state, "llm_model"):
        print(f"Loading LLM model ...")
        app.state.llm_model = None 
        print(f"LLM model loaded successfully!")


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
