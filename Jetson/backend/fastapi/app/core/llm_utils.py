import requests
import torch
from transformers import AutoTokenizer, AutoModel
import json
import os 
from dotenv import load_dotenv
import logging

# 로그 설정
logging.basicConfig(level=logging.INFO)
 

def get_embedding_model(model_path: str='nlpai-lab/KoE5', device: torch.device='cuda'):
    """지정된 경로에서 임베딩 모델(예: KoE5)을 로드하여 토크나이저와 모델을 반환합니다.

    Args:
        model_path (str): 
            사전 학습된 모델의 로컬 경로나 Hugging Face 모델 식별자.
        device (torch.device): 
            모델을 로드할 디바이스(CPU 또는 GPU).

    Returns:
        Tuple[AutoTokenizer, AutoModel]:
            로드된 토크나이저와 모델을 포함하는 튜플. 주어진 디바이스에서 사용 가능.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    emb_model = AutoModel.from_pretrained(
        model_path, 
        torch_dtype=torch.float16,  # fp16 사용 시 GPU 메모리 절감 가능
        # low_cpu_mem_usage=True  # CPU 메모리 사용량 최소화
    ).to(device)
    emb_model.eval()
    return tokenizer, emb_model


def encode_text(texts: list, tokenizer, model, device: torch.device, batch_size: int = 16):
    """입력된 텍스트 리스트를 임베딩 벡터로 변환하는 함수.

    Args:
        texts (list): 
            임베딩 변환할 텍스트 문자열 리스트.
        tokenizer (transformers.PreTrainedTokenizer): 
            Hugging Face Transformers의 토크나이저로, 입력된 텍스트를 토큰으로 변환하는 역할을 수행.
        model (transformers.PreTrainedModel): 
            Hugging Face Transformers의 사전 학습된 모델로, 토큰화된 입력을 임베딩 벡터로 변환.
        device (torch.device): 
            추론을 수행할 디바이스(CPU 또는 GPU).
        batch_size (int, optional): 
            한 번에 처리할 텍스트 개수. 기본값은 16.

    Returns:
        torch.Tensor:
            입력된 각 문장에 대한 임베딩 벡터를 포함하는 2차원 텐서 (shape: [num_texts, embedding_dim]).
    """
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu()
        all_embeddings.append(embeddings)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings


def ollama_rag(prompt: str, llm_model: str = 'exaone3.5:2.4b', 
               temperature: float = 0.5, top_p: float = 0.9, max_tokens: int = 512, stream: bool=False) -> str:
    """Ollama 서버에 프롬프트를 전송하여 생성된 응답을 반환하는 함수.

    Args:
        prompt (str):
            LLM(대규모 언어 모델)에게 전달할 입력 프롬프트.
        llm_model (str, optional):
            Ollama 서버에서 사용할 모델의 이름. 
            기본값은 'exaone3.5:2.4b'.
        temperature (float, optional):
            생성 결과의 다양성을 조절하는 온도 값. 높은 값일수록 더 창의적인 응답을 생성.
            기본값은 0.5.
        top_p (float, optional):
            Nucleus sampling(Top-p) 기법을 적용하여 높은 확률을 가진 토큰들만 선택하도록 제한.
            기본값은 0.9.
        max_tokens (int, optional):
            생성할 최대 토큰 수를 설정. 기본값은 512.

    Returns:
        str:
            Ollama 서버에서 생성된 응답 텍스트.
    """
    payload = {
        'model': llm_model,
        'prompt': prompt,
        'format': 'json',
        'stream': False,
        'options': {
            'temperature': temperature,
            'top_p': top_p,
            'max_tokens': max_tokens
        }
    }

    response = requests.post(OLLAMA_GENERATE_URL, json=payload)
    
    response.raise_for_status()

    if response.status_code == 200:
        try:
            response_json = response.json()
            logging.info(f"Ollama Generated Answer: {response.json()}")
            return response_json.get("response", "").strip()
        except json.JSONDecodeError:
            return response.text.strip()
    else:
        return ""