'''
embedding 생성 로직.
아래는 Solar Embedding API 연동 예시 코드입니다.
'''

import httpx
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel

# 환경 변수 로드
load_dotenv()
SOLAR_API_URL = os.getenv("SOLAR_API_URL")
SOLAR_API_KEY = os.getenv("SOLAR_API_KEY")

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
        # torch_dtype=torch.float16,  # fp16 사용 시 GPU 메모리 절감 가능
        # low_cpu_mem_usage=True  # CPU 메모리 사용량 최소화
        
        
    ).to(device)
    emb_model.eval()
    return tokenizer, emb_model

async def get_embedding(text: str):
    """
    Solar Embedding API를 호출하여 텍스트 임베딩을 생성하는 함수.
    """
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {SOLAR_API_KEY}"}
        payload = {"text": text}

        try:
            response = await client.post(SOLAR_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            return response.json().get("embedding")  # API 응답에서 임베딩 값 추출
        except Exception as e:
            print(f"Solar Embedding API Error: {e}")
            return None