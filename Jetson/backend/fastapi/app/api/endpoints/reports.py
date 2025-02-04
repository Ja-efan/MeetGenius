from fastapi import APIRouter
from pydantic import BaseModel # 데이터 검증을 위한 모델
import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

model_name = 'gangyeolkim/kobart-korean-summarizer-v2'
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

router = APIRouter(
    prefix="/api/reports",
)

# 실제로는 ChromaDB에서 가져와야 함 !!!
class OriginText(BaseModel):
    doc_ids: list[int] # int로 받는지? str으로 받는지?
    text: str # 요약할 '수정 원문 내용'

@router.get("/")
def test():
    return 'reports page'

@router.post("/summary/")
async def summarize_text(origin_text: OriginText):
    text = origin_text.text.replace('\n', ' ') # 줄바꿈 제거
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True) # 일단은 512 토큰으로 자르기 

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=150,
            early_stopping=True
        )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return {
        "origin_text": text,
        "summary": summary
    }

"""
swagger ui로 테스트할 텍스트
"스마트뷰티 캠페인은 현대 뷰티 시장에서 20대와 30대 여성을 주요 타겟층으로 설정하여 AI 기반 개인 맞춤형 뷰티 솔루션을 제공하는 것을 목표로 한다. 시장 조사에 따르면, 20대 여성의 약 65%가 뷰티와 관련된 새로운 기술에 관심을 가지고 있으며, 이 중 40%는 AI를 활용한 피부 분석 기술에 대해 긍정적인 반응을 보였다. 특히, 2024년 글로벌 뷰티 기술 시장의 성장률은 약 12.8%로 예측되고 있으며, 스마트뷰티는 이러한 흐름에 맞춰 혁신적인 기술과 마케팅 전략을 통해 경쟁 우위를 확보하려 하고 있다. 스마트뷰티 캠페인의 가장 큰 강점은 소셜미디어 플랫폼을 활용한 마케팅 전략이다. 인스타그램은 타겟층의 78%가 일주일에 최소 5일 이상 사용하는 플랫폼으로, 소비자 참여와 브랜드 노출을 극대화하는 데 이상적이다. 틱톡은 젊은 층에서 바이럴 콘텐츠를 확산시키는 데 유리하며, 스마트뷰티는 틱톡 해시태그 챌린지를 통해 약 5천만 조회수를 기록하는 것을 목표로 하고 있다. 유튜브는 심층적인 콘텐츠를 전달할 수 있는 매체로 활용되어, 제품 리뷰 및 사용법 영상을 통해 소비자의 신뢰를 쌓는 데 중점을 둔다. 각 플랫폼은 스마트뷰티의 메시지를 효과적으로 전달하기 위한 역할을 분담하고 있다. 스마트뷰티의 주요 경쟁사로는 B사의 글로벌 뷰티 스캐너가 있다. 이 제품은 AI 기술로 피부 상태를 분석하고 맞춤형 솔루션을 제공하며, 시장 점유율이 약 28%에 달한다. 그러나 스마트뷰티는 경쟁사와 차별화를 위해 정서적 연결을 강화하는 전략을 채택했다. 예를 들어, 소비자 커뮤니티 플랫폼을 구축하여 사용자 리뷰와 경험을 공유하고, AI 분석 결과를 기반으로 한 개인 맞춤형 뷰티 케어 플랜을 제공함으로써 차별화를 꾀하고 있다. 이러한 플랜은 사용자별로 최적화된 스킨케어 루틴과 제품 추천을 포함하며, 약 87%의 사용자 만족도를 목표로 한다. 스마트뷰티 캠페인의 초기 예산은 약 20억 원으로 책정되었다. 이 중 40%는 소셜미디어 마케팅에 배정되었으며, 나머지는 제품 개발(30%), 소비자 체험 이벤트(20%), 기술 연구 및 데이터 보안 강화(10%)에 사용될 예정이다. 소셜미디어 마케팅 예산은 약 8억 원으로, 인스타그램 광고, 틱톡 캠페인, 유튜브 콘텐츠 제작 등에 집중적으로 투자된다. 특히, 틱톡 캠페인을 통해 월간 캠페인 참여율 15% 증가와 브랜드 인지도 상승을 기대하고 있다. 스마트뷰티 캠페인은 리스크 관리 전략도 철저히 마련하고 있다. 예를 들어, 개인정보 보호와 관련된 소비자 우려를 해소하기 위해 데이터 암호화 기술을 도입하고, 소비자가 데이터 사용 방식을 직접 관리할 수 있는 옵션을 제공한다. 또한, 초기 사용자 피드백을 기반으로 제품 개선 주기를 단축하여 출시 후 6개월 이내에 주요 개선 사항을 반영할 계획이다. 시장 진입 초기 발생할 수 있는 높은 마케팅 비용 문제를 해결하기 위해, 캠페인의 ROI(Return on Investment)를 정기적으로 분석하고, 효율성이 낮은 채널에는 예산을 줄이며 고효율 채널로 집중 투자하는 방식으로 운영된다. 스마트뷰티 캠페인은 소비자 중심의 혁신과 정서적 연결을 기반으로 경쟁사와 차별화된 전략을 제시하며, 뷰티 산업에서 독보적인 입지를 확보하는 것을 목표로 한다. 2025년까지 글로벌 시장 점유율 15%를 달성하고, 고객 만족도 90% 이상을 유지하며, 매출 성장률 연평균 25%를 기록하는 것을 중장기적 목표로 설정하고 있다. 이를 통해 스마트뷰티는 단순한 뷰티 기기를 넘어 소비자의 라이프스타일을 변화시키는 동반자로 자리 잡고자 한다."
"""