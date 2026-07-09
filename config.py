"""공통 설정 — 경로, 모델 이름 등을 한 곳에서 관리."""
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # .env 의 OPENAI_API_KEY 등 로드

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "gmp"          # 미리 색인할 GMP 규정집 원본
CHROMA_DIR = BASE_DIR / "storage" / "chroma"   # 벡터DB 영구 저장 위치
COLLECTION = "gmp_docs"                          # 미리 색인된 기본 컬렉션

# 모델 — 한국어 GMP 문서 기준 (비용 절감 설정)
CHAT_MODEL = "gpt-4o-mini"              # 답변 + 이미지(비전) 인식 겸용. 정확도 더 원하면 "gpt-4o"
EMBED_MODEL = "text-embedding-3-small"  # 저렴한 임베딩. 바꾸면 ingest.py --reset 재색인 필요

# 청크 설정 (규제 문서는 조항 단위 보존이 중요 → 다소 큰 청크 + 겹침)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5  # 검색 시 가져올 청크 수
HISTORY_WINDOW = 3  # 전문 그대로 유지할 최근 대화 턴 수 (초과분은 요약)
MAX_QUERY_REWRITES = 2  # Agentic RAG: 검색 결과가 부실할 때 질문 재작성 최대 횟수
WEB_SEARCH_FALLBACK = True  # 내부 문서 검색이 끝내 실패하면 웹 검색으로 폴백 (OpenAI web_search 툴, 호출당 추가 과금)

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
