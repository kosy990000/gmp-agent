# GMP 문서 에이전트 — 개발 진행 기록

## 프로젝트 목표
한국어 GMP(제조·품질관리기준) 문서를 근거로 답변하는 멀티모달 RAG 챗봇.
- 미리 색인된 GMP 규정집 + 채팅에 올린 PDF/이미지를 함께 활용
- 답변에 **출처 인용** 필수 (규제 문서 신뢰성)
- **환각 방지**: 문서에 없으면 "찾을 수 없습니다" 답변

---

## 환경

| 항목 | 값 |
|------|-----|
| 경로 | `/home/ko/Agent/gmp-agent/` |
| Python | 3.12 (`.venv/`) |
| 답변 모델 | `gpt-4o-mini` |
| 임베딩 모델 | `text-embedding-3-small` |
| 벡터 DB | ChromaDB (`storage/chroma/`) |
| API 키 | `.env` → `OPENAI_API_KEY` |

---

## 파일 구조

```
gmp-agent/
├── data/gmp/                 ← GMP 규정집 PDF/Word 원본 (색인 대상)
│   └── 우수화장품 제조 및 품질관리기준(CGMP) 해설서.pdf
├── storage/chroma/           ← 벡터DB (색인 결과 영구 저장)
├── .venv/                    ← Python 3.12 가상환경
├── .env                      ← OPENAI_API_KEY
├── config.py                 ← 경로·모델·청크 설정 (한 곳에서 관리)
├── ingest.py                 ← data/gmp → 청크 → 임베딩 → Chroma 색인
├── rag.py                    ← 검색 + GPT-4o 답변 엔진
├── vision.py                 ← 업로드 이미지 → GPT-4o 비전 인식
├── app.py                    ← Streamlit 채팅 UI
├── requirements.txt
├── README.md
└── PROGRESS.md               ← 이 파일
```

---

## 개발 단계별 기록

### ✅ 프로토타입 (완료)

**구현 내용**
- `config.py` — 모델·경로·청크 설정 중앙화
- `ingest.py` — PDF/Word → 1000자 청크 → OpenAI 임베딩 → Chroma 저장. `--reset` 플래그로 재색인 지원
- `rag.py` — 벡터 유사도 검색 → GPT-4o 답변. 출처 인용·환각 방지 프롬프트 고정
- `vision.py` — 이미지 bytes → GPT-4o 비전 → 텍스트/표 인식
- `app.py` — Streamlit 채팅 UI. 사이드바에서 PDF/이미지 업로드. 참고 청크 펼쳐보기

**비용 절감** (초기 설정 대비)

| 용도 | 변경 전 | 변경 후 | 절감 |
|------|---------|---------|------|
| 답변·이미지 | `gpt-4o` | `gpt-4o-mini` | ~16배↓ |
| 임베딩 | `text-embedding-3-large` | `text-embedding-3-small` | ~5배↓ |

> 비용 절감 후 동일 질문 테스트 통과. 정확도가 더 필요하면 `config.py`의 `CHAT_MODEL`을 `"gpt-4o"`로 변경.

**현재 색인 현황**
- `우수화장품 제조 및 품질관리기준(CGMP) 해설서.pdf` — 290페이지 → 350청크

**동작 확인**
```
질문: 교육훈련 기록은 어떻게 관리해야 하나요?
답변: 교육훈련계획서, 실시 및 평가 기록, 개인별 이력서를 작성·보관...
      [출처: 우수화장품 제조 및 품질관리기준(CGMP) 해설서.pdf p.24]
```

---

### ✅ 1단계 — 하이브리드 검색 (완료)

**목적**: 한국어 GMP 조항번호·전문용어를 키워드로도 검색 (기존 의미 검색만으로는 놓치는 경우 보완)

**변경 파일**: `rag.py` — `retrieve()` 함수만 교체. 다른 파일 무변경.

**적용 기법** (`langchain-kr/10-Retriever/10-Kiwi-BM25Retriever.ipynb` 참고)
- **Kiwi 형태소 분석기** — `교육훈련` → `['교육', '훈련']` 으로 분리해 BM25 인덱스 구축
- **EnsembleRetriever** — BM25 40% + dense 60% 가중치 앙상블
- `_get_all_chunks()`, `_get_base_vectorstore()` — `@lru_cache`로 앱 기동 시 1회만 로드

**전후 비교**

| | 프로토타입 | 1단계 |
|---|---|---|
| 검색 방식 | dense 벡터 검색 | **Kiwi-BM25 + dense 앙상블** |
| 한국어 처리 | 없음 | **형태소 분석** |
| 반환 청크 수 | 최대 5개 | 최대 10개 (중복 제거) |

---

## 다음 단계 (예정)

| 단계 | 내용 | 참고 튜토리얼 |
|------|------|--------------|
| **2단계** | 대화 맥락 기억 — "그럼 그건 몇 년 보관해?" 같은 후속 질문 이해 | `12-RAG/03-Conversation-With-History.ipynb` |
| **3단계** | HWP(한글파일) 로더 + 표·이미지 포함 멀티모달 RAG 강화 | `06-DocumentLoader/02-HWP-Loader.ipynb`, `12-RAG/10-Multi_modal_RAG-GPT-4o.ipynb` |
| **4단계** | Agentic RAG — 스스로 검색 전략 판단 | `15-Agent/06-Agentic-RAG.ipynb` |
| **5단계** | 품질 평가 — 환각·근거성 수치 측정 | `16-Evaluations/11-LangSmith-Groundedness-Evaluation.ipynb` |

---

## 실행 방법

```bash
cd /home/ko/Agent/gmp-agent

# GMP 문서 추가 시 (최초 1회 또는 문서 변경 시)
.venv/bin/python ingest.py --reset

# 앱 실행
.venv/bin/streamlit run app.py
# → 브라우저에서 http://localhost:8501
```
