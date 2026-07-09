# GMP 문서 에이전트 — 개발 진행 기록

## 프로젝트 목표
한국어 GMP(제조·품질관리기준) 문서를 근거로 답변하는 멀티모달 RAG 챗봇.
- 미리 색인된 GMP 규정집 + 채팅에 올린 PDF/이미지를 함께 활용
- 답변에 **출처 인용** 필수 (규제 문서 신뢰성)
- **환각 방지**: 문서에 없으면 웹 검색 폴백(⚠️ 웹 출처 명시), 그마저 없으면 "찾을 수 없습니다" 답변

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
├── ingest.py                 ← data/gmp → 청크 → 임베딩 → Chroma 색인 (+ HWP 로더, 이미지 설명)
├── rag.py                    ← 검색 + GPT-4o 답변 엔진 (+ 대화 맥락, 히스토리 요약)
├── graph.py                  ← LangGraph Agentic RAG (검색 판단·관련성 평가·재검색)
├── vision.py                 ← 업로드 이미지 → GPT-4o 비전 인식
├── db.py                     ← SQLite 대화 로그 (storage/history.db)
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

### ✅ 2단계 — 대화 맥락 기억 + 히스토리 압축 (완료)

**목적**: "그럼 그건 몇 년 보관해?" 같은 후속 질문 이해 + 긴 대화의 토큰 비용 억제

**변경 파일**: `rag.py`(함수 추가), `app.py`(호출부), `db.py`(신규), `config.py`(`HISTORY_WINDOW=3`)

**구현 내용** (`langchain-kr/12-RAG/03-Conversation-With-History.ipynb` 참고)
- `answer_with_history()` — 후속 질문을 LLM으로 **독립 검색어로 재작성**(condense) 후 검색, 원래 질문 + 대화 히스토리로 최종 답변
- **히스토리 압축**: 최근 `HISTORY_WINDOW=3`턴만 전문 유지, 창 밖으로 밀려난 대화는 `summarize_history()`로 요약 (대화 20턴 기준 ~4000토큰 → ~450토큰)
- `summarize_history()` — LLM 1회 호출로 요약 + 주제(topics) + 질문 유형(question_pattern) 동시 추출 (JSON)
- `db.py` — SQLite(`storage/history.db`)에 session_id·timestamp·summary·topics·question_pattern 저장. `fetch_logs()`, `fetch_topic_stats()` 조회 함수 포함

---

### ✅ 3단계 — 멀티모달 RAG, 심플 방식 (완료)

**목적**: HWP(한글파일) 지원 + 문서 내 표·그림도 검색 대상에 포함

**변경 파일**: `ingest.py`만 수정 — rag.py·app.py 무변경 (설명문이 일반 텍스트 청크로 색인되므로)

**구현 내용** (`06-DocumentLoader/02-HWP-Loader.ipynb`, `12-RAG/10-Multi_modal_RAG-GPT-4o.ipynb` 참고)
- **HWP 로더** `_load_hwp()` — `olefile`로 OLE 파싱 → BodyText 섹션 zlib 해제 → UTF-16LE 텍스트 추출
  - `langchain_teddynote.HWPLoader`는 구버전 `langchain.schema` 의존이라 사용 불가 → 직접 구현
- **이미지 설명 색인** `extract_image_descriptions()` — PyMuPDF로 PDF 이미지 추출 → GPT-4o 비전이 한국어 설명문 생성 → `type: image_description` 메타데이터로 Chroma 저장
  - `MIN_IMAGE_BYTES=10000` 미만 작은 이미지(로고·구분선) 제외, "해당 없음" 응답 제외
  - `--no-vision` 플래그로 비전 호출 건너뜀 (비용 절약)

**보완 예정** (README.md에도 기록)
- MultiVectorRetriever 방식 업그레이드 (이미지 원본 보존, 표 구조 분석)
- HWP 표 추출, 이미지 설명 캐싱 (재색인 시 재처리 방지)
- DB 통계 사이드바 뷰어

---

### ✅ 4단계 — LangGraph Agentic RAG (완료)

**목적**: 검색 전략을 스스로 판단 — 검색 필요 여부, 검색 결과 관련성 평가, 부실하면 질문을 고쳐 재검색

**변경 파일**: `graph.py`(신규), `rag.py`(`build_chat_history()` 헬퍼 추출), `app.py`(호출부 교체), `config.py`(`MAX_QUERY_REWRITES=2`)

**그래프 구조** (`17-LangGraph/02-Structures/06-LangGraph-Agentic-RAG.ipynb` 참고)

```
START → route ─┬→ retrieve → grade ─┬→ generate → END
               │      ↑             └→ rewrite ─┘(재검색)
               └→ direct → END
```

- **route** — 검색 필요 판단 + 후속 질문 독립 검색어 재작성을 structured output 으로 **LLM 1회에 동시 처리** (기존 condense 단계 흡수)
- **retrieve** — 기존 `rag.retrieve()` 하이브리드 검색 그대로 재사용
- **grade** — 검색 청크가 질문과 관련 있는지 이진 평가 (조건부 엣지)
- **rewrite** — 관련성 낮으면 GMP 용어로 검색 질문 개선 후 재검색. `MAX_QUERY_REWRITES=2`회 초과 시 그대로 generate (환각 방지 프롬프트가 "찾을 수 없습니다" 처리)
- **direct** — 인사말·잡담은 검색 없이 짧게 응답 (근거 문서 0개)
- 그래프는 `@lru_cache`로 1회만 컴파일. 체크포인터 없음 — 히스토리는 기존 app.py session_state 방식 유지

**인터페이스**: `graph.answer_agentic(question, recent_messages, history_summary, extra_store)` — `answer_with_history()` 와 동일 시그니처라 app.py 는 호출부 한 줄만 교체. `rag.answer_with_history()` 는 비교·폴백용으로 유지.

**LangGraph Studio (시각화·디버깅)**
- `langgraph-cli[inmem]` 설치, `langgraph.json` 작성 (`graph.py:studio_graph` 진입점)
- 실행: `.venv/bin/langgraph dev` → https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024 (무료 LangSmith 로그인 필요)
- Studio 에서 `question` 만 입력해도 돌도록 노드들이 상태 키를 `.get()` 기본값으로 읽음
- 라우팅 버그 수정: "작업장 청소" 같은 업무 질문이 direct 로 빠지던 문제 → 라우터 프롬프트를 검색 기본값으로 강하게 편향 (인사말·잡담만 direct)

**동작 확인**
- 문서 질문 → retrieve→generate, 출처 인용 포함 답변 (근거 8청크)
- 후속 질문("그럼 그 기록은 몇 년 보관해야 해?") → 맥락 반영 재작성 후 검색, 문서에 없는 내용은 "찾을 수 없습니다" (환각 방지 유지)
- 인사말 → direct 경로, 근거 문서 0개

---

### ✅ 5단계 — LangSmith 품질 평가 (완료)

**목적**: 환각·근거성을 수치로 측정 — "잘 되는 것 같다"를 "몇 %"로

**변경 파일**: `eval_rag.py`(신규). `.env`에 `LANGSMITH_TRACING/API_KEY/PROJECT` 추가됨 (앱 실행도 전부 자동 추적)

**구현 내용** (`16-Evaluations/11-LangSmith-Groundedness-Evaluation.ipynb` 참고)
- 노트북의 Upstage 검사기는 별도 API 키가 필요해서, OpenAI 기반 **자체 LLM-as-judge** 로 구현 (structured output 이진 채점 + 판정 근거)
- **데이터셋** `GMP-RAG-EVAL` — CGMP 해설서로 답 가능한 8문항 + **문서에 없는 2문항(환각 유도)**
- **평가자 2개**: `groundedness`(답변이 검색 문맥에만 근거하는가 — 거절 답변은 yes), `relevance`(질문에 실제로 답하는가)
- `client.evaluate(target, data=..., evaluators=[...])` — target 은 그래프 실행 후 `{"answer", "context"}` 반환

**첫 평가 결과** (experiment: `agentic-rag-ed2d08ca`)

| 지표 | 점수 |
|------|------|
| groundedness | **10/10 (100%)** |
| relevance | **10/10 (100%)** |

- 환각 유도 2문항 모두 "제공된 문서에서 해당 내용을 찾을 수 없습니다"로 정직하게 거절 (원본 답변 직접 확인)
- 재실행: `.venv/bin/python eval_rag.py` — 상세는 LangSmith 웹 → Datasets & Experiments → GMP-RAG-EVAL

---

### ✅ 6단계 — 웹 검색 폴백 (완료, 2026-07-09)

**목적**: 내부 문서에서 끝내 답을 못 찾으면 "찾을 수 없습니다"로 끝내는 대신, 웹 검색으로 마지막 시도 — 단 웹 출처임을 명확히 표시

**변경 파일**: `graph.py`(web_search 노드 추가), `config.py`(`WEB_SEARCH_FALLBACK=True`)

**그래프 구조** (변경 후)

```
START → route ─┬→ retrieve → grade ─┬→ generate → END
               │      ↑             ├→ rewrite ─┘(재검색)
               │      │             └→ web_search → generate   ← 신규
               └→ direct → END
```

**설계 결정 — 검색과 답변 생성의 역할 분리**
- Tavily 등 외부 검색 API 대신 **OpenAI 내장 `web_search` 툴**(Responses API) 사용 — 별도 API 키·패키지 불필요, 기존 OPENAI_API_KEY 하나로 동작
- 검색 모델은 **답변하지 않고 "찾은 내용을 출처 URL과 함께 텍스트로 정리"만** 수행 → 그 텍스트를 `Document`로 포장해 기존 `format_context()` → `generate` 파이프라인에 그대로 투입 (출처 인용·환각 방지 규칙 전부 재사용)
- `rewrite` 한도(`MAX_QUERY_REWRITES=2`) 초과 시 generate 대신 web_search 로 분기. 웹 검색 실패·결과 없음이면 종전 동작("찾을 수 없습니다") 유지
- 웹 폴백 답변은 첫 줄에 **"⚠️ 내부 문서에서 찾지 못해 웹 검색 결과를 근거로 한 답변입니다"** 강제 + `[출처: URL]` 형식 (규제 문서 특성상 내부 규정과 웹 정보가 섞여 보이면 안 됨)
- 검색 프롬프트에서 식약처(mfds.go.kr) 등 공식 규제기관 자료 우선 지시
- `config.WEB_SEARCH_FALLBACK` 스위치로 끄기 가능 (웹 검색 호출당 추가 과금 있음)

**동작 확인**
- EU GMP Annex 1 질문(내부 화장품 CGMP 문서에 없음) → `route→retrieve→rewrite×2→retrieve→web_search→generate` 경로, ⚠️ 표시 + raps.org 출처 답변
- 교육훈련 질문(내부 문서에 있음) → 기존 경로 그대로, 회귀 없음
- GMP 무관 질문(최저시급) → 여전히 direct 로 차단
- Streamlit UI 에서 헤드리스 브라우저로 채팅 end-to-end 확인 (출처 인용·참고 청크 표시 정상)

---

### ✅ 코드 리뷰 반영 — 버그 3건 수정 (완료, 2026-07-09)

Codex 코드 리뷰(`/codex:review`)가 지적한 기존 코드 버그 3건 수정.

**1. [P2] `app.py` — 이미 요약된 대화를 매번 재요약**
- `old_messages` 전체(대화 처음부터)를 매 턴 `summarize_history()`에 넘겨, 긴 세션에서 같은 메시지가 반복 재처리 → 토큰 압축 목적 무력화 + 요약 왜곡
- 수정: `old_messages[summarized_count:]` 슬라이스로 **새로 창 밖으로 밀려난 메시지만 증분 요약**. 시뮬레이션으로 각 메시지가 정확히 1회만 투입됨을 확인

**2. [P2] `ingest.py` — PDF 추출 이미지를 전부 `image/jpeg`로 라벨링**
- PyMuPDF `extract_image()`는 원본 포맷 그대로 반환하는데 MIME 하드코딩 → 비전 API 거부 → except 가 조용히 skip → 이미지 설명 색인 누락
- 수정: `ext` 기반 실제 MIME 사용(jpeg/png/gif/webp), 미지원 포맷(jpx·tiff 등)은 `fitz.Pixmap`으로 PNG 변환(CMYK→RGB 포함)
- **실측: 현재 코퍼스 PDF 이미지 319개 중 242개(76%)가 PNG** — 기존 색인에서 상당수 누락됐을 가능성 높음 → **재색인 필요 (아래 다음 단계)**

**3. [P3] `rag.py` — 캐시된 BM25 가 `retrieve(k=...)` 무시**
- `lru_cache` 도입 후 BM25 쪽 k 가 최초 빌드 시점 `TOP_K=5`로 고정 → k 를 바꾸면 dense 와 개수가 어긋나 앙상블 비율(40:60) 왜곡되는 잠복 버그 (현재 호출부는 전부 기본값이라 실동작 영향 없음)
- 수정: 인덱스는 캐시 재사용, `retrieve()` 안에서 `bm25.k = k` 호출마다 설정. k=3/k=8 실측 확인

---

## 다음 단계 (예정)

- **이미지 재색인** `.venv/bin/python ingest.py --reset` — MIME 버그 수정 후 미실행 상태. 비전 호출 ~319회 비용 발생하므로 실행 판단 필요
- 평가 데이터셋 확장 (후속 질문 시나리오, 이미지 설명 청크 검증 문항, **웹 폴백 시나리오**)
- README.md 보완 예정 항목: MultiVectorRetriever, HWP 표 추출, 이미지 설명 캐싱, DB 통계 사이드바
- 프롬프트·검색 파라미터 변경 시 eval_rag.py 로 회귀 테스트

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
