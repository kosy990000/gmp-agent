# GMP 문서 에이전트

한국어 GMP(제조·품질관리) 문서를 근거로 답하는 멀티모달 RAG 챗봇.
미리 색인된 GMP 규정집 + 채팅에 올린 PDF/이미지를 함께 활용하며, **답변에 출처를 인용**합니다.

## 구성
| 파일 | 역할 |
|------|------|
| `config.py` | 경로·모델·청크 설정 |
| `ingest.py` | `data/gmp/` 의 GMP 규정집을 Chroma에 색인 |
| `rag.py` | 검색 + GPT-4o 답변 (출처 인용·환각 방지) |
| `vision.py` | 업로드 이미지 → GPT-4o 비전 인식 |
| `app.py` | Streamlit 채팅 UI |

## 사용법
```bash
# 1) GMP 규정집(PDF/Word) 넣기
#    → data/gmp/ 폴더에 복사

# 2) 색인 (최초 1회, 문서 추가 시 다시)
.venv/bin/python ingest.py            # 추가 색인
.venv/bin/python ingest.py --reset    # 처음부터 다시

# 3) 앱 실행
.venv/bin/streamlit run app.py
```

## 모델
- 답변·이미지 인식: `gpt-4o`
- 임베딩: `text-embedding-3-large`
- API 키: `.env` 의 `OPENAI_API_KEY`

## 개발 로드맵

```
[✅ 프로토타입]
  질문 ──► dense 검색 (벡터 유사도)
            └─► GPT-4o-mini 답변 + 출처 인용

  파일: config.py / ingest.py / rag.py / vision.py / app.py
  색인: CGMP 해설서 290p → 350청크

──────────────────────────────────────────────────

[✅ 1단계] 하이브리드 검색
  질문 ──► BM25 (키워드 40%)  ─┐
        └► dense (의미  60%)  ─┴─► 앙상블 ──► GPT 답변

  추가: Kiwi 형태소 분석 ("교육훈련" → ["교육","훈련"])
  수정: rag.py 의 retrieve() + BM25 lru_cache 캐싱

──────────────────────────────────────────────────

[✅ 2단계] 대화 맥락 기억
  질문 + 이전 대화 ──► 질문 재작성
                        └─► 하이브리드 검색 ──► GPT 답변

  추가: answer_with_history() — 후속 질문 재작성 + 최근 N턴 + 요약 방식
  수정: rag.py 에 summarize_history() 추가, app.py 창 관리 로직 추가
  토큰: 대화 20턴 기준 ~4000토큰 → ~450토큰 (요약 150 + 최근 3턴 300)

──────────────────────────────────────────────────

[✅ 3단계] 멀티모달 RAG (심플 방식)
  PDF 텍스트 청크          ──► Chroma (기존)
  PDF 이미지 → GPT-4o 설명문 → Chroma (추가)
  HWP → OLE 파싱 → 텍스트 청크 → Chroma (추가)

  수정: ingest.py — HWP 로더(_load_hwp) + 이미지 설명 추출(extract_image_descriptions)
  옵션: --no-vision 플래그로 이미지 설명 생성 건너뜀 (비용 절약)

[⬜ 4단계] Agentic RAG
  질문 ──► LangGraph 가 검색 전략 스스로 판단·반복

[⬜ 5단계] 품질 평가
  LangSmith 로 환각률 · 근거성 수치 측정
```

## 보완 예정

### 1단계 — 하이브리드 검색
- [ ] BM25 가중치(현재 40:60) 실험적으로 조정
- [ ] 청크 크기(CHUNK_SIZE=1000) GMP 조항 단위 최적화

### 2단계 — 대화 맥락
- [ ] 히스토리 SQLite 영구 저장 (현재 세션 메모리만, 새로고침 시 사라짐)
- [ ] DB 통계 사이드바 뷰어 (주제별 질문 빈도 시각화)
- [ ] HISTORY_WINDOW 값 사용 패턴 기반으로 조정

### 3단계 — 멀티모달 RAG
- [ ] MultiVectorRetriever 방식으로 업그레이드 (이미지 원본 보존, 표 구조 분석)
- [ ] HWP 로더 표 추출 보완 (현재 텍스트만, 표 셀 구조 손실)
- [ ] 이미지 필터링 기준(MIN_IMAGE_BYTES) 문서별 조정
- [ ] 색인 시 이미지 설명 캐싱 (재색인 시 동일 이미지 재처리 방지)

### 4단계 (예정)
- [ ] LangGraph 기반 Agentic RAG 구조로 전환
  - 관련성 판단 노드 → 쿼리 재작성 노드 → 환각 체크 노드
- [ ] 현재 retrieve(), answer_with_history() 를 그래프 노드로 리팩토링

### 5단계 (예정)
- [ ] LangSmith Groundedness 평가 연동
- [ ] 테스트 질문셋 구축 (GMP 조항 기반)
