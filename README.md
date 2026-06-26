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

## 다음 단계(아이디어)
- 한국어 키워드 매칭 강화를 위한 BM25(kiwi 토크나이저) 하이브리드 검색
- 표/양식 보존을 위한 레이아웃 인식 파서(예: Upstage Document Parse)
- 답변 근거 평가(LangSmith) 및 환각 자동 점검
