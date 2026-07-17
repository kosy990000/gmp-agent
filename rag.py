"""검색(retrieval) + GPT-4o 답변 체인. 규제 문서라 '출처 인용 + 환각 방지'가 핵심."""
from functools import lru_cache

from kiwipiepy import Kiwi
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

import config

# 형태소 분석기 — 모듈 로드 시 1회만 초기화
_kiwi = Kiwi()


# BM25 키워드 검색을 위해 한국어 텍스트를 형태소 단위로 분리
# "교육훈련" → ["교육", "훈련"] 으로 쪼개야 단어 단위 검색이 정확해짐
def _kiwi_tokenize(text: str) -> list[str]:
    return [t.form for t in _kiwi.tokenize(text)]


# Chroma에서 전체 청크를 Document 리스트로 한 번만 로드 — BM25 인덱스 구축용
# lru_cache 로 앱 기동 후 첫 호출 시 1회만 실행, 이후 메모리에서 재사용
@lru_cache(maxsize=1)
def _get_all_chunks() -> list[Document]:
    vs = _get_base_vectorstore()
    raw = vs.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]


# 디스크의 Chroma 벡터 DB에 연결 — dense(의미) 검색의 기반
# lru_cache 로 커넥션을 1회만 열고 재사용
@lru_cache(maxsize=1)
def _get_base_vectorstore() -> Chroma:
    return Chroma(
        collection_name=config.COLLECTION,
        embedding_function=OpenAIEmbeddings(model=config.EMBED_MODEL),
        persist_directory=str(config.CHROMA_DIR),
    )


# BM25 인덱스를 전체 청크로 한 번만 빌드 — 매 질문마다 재생성하지 않도록 캐싱
@lru_cache(maxsize=1)
def _get_bm25_retriever() -> BM25Retriever:
    chunks = _get_all_chunks()
    return BM25Retriever.from_documents(chunks, preprocess_func=_kiwi_tokenize, k=config.TOP_K)


SYSTEM_PROMPT = """당신은 GMP(제조·품질관리기준) 문서 전문 어시스턴트입니다.
규칙:
1. 반드시 아래 <문맥>에 주어진 내용만 근거로 답하세요. 문맥에 없으면 "제공된 문서에서 해당 내용을 찾을 수 없습니다"라고 답하세요. 추측하지 마세요.
2. 답변 끝에 근거가 된 출처를 [출처: 파일명 p.페이지] 형식으로 표기하세요.
3. 규제 문서이므로 임의로 요약·각색하지 말고, 조항의 의미를 정확히 전달하세요.
4. 한국어로 답하세요. 문맥에 영어 문서(ICH 가이드라인 등)가 포함된 경우에도 한국어로 답하되, 핵심 전문 용어는 영어 원문을 괄호로 병기하세요. 예: 안정성 시험(stability testing)"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "<문맥>\n{context}\n</문맥>\n\n질문: {question}"),
])


# 검색된 청크 목록을 GPT에 넘길 문맥 문자열로 조립
# 각 청크 앞에 [파일명 p.페이지] 태그를 붙여 GPT가 출처를 인용할 수 있게 함
def format_context(docs: list[Document]) -> str:
    blocks = []
    for d in docs:
        src = d.metadata.get("source", "?")
        page = d.metadata.get("page")
        tag = f"{src} p.{page + 1}" if isinstance(page, int) else src
        blocks.append(f"[{tag}]\n{d.page_content}")
    return "\n\n---\n\n".join(blocks) if blocks else "(관련 문서 없음)"


# 하이브리드 검색: BM25(키워드 40%) + dense(의미 60%) 앙상블
# 세션 업로드 PDF(extra_store)가 있으면 거기서도 dense 검색해서 결과에 합침
def retrieve(question: str, extra_store: Chroma | None = None, k: int = config.TOP_K):
    bm25 = _get_bm25_retriever()
    bm25.k = k  # 인덱스는 캐시 재사용, 결과 수만 호출마다 반영 — dense 와 개수를 맞춰 앙상블 비율 유지
    dense = _get_base_vectorstore().as_retriever(search_kwargs={"k": k})
    ensemble = EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[0.4, 0.6],
    )
    docs = ensemble.invoke(question)

    if extra_store is not None:
        docs += extra_store.similarity_search(question, k=k)

    return docs


# 질문 하나를 받아 검색 → 문맥 조립 → GPT 답변까지 처리하는 메인 함수
# app.py 는 이 함수만 호출하면 (답변 텍스트, 근거 문서 리스트) 를 돌려받음
def answer(question: str, extra_store: Chroma | None = None):
    docs = retrieve(question, extra_store)
    context = format_context(docs)
    llm = ChatOpenAI(model=config.CHAT_MODEL, temperature=0)
    chain = PROMPT | llm
    resp = chain.invoke({"context": context, "question": question})
    return resp.content, docs


# 오래된 대화를 요약하고 질문 주제·패턴을 동시에 추출
# 반환: {"summary": str, "topics": list[str], "question_pattern": str}
def summarize_history(old_messages: list[dict], existing_summary: str = "") -> dict:
    turns = "\n".join(
        f"{'사용자' if m['role'] == 'user' else '어시스턴트'}: {m['content']}"
        for m in old_messages
    )
    base = f"[기존 요약]\n{existing_summary}\n\n" if existing_summary else ""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """GMP 문서 상담 대화를 분석해 아래 JSON 형식으로만 답하세요. 다른 텍스트는 쓰지 마세요.

{{
  "summary": "사용자가 X를 물었고 Y라고 답했다. (반드시 3문장 이내로 압축, 원문 그대로 쓰지 말 것)",
  "topics": ["주제1", "주제2"],
  "question_pattern": "아래 중 가장 적합한 것 하나만: 기간 문의 / 절차 문의 / 기준 확인 / 서류 관리 / 기타"
}}"""),
        ("human", f"{base}[대화]\n{turns}"),
    ])
    llm = ChatOpenAI(model=config.CHAT_MODEL, temperature=0)
    raw = (prompt | llm).invoke({}).content.strip()

    import json, re
    try:
        # LLM이 ```json ... ``` 블록으로 감쌀 때 대비
        cleaned = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
        return json.loads(cleaned)
    except Exception:
        # 파싱 실패 시 raw 텍스트를 요약으로만 사용
        return {"summary": raw, "topics": [], "question_pattern": "기타"}


# 요약을 SystemMessage 로, 최근 대화를 Human/AI 메시지로 변환
# answer_with_history() 와 graph.py(Agentic RAG) 가 공유
def build_chat_history(recent_messages: list[dict], history_summary: str = ""):
    chat_history = []
    if history_summary:
        from langchain_core.messages import SystemMessage
        chat_history.append(SystemMessage(content=f"[이전 대화 요약]\n{history_summary}"))
    chat_history += [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in recent_messages
    ]
    return chat_history


# 대화 맥락을 고려한 답변 — 오래된 대화는 요약, 최근 N턴만 전문 유지
# history_summary: 오래된 대화 요약 (app.py 의 session_state 에서 관리)
# recent_messages: 최근 N턴 메시지 (현재 질문 제외)
def answer_with_history(
    question: str,
    recent_messages: list[dict],
    history_summary: str = "",
    extra_store: Chroma | None = None,
):
    chat_history = build_chat_history(recent_messages, history_summary)

    llm = ChatOpenAI(model=config.CHAT_MODEL, temperature=0)

    # 히스토리가 없으면 바로 검색 (첫 질문은 재작성 불필요)
    if not chat_history:
        search_query = question
    else:
        # 후속 질문을 독립적인 검색 질문으로 재작성
        condense_prompt = ChatPromptTemplate.from_messages([
            ("system", "아래 대화 기록을 참고해 후속 질문을 단독으로 이해 가능한 검색 질문으로 재작성하세요. 이미 독립적이면 그대로 반환하세요. 한국어로만 답하세요."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "후속 질문: {question}"),
        ])
        rephrased = (condense_prompt | llm).invoke({"chat_history": chat_history, "question": question})
        search_query = rephrased.content.strip()

    # 재작성된 질문으로 검색
    docs = retrieve(search_query, extra_store)
    context = format_context(docs)

    # 원래 질문 + 대화 히스토리 + 문맥으로 최종 답변 생성
    history_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "<문맥>\n{context}\n</문맥>\n\n질문: {question}"),
    ])
    resp = (history_prompt | llm).invoke({
        "chat_history": chat_history,
        "context": context,
        "question": question,
    })
    return resp.content, docs
