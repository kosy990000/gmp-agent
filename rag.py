"""검색(retrieval) + GPT-4o 답변 체인. 규제 문서라 '출처 인용 + 환각 방지'가 핵심."""
from functools import lru_cache

from kiwipiepy import Kiwi
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

import config

# 형태소 분석기 — 모듈 로드 시 1회만 초기화
_kiwi = Kiwi()


def _kiwi_tokenize(text: str) -> list[str]:
    """한국어 형태소 분석 토크나이저 (BM25 전용)."""
    return [t.form for t in _kiwi.tokenize(text)]


@lru_cache(maxsize=1)
def _get_all_chunks() -> list[Document]:
    """Chroma에서 전체 청크를 한 번만 로드해 BM25 인덱스를 위한 문서 목록으로 반환."""
    vs = _get_base_vectorstore()
    raw = vs.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]


@lru_cache(maxsize=1)
def _get_base_vectorstore() -> Chroma:
    return Chroma(
        collection_name=config.COLLECTION,
        embedding_function=OpenAIEmbeddings(model=config.EMBED_MODEL),
        persist_directory=str(config.CHROMA_DIR),
    )

SYSTEM_PROMPT = """당신은 GMP(제조·품질관리기준) 문서 전문 어시스턴트입니다.
규칙:
1. 반드시 아래 <문맥>에 주어진 내용만 근거로 답하세요. 문맥에 없으면 "제공된 문서에서 해당 내용을 찾을 수 없습니다"라고 답하세요. 추측하지 마세요.
2. 답변 끝에 근거가 된 출처를 [출처: 파일명 p.페이지] 형식으로 표기하세요.
3. 규제 문서이므로 임의로 요약·각색하지 말고, 조항의 의미를 정확히 전달하세요.
4. 한국어로 답하세요."""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "<문맥>\n{context}\n</문맥>\n\n질문: {question}"),
])


def format_context(docs: list[Document]) -> str:
    """검색된 청크를 출처 표시와 함께 문맥 문자열로 만든다."""
    blocks = []
    for d in docs:
        src = d.metadata.get("source", "?")
        page = d.metadata.get("page")
        tag = f"{src} p.{page + 1}" if isinstance(page, int) else src
        blocks.append(f"[{tag}]\n{d.page_content}")
    return "\n\n---\n\n".join(blocks) if blocks else "(관련 문서 없음)"


def retrieve(question: str, extra_store: Chroma | None = None, k: int = config.TOP_K):
    """하이브리드 검색: Kiwi-BM25(키워드) + dense(의미) 앙상블.
    GMP 조항번호·전문용어는 키워드로, 의미·맥락은 dense로 보완한다.
    """
    # --- 기본 GMP 색인 하이브리드 검색 ---
    chunks = _get_all_chunks()
    bm25 = BM25Retriever.from_documents(chunks, preprocess_func=_kiwi_tokenize, k=k)
    dense = _get_base_vectorstore().as_retriever(search_kwargs={"k": k})
    ensemble = EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[0.4, 0.6],  # 의미 검색에 조금 더 가중치
    )
    docs = ensemble.invoke(question)

    # --- 세션 업로드 PDF가 있으면 거기서도 검색해서 합침 ---
    if extra_store is not None:
        docs += extra_store.similarity_search(question, k=k)

    return docs


def answer(question: str, extra_store: Chroma | None = None):
    """질문에 대해 검색 → GPT-4o 답변. (답변문자열, 근거문서목록) 반환."""
    docs = retrieve(question, extra_store)
    context = format_context(docs)
    llm = ChatOpenAI(model=config.CHAT_MODEL, temperature=0)
    chain = PROMPT | llm
    resp = chain.invoke({"context": context, "question": question})
    return resp.content, docs
