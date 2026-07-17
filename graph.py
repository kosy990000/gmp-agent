"""4단계 — LangGraph Agentic RAG. 검색 전략을 스스로 판단하는 그래프 구조.

기존 rag.py 의 검색·답변을 그래프 노드로 재구성:

    START → route ─┬→ retrieve → grade ─┬→ generate → END
                   │      ↑             ├→ rewrite ─┘(재검색)
                   │      │             └→ web_search → generate
                   └→ direct → END

- route:      검색 필요 여부 판단 + 후속 질문을 한국어·영어 검색어로 재작성 (LLM 1회로 동시 처리)
- retrieve:   rag.retrieve() 하이브리드 검색을 한·영 검색어로 각각 실행 후 중복 제거
              (ICH 등 영어 문서는 영어 검색어라야 BM25 키워드 매칭이 됨)
- grade:      검색 결과가 질문과 관련 있는지 LLM 평가
- rewrite:    관련성 낮으면 검색 질문을 개선해 재검색 (최대 MAX_QUERY_REWRITES회)
- web_search: 재검색까지 실패하면 웹 검색(OpenAI web_search 툴)으로 폴백 — 찾은 내용을
              텍스트로 받아 기존 generate 문맥에 투입. 답변에 웹 출처임을 명시
- direct:     인사말 등 문서와 무관한 입력은 검색 없이 바로 응답
- generate:   기존 출처 인용·환각 방지 프롬프트로 최종 답변

app.py 는 answer_agentic() 만 호출 — answer_with_history() 와 동일하게
(답변 텍스트, 근거 문서 리스트)를 돌려받음.
"""
from functools import lru_cache
from itertools import zip_longest
from typing import Literal, TypedDict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

import config
import rag


# 그래프를 흐르는 상태 — 노드는 바뀐 키만 dict 로 반환하면 LangGraph 가 병합
class GraphState(TypedDict):
    question: str                    # 사용자 원래 질문
    chat_history: list[BaseMessage]  # 요약 + 최근 대화 (rag.build_chat_history 결과)
    needs_search: bool               # route 노드의 검색 필요 판단
    search_query: str                # 검색에 실제 사용할 질문 (재작성될 수 있음)
    search_query_en: str             # 영어 문서(ICH 등) 검색용 영어 검색 질문
    docs: list[Document]             # 검색된 청크
    rewrite_count: int               # 질문 개선 횟수 (무한 루프 방지)
    web_searched: bool               # 웹 검색 폴백 사용 여부 — generate 가 웹 출처 표시에 사용
    answer: str                      # 최종 답변
    extra_store: Chroma | None       # 세션 업로드 PDF 벡터스토어


def _llm() -> ChatOpenAI:
    return ChatOpenAI(model=config.CHAT_MODEL, temperature=0)


# ── route: 검색 필요 판단 + 후속 질문 재작성 (LLM 1회) ──────────────


class _RouteDecision(BaseModel):
    """검색 필요 여부와 독립 검색 질문을 한 번에 판단"""

    needs_search: bool = Field(
        description="기본값 true. 명백한 인사말·잡담·감사 표현일 때만 false. "
                    "업무·절차·기준·관리에 관한 질문은 전부 true."
    )
    search_query: str = Field(
        description="대화 맥락을 반영해 단독으로 이해 가능하게 재작성한 한국어 검색 질문. 이미 독립적이면 원래 질문 그대로."
    )
    search_query_en: str = Field(
        description="같은 질문을 ICH 가이드라인 등 영어 규제 문서 검색용으로 번역한 영어 검색 질문. "
                    "stability testing, impurities, validation 같은 GMP/ICH 공식 용어를 사용."
    )


def route(state: GraphState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 GMP(제조·품질관리기준) 문서 검색 시스템의 라우터입니다. "
                   "사용자는 GMP 규정에 대해 묻는 실무자이므로, 업무와 조금이라도 관련된 질문은 "
                   "모두 문서 검색이 필요합니다(needs_search=true). "
                   "명백한 인사말·잡담·감사 표현일 때만 needs_search=false 로 판단하세요. "
                   "그리고 대화 기록을 참고해 후속 질문을 단독으로 이해 가능한 검색 질문으로 재작성하고, "
                   "영어 규제 문서(ICH 가이드라인 등) 검색을 위해 같은 질문을 영어로도 작성하세요."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "질문: {question}"),
    ])
    # .get() 기본값 — Studio 에서 question 만 입력해도 실행되도록
    decision = (prompt | _llm().with_structured_output(_RouteDecision)).invoke({
        "chat_history": state.get("chat_history", []),
        "question": state["question"],
    })
    return {
        "needs_search": decision.needs_search,
        "search_query": decision.search_query.strip() or state["question"],
        "search_query_en": decision.search_query_en.strip(),
    }


def route_decision(state: GraphState) -> Literal["retrieve", "direct"]:
    return "retrieve" if state["needs_search"] else "direct"


# ── retrieve: 기존 하이브리드 검색 재사용 ───────────────────────────


def retrieve(state: GraphState) -> dict:
    # 한국어 검색어(국내 문서) + 영어 검색어(ICH 등 영어 문서)로 각각 하이브리드 검색.
    # 영어 문서는 한국어 질문으로 BM25 키워드 매칭이 안 되므로 영어 검색어가 필수
    ko_docs = rag.retrieve(state["search_query"], state.get("extra_store"))
    en_docs = rag.retrieve(state["search_query_en"], state.get("extra_store")) \
        if state.get("search_query_en") else []

    # 한·영 결과를 상위부터 번갈아(interleave) 섞음 — 단순 append 하면 영어 결과가
    # 항상 뒤로 밀려 상위 컷오프에서 통째로 잘리므로, 양쪽 상위 청크를 고르게 보존
    merged = []
    for ko, en in zip_longest(ko_docs, en_docs):
        if ko is not None:
            merged.append(ko)
        if en is not None:
            merged.append(en)

    # 같은 청크 중복 제거 (출처+페이지+본문 앞부분 기준) 후 상위 N개로 제한
    seen, unique = set(), []
    for d in merged:
        key = (d.metadata.get("source"), d.metadata.get("page"), d.page_content[:80])
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return {"docs": unique[:config.RETRIEVE_MERGE_LIMIT]}


# ── grade: 검색 결과 관련성 평가 (조건부 엣지) ──────────────────────


class _Grade(BaseModel):
    """검색 결과 관련성 이진 평가"""

    binary_score: str = Field(description="문서가 질문과 관련 있으면 'yes', 없으면 'no'")


def grade_documents(state: GraphState) -> Literal["generate", "rewrite", "web_search"]:
    # 재작성 한도 초과 — 웹 검색 폴백이 켜져 있으면 웹에서 마지막으로 시도,
    # 꺼져 있으면 그대로 답변 생성 (문맥에 없으면 프롬프트가 "찾을 수 없습니다"로 처리)
    if state.get("rewrite_count", 0) >= config.MAX_QUERY_REWRITES:
        return "web_search" if config.WEB_SEARCH_FALLBACK else "generate"

    prompt = ChatPromptTemplate.from_messages([
        ("system", "검색된 문서가 질문에 답하는 데 관련이 있는지 평가하세요. "
                   "질문의 키워드나 의미와 연결되는 내용이 있으면 'yes', 없으면 'no'."),
        ("human", "[질문]\n{question}\n\n[검색된 문서]\n{context}"),
    ])
    result = (prompt | _llm().with_structured_output(_Grade)).invoke({
        "question": state["search_query"],
        "context": rag.format_context(state["docs"]),
    })
    return "generate" if result.binary_score == "yes" else "rewrite"


# ── rewrite: 검색 질문 개선 후 재검색 ───────────────────────────────


class _RewrittenQueries(BaseModel):
    """실패한 검색 질문을 한국어·영어 두 벌로 개선"""

    search_query: str = Field(
        description="GMP 규정 용어를 사용해 검색이 잘 되도록 재작성한 한국어 검색 질문"
    )
    search_query_en: str = Field(
        description="ICH 가이드라인 등 영어 규제 문서 검색용으로 공식 용어를 써서 재작성한 영어 검색 질문"
    )


def rewrite(state: GraphState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "검색 질문이 GMP/ICH 문서에서 관련 내용을 찾지 못했습니다. "
                   "질문의 의도를 유지하면서 검색이 잘 되도록 한국어 검색 질문과 "
                   "영어 검색 질문(ICH 등 영어 문서용)을 각각 재작성하세요."),
        ("human", "원래 질문: {question}\n실패한 한국어 검색 질문: {search_query}\n실패한 영어 검색 질문: {search_query_en}"),
    ])
    resp = (prompt | _llm().with_structured_output(_RewrittenQueries)).invoke({
        "question": state["question"],
        "search_query": state["search_query"],
        "search_query_en": state.get("search_query_en", ""),
    })
    # LLM이 빈 문자열을 반환하면 이전 검색어(없으면 원 질문)로 보정 — 빈 쿼리로 재검색 방지
    new_ko = resp.search_query.strip() or state["search_query"] or state["question"]
    new_en = resp.search_query_en.strip() or state.get("search_query_en", "")
    return {
        "search_query": new_ko,
        "search_query_en": new_en,
        "rewrite_count": state.get("rewrite_count", 0) + 1,
    }


# ── web_search: 내부 검색 실패 시 웹 검색 폴백 ──────────────────────


# 검색 모델 응답에서 본문 텍스트와 출처 URL(url_citation annotation)을 추출
# Responses API 는 content 를 블록 리스트로 반환하므로 str/list 둘 다 처리
def _extract_text_and_urls(msg) -> tuple[str, list[str]]:
    if isinstance(msg.content, str):
        return msg.content, []
    texts, urls = [], []
    for block in msg.content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            texts.append(block.get("text", ""))
            for ann in block.get("annotations", []):
                if isinstance(ann, dict) and ann.get("type") == "url_citation":
                    url = ann.get("url")
                    if url and url not in urls:
                        urls.append(url)
    return "\n".join(texts), urls


def web_search(state: GraphState) -> dict:
    # 검색 전담 호출 — 답변 생성은 하지 않고 "찾은 내용을 출처와 함께 텍스트로 정리"만 시킴.
    # 최종 답변은 기존 generate 가 이 텍스트를 문맥으로 받아 생성 (출처 인용·환각 방지 규칙 재사용)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 GMP(제조·품질관리기준) 규정 전문 웹 검색 담당자입니다. "
                   "질문에 답하는 데 필요한 정보를 웹에서 검색하세요. "
                   "식약처(mfds.go.kr) 등 공식 규제기관 자료를 우선하세요. "
                   "직접 답변하지 말고, 찾은 내용을 요약·각색 없이 출처 URL과 함께 한국어로 정리만 하세요. "
                   "각 내용 뒤에 반드시 출처 URL을 붙이세요. 찾지 못했으면 '검색 결과 없음'이라고만 쓰세요."),
        ("human", "{question}"),
    ])
    llm = ChatOpenAI(model=config.CHAT_MODEL, temperature=0, use_responses_api=True).bind_tools(
        [{"type": "web_search"}]
    )
    try:
        resp = (prompt | llm).invoke({"question": state["search_query"]})
        text, urls = _extract_text_and_urls(resp)
    except Exception:
        # 웹 검색 실패 시 기존 검색 결과 그대로 generate — 종전 동작("찾을 수 없습니다")과 동일
        return {"web_searched": False}

    if not text.strip() or "검색 결과 없음" in text[:30]:
        return {"web_searched": False}

    if urls:
        text += "\n\n[참고 URL]\n" + "\n".join(urls)
    doc = Document(page_content=text, metadata={"source": "웹 검색 (내부 문서 아님)"})
    return {"docs": [doc], "web_searched": True}


# ── generate / direct: 최종 답변 ────────────────────────────────────


def generate(state: GraphState) -> dict:
    system = rag.SYSTEM_PROMPT
    if state.get("web_searched"):
        system += (
            "\n\n[주의] 내부 GMP 문서에서 답을 찾지 못해 <문맥>이 웹 검색 결과입니다. "
            "답변 첫 줄에 반드시 \"⚠️ 내부 문서에서 찾지 못해 웹 검색 결과를 근거로 한 답변입니다.\"라고 명시하고, "
            "출처는 [출처: URL] 형식으로 표기하세요."
        )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "<문맥>\n{context}\n</문맥>\n\n질문: {question}"),
    ])
    resp = (prompt | _llm()).invoke({
        "chat_history": state.get("chat_history", []),
        "context": rag.format_context(state.get("docs", [])),
        "question": state["question"],
    })
    return {"answer": resp.content}


def generate_direct(state: GraphState) -> dict:
    # 검색이 불필요한 입력(인사말 등) — 출처 인용 규칙 없이 짧게 응답
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 GMP(제조·품질관리기준) 문서 전문 어시스턴트입니다. "
                   "문서 검색 없이 간단히 한국어로 응답하고, GMP 관련 질문을 하도록 안내하세요."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    resp = (prompt | _llm()).invoke({
        "chat_history": state.get("chat_history", []),
        "question": state["question"],
    })
    return {"answer": resp.content, "docs": []}


# ── 그래프 조립 — 앱 기동 후 1회만 컴파일 ───────────────────────────


@lru_cache(maxsize=1)
def _get_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("route", route)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
    workflow.add_node("direct", generate_direct)

    workflow.add_edge(START, "route")
    workflow.add_conditional_edges("route", route_decision, {"retrieve": "retrieve", "direct": "direct"})
    workflow.add_conditional_edges(
        "retrieve", grade_documents,
        {"generate": "generate", "rewrite": "rewrite", "web_search": "web_search"},
    )
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
    workflow.add_edge("direct", END)
    return workflow.compile()


# LangGraph Studio(`langgraph dev`) 진입점 — langgraph.json 에서 참조
studio_graph = _get_graph()


# 대화 맥락을 고려한 Agentic RAG 답변 — answer_with_history() 와 같은 인터페이스
def answer_agentic(
    question: str,
    recent_messages: list[dict],
    history_summary: str = "",
    extra_store: Chroma | None = None,
):
    result = _get_graph().invoke({
        "question": question,
        "chat_history": rag.build_chat_history(recent_messages, history_summary),
        "needs_search": True,
        "search_query": question,
        "search_query_en": "",
        "docs": [],
        "rewrite_count": 0,
        "web_searched": False,
        "answer": "",
        "extra_store": extra_store,
    })
    return result["answer"], result["docs"]
