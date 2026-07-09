"""5단계 — LangSmith 품질 평가. Agentic RAG 답변의 근거성(Groundedness)을 수치로 측정.

평가 항목 (LLM-as-judge, 각 예제마다 채점):
- groundedness: 답변이 검색된 문맥에만 근거하는가 (환각 측정, 핵심 지표)
- relevance:    답변이 질문에 실제로 답하는가

사용법:
    .venv/bin/python eval_rag.py              # 데이터셋 업로드(최초 1회) + 평가 실행
결과는 LangSmith 웹 → Datasets & Experiments → GMP-RAG-EVAL 에서 확인.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import Client
from pydantic import BaseModel, Field

import config
import graph
import rag

DATASET_NAME = "GMP-RAG-EVAL"

# 평가 질문 — 앞 8개는 CGMP 해설서에서 답 가능, 마지막 2개는 문서에 없는 내용(환각 유도 테스트)
QUESTIONS = [
    "교육훈련 기록은 어떻게 관리해야 하나요?",
    "작업장 청소와 소독은 어떻게 해야 하나요?",
    "원자재 보관 조건은 어떻게 되나요?",
    "일탈이 발생하면 어떻게 처리해야 하나요?",
    "제조번호는 어떻게 부여하나요?",
    "부적합품은 어떻게 관리하나요?",
    "품질관리를 위한 검체 채취는 누가 하나요?",
    "작업자의 위생 관리 기준은 무엇인가요?",
    # 문서에 없음 — "찾을 수 없습니다"로 답해야 groundedness 통과
    "의약품 GMP의 밸리데이션 주기는 몇 년인가요?",
    "화장품 광고 심의 절차는 어떻게 되나요?",
]


def ensure_dataset(client: Client) -> None:
    """평가 데이터셋이 없으면 생성하고 질문 예제를 업로드."""
    if client.has_dataset(dataset_name=DATASET_NAME):
        return
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="GMP Agentic RAG 품질 평가 — CGMP 해설서 기반 8문항 + 문서 외 2문항",
    )
    client.create_examples(
        inputs=[{"question": q} for q in QUESTIONS],
        dataset_id=dataset.id,
    )
    print(f"[*] 데이터셋 생성: {DATASET_NAME} ({len(QUESTIONS)}개 질문)")


# 평가 대상 — 질문 하나를 그래프에 태우고 답변 + 검색 문맥을 반환
def target(inputs: dict) -> dict:
    answer, docs = graph.answer_agentic(inputs["question"], recent_messages=[])
    return {"answer": answer, "context": rag.format_context(docs)}


class _JudgeScore(BaseModel):
    """LLM 판정자 이진 채점"""

    score: str = Field(description="기준을 충족하면 'yes', 아니면 'no'")
    reason: str = Field(description="판정 근거 한 문장")


def _judge(system: str, human: str, variables: dict) -> _JudgeScore:
    llm = ChatOpenAI(model=config.CHAT_MODEL, temperature=0).with_structured_output(_JudgeScore)
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    return (prompt | llm).invoke(variables)


# 근거성 평가 — 답변의 모든 주장이 문맥에서 확인되는가 (환각이면 no)
def groundedness(inputs: dict, outputs: dict) -> dict:
    result = _judge(
        system="답변이 <문맥>에 있는 내용만으로 작성됐는지 평가하세요. "
               "문맥에 없는 주장·수치가 하나라도 있으면 'no'. "
               "'해당 내용을 찾을 수 없습니다' 같은 거절 답변은 지어낸 것이 아니므로 'yes'.",
        human="<문맥>\n{context}\n</문맥>\n\n<답변>\n{answer}\n</답변>",
        variables={"context": outputs["context"], "answer": outputs["answer"]},
    )
    return {"key": "groundedness", "score": int(result.score == "yes"), "comment": result.reason}


# 관련성 평가 — 답변이 질문이 물어본 것에 실제로 답하는가
def relevance(inputs: dict, outputs: dict) -> dict:
    result = _judge(
        system="답변이 질문이 물어본 내용에 실제로 답하는지 평가하세요. "
               "동문서답이면 'no'. 문서에 없어서 답할 수 없다고 명확히 안내하는 것도 'yes'.",
        human="<질문>\n{question}\n</질문>\n\n<답변>\n{answer}\n</답변>",
        variables={"question": inputs["question"], "answer": outputs["answer"]},
    )
    return {"key": "relevance", "score": int(result.score == "yes"), "comment": result.reason}


def main():
    client = Client()
    ensure_dataset(client)

    print(f"[*] 평가 실행 중... ({len(QUESTIONS)}개 질문 × 그래프 + 판정자 2개)")
    results = client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[groundedness, relevance],
        experiment_prefix="agentic-rag",
        max_concurrency=2,
    )

    # 콘솔에 요약 출력 — 상세는 LangSmith 웹 Experiments 탭
    rows = list(results)
    scores = {"groundedness": [], "relevance": []}
    for row in rows:
        for res in row["evaluation_results"]["results"]:
            scores[res.key].append(res.score)
    print()
    for key, vals in scores.items():
        print(f"  {key}: {sum(vals)}/{len(vals)} ({100 * sum(vals) / len(vals):.0f}%)")
    print(f"\n[✓] 완료 — LangSmith 웹 → Datasets & Experiments → {DATASET_NAME} 에서 상세 확인")


if __name__ == "__main__":
    main()
