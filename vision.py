"""업로드된 이미지를 GPT-4o 비전으로 인식. (스캔/사진 GMP 자료, 표 캡처 등)"""
import base64

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

import config


def encode_image(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def read_image(image_bytes: bytes, mime: str = "image/png", question: str | None = None) -> str:
    """이미지에서 텍스트/내용을 추출하거나, 질문이 있으면 그에 답한다."""
    b64 = encode_image(image_bytes)
    instruction = question or (
        "이 이미지는 GMP 관련 자료입니다. 보이는 텍스트와 표를 빠짐없이 그대로 옮겨 적고, "
        "핵심 내용을 한국어로 정리하세요. 보이지 않는 내용은 추측하지 마세요."
    )
    llm = ChatOpenAI(model=config.CHAT_MODEL, temperature=0, max_tokens=2000)
    msg = HumanMessage(content=[
        {"type": "text", "text": instruction},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
    ])
    return llm.invoke([msg]).content
