"""GMP 문서 에이전트 — Streamlit 채팅 UI (PDF/이미지 업로드 + 출처 인용 답변).

실행:
    .venv/bin/streamlit run app.py
"""
import tempfile
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import config
import rag
import vision

st.set_page_config(page_title="GMP 문서 에이전트", page_icon="🏭", layout="wide")
st.title("🏭 GMP 문서 에이전트")
st.caption("미리 색인된 GMP 규정집 + 채팅에 올린 PDF/이미지를 근거로 답변합니다. (출처 인용)")


@st.cache_resource(show_spinner=False)
def session_embeddings():
    return OpenAIEmbeddings(model=config.EMBED_MODEL)


def index_uploaded_pdf(file) -> int:
    """업로드 PDF를 세션용 인메모리 Chroma에 추가. 추가된 청크 수 반환."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.getbuffer())
        tmp_path = tmp.name
    docs = PyMuPDFLoader(tmp_path).load()
    for d in docs:
        d.metadata["source"] = file.name
    Path(tmp_path).unlink(missing_ok=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    if st.session_state.session_store is None:
        st.session_state.session_store = Chroma(
            collection_name="session_uploads",
            embedding_function=session_embeddings(),
        )  # persist_directory 없음 → 인메모리 (세션 한정)
    st.session_state.session_store.add_documents(chunks)
    return len(chunks)


# ── 세션 상태 초기화 ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_store" not in st.session_state:
    st.session_state.session_store = None

# ── 사이드바: 파일 업로드 ─────────────────────────────────────────
with st.sidebar:
    st.header("📎 문서 업로드")
    st.write("이 세션에서만 질문에 활용할 PDF/이미지를 올리세요.")

    pdf_files = st.file_uploader("PDF 업로드", type=["pdf"], accept_multiple_files=True)
    if pdf_files and st.button("PDF 색인하기"):
        for f in pdf_files:
            with st.spinner(f"색인 중: {f.name}"):
                n = index_uploaded_pdf(f)
            st.success(f"{f.name}: {n}개 청크 추가됨")

    st.divider()
    img_file = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg"])
    if img_file:
        st.image(img_file, caption=img_file.name, use_container_width=True)
        if st.button("이미지 인식하기"):
            mime = f"image/{'jpeg' if img_file.type.endswith('jpeg') else img_file.type.split('/')[-1]}"
            with st.spinner("GPT-4o 비전으로 인식 중..."):
                result = vision.read_image(img_file.getvalue(), mime=mime)
            st.session_state.messages.append(
                {"role": "assistant", "content": f"🖼️ **이미지 인식 결과 ({img_file.name})**\n\n{result}"}
            )
            st.rerun()

# ── 기존 대화 출력 ────────────────────────────────────────────────
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ── 채팅 입력 ─────────────────────────────────────────────────────
if prompt := st.chat_input("GMP 문서에 대해 질문하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("문서 검색 + 답변 생성 중..."):
            try:
                content, docs = rag.answer(prompt, extra_store=st.session_state.session_store)
            except Exception as e:
                content, docs = f"⚠️ 오류: {e}", []
        st.markdown(content)
        if docs:
            with st.expander("🔎 참고한 문서 청크"):
                for d in docs:
                    src = d.metadata.get("source", "?")
                    page = d.metadata.get("page")
                    tag = f"{src} p.{page + 1}" if isinstance(page, int) else src
                    st.markdown(f"**{tag}**")
                    st.text(d.page_content[:500])
    st.session_state.messages.append({"role": "assistant", "content": content})
