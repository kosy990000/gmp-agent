"""data/gmp 안의 GMP 규정집(PDF/Word)을 청크로 쪼개 Chroma 벡터DB에 색인.

사용법:
    .venv/bin/python ingest.py            # data/gmp 전체 색인
    .venv/bin/python ingest.py --reset    # 기존 색인 지우고 새로
"""
import argparse
import shutil

from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import config


def load_documents(folder):
    """폴더 안의 PDF/DOCX 파일을 모두 로드. 메타데이터에 출처 파일명 기록."""
    docs = []
    files = [p for p in folder.rglob("*") if p.suffix.lower() in {".pdf", ".docx"}]
    if not files:
        print(f"[!] {folder} 안에 PDF/DOCX 파일이 없습니다. 문서를 넣어주세요.")
        return docs
    for path in files:
        print(f"  - 로드: {path.name}")
        if path.suffix.lower() == ".pdf":
            loaded = PyMuPDFLoader(str(path)).load()  # 페이지별 Document, page 메타 포함
        else:
            loaded = Docx2txtLoader(str(path)).load()
        for d in loaded:
            d.metadata["source"] = path.name
        docs.extend(loaded)
    return docs


def split_documents(docs):
    """한국어 규제 문서 기준으로 청크 분할. 문단·문장 경계 우선."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_index(reset=False):
    if reset and config.CHROMA_DIR.exists():
        print("[*] 기존 색인 삭제...")
        shutil.rmtree(config.CHROMA_DIR)
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[*] 문서 로드 중: {config.DATA_DIR}")
    docs = load_documents(config.DATA_DIR)
    if not docs:
        return
    chunks = split_documents(docs)
    print(f"[*] {len(docs)}개 페이지 → {len(chunks)}개 청크")

    embeddings = OpenAIEmbeddings(model=config.EMBED_MODEL)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.COLLECTION,
        persist_directory=str(config.CHROMA_DIR),
    )
    print(f"[✓] 색인 완료 → {config.CHROMA_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="기존 색인 삭제 후 재색인")
    args = parser.parse_args()
    build_index(reset=args.reset)
