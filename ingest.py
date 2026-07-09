"""data/gmp 안의 GMP 규정집(PDF/DOCX/HWP)을 청크로 쪼개 Chroma 벡터DB에 색인.

사용법:
    .venv/bin/python ingest.py            # data/gmp 전체 색인
    .venv/bin/python ingest.py --reset    # 기존 색인 지우고 새로
    .venv/bin/python ingest.py --no-vision  # 이미지 설명 생성 건너뜀 (비용 절약)
"""
import argparse
import base64
import shutil

import zlib

import fitz  # PyMuPDF
import olefile
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

import config

# 이미지 설명 생성에 사용할 최소 이미지 크기 (너무 작은 로고·아이콘 제외)
MIN_IMAGE_BYTES = 10_000


def _load_hwp(path) -> list[Document]:
    """HWP(한글) 파일에서 텍스트를 추출해 Document 리스트로 반환.
    HWP5 OLE 구조의 BodyText 섹션을 zlib 압축 해제 후 UTF-16LE 디코딩.
    """
    if not olefile.isOleFile(str(path)):
        print(f"    [!] OLE 형식이 아닌 HWP 파일 건너뜀: {path.name}")
        return []

    ole = olefile.OleFileIO(str(path))
    texts = []
    # BodyText/Section0, Section1, ... 순서로 읽음
    for i in range(100):
        entry = f"BodyText/Section{i}"
        if not ole.exists(entry):
            break
        data = ole.openstream(entry).read()
        try:
            decompressed = zlib.decompress(data, -15)
        except zlib.error:
            decompressed = data
        # HWP 텍스트 레코드에서 한글 유니코드(UTF-16LE) 추출
        extracted = []
        j = 0
        while j + 4 <= len(decompressed):
            header = int.from_bytes(decompressed[j:j+4], "little")
            rec_type = header & 0x3FF
            rec_size = (header >> 20) & 0xFFF
            if rec_size == 0xFFF and j + 8 <= len(decompressed):
                rec_size = int.from_bytes(decompressed[j+4:j+8], "little")
                j += 4
            j += 4
            if rec_type == 67 and j + rec_size <= len(decompressed):  # HWPTAG_PARA_TEXT
                chunk = decompressed[j:j+rec_size]
                chars = []
                for k in range(0, len(chunk) - 1, 2):
                    code = int.from_bytes(chunk[k:k+2], "little")
                    if code in (0x0D, 0x0A, 0x2029):
                        chars.append("\n")
                    elif 0x20 <= code <= 0xD7FF or 0xE000 <= code <= 0xFFFD:
                        chars.append(chr(code))
                extracted.append("".join(chars))
            j += rec_size
        if extracted:
            texts.append("\n".join(extracted))
    ole.close()

    if not texts:
        return []
    return [Document(page_content="\n".join(texts), metadata={"source": path.name, "page": 0})]


def load_documents(folder) -> list[Document]:
    """PDF / DOCX / HWP 파일을 모두 로드. 메타데이터에 출처 파일명 기록."""
    docs = []
    exts = {".pdf", ".docx", ".hwp"}
    files = [p for p in folder.rglob("*") if p.suffix.lower() in exts]
    if not files:
        print(f"[!] {folder} 안에 문서 파일이 없습니다.")
        return docs
    for path in files:
        print(f"  - 로드: {path.name}")
        ext = path.suffix.lower()
        if ext == ".pdf":
            loaded = PyMuPDFLoader(str(path)).load()
        elif ext == ".docx":
            loaded = Docx2txtLoader(str(path)).load()
        elif ext == ".hwp":
            loaded = _load_hwp(path)
        for d in loaded:
            d.metadata["source"] = path.name
        docs.extend(loaded)
    return docs


def extract_image_descriptions(folder, vision_model: str) -> list[Document]:
    """PDF 페이지에서 이미지를 추출하고 GPT-4o 비전으로 설명문 청크를 생성."""
    llm = ChatOpenAI(model=vision_model, max_tokens=512)
    desc_docs = []

    pdf_files = [p for p in folder.rglob("*") if p.suffix.lower() == ".pdf"]
    for path in pdf_files:
        print(f"  - 이미지 추출: {path.name}")
        pdf = fitz.open(str(path))
        for page_num, page in enumerate(pdf):
            for img_info in page.get_images():
                xref = img_info[0]
                extracted = pdf.extract_image(xref)
                img_bytes, ext = extracted["image"], extracted["ext"]

                # 너무 작은 이미지(로고·구분선 등) 제외
                if len(img_bytes) < MIN_IMAGE_BYTES:
                    continue

                # PDF 내부 이미지는 원본 포맷 그대로 추출됨 — 비전 API가 받는 포맷만
                # 실제 MIME 으로 전송하고, 나머지(jpx·tiff 등)는 PNG 로 변환
                if ext in ("jpeg", "jpg"):
                    mime = "image/jpeg"
                elif ext in ("png", "gif", "webp"):
                    mime = f"image/{ext}"
                else:
                    pix = fitz.Pixmap(pdf, xref)
                    if pix.colorspace and pix.colorspace.n > 3:  # CMYK 등은 RGB 변환 필요
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    img_bytes, mime = pix.tobytes("png"), "image/png"

                b64 = base64.b64encode(img_bytes).decode()
                try:
                    resp = llm.invoke([
                        HumanMessage(content=[
                            {"type": "text", "text": (
                                "이 이미지는 한국 GMP(제조·품질관리) 문서의 일부입니다. "
                                "이미지 안의 표, 그림, 수치를 한국어로 간결하게 설명하세요. "
                                "내용이 없거나 장식용이면 '해당 없음'이라고만 답하세요."
                            )},
                            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                        ])
                    ])
                    description = resp.content.strip()
                except Exception as e:
                    print(f"    [!] 비전 오류 (p.{page_num+1}): {e}")
                    continue

                if description == "해당 없음":
                    continue

                desc_docs.append(Document(
                    page_content=description,
                    metadata={
                        "source": path.name,
                        "page": page_num,
                        "type": "image_description",
                    },
                ))
                print(f"    + 이미지 설명 추가 (p.{page_num+1}): {description[:50]}...")
        pdf.close()

    return desc_docs


def split_documents(docs: list[Document]) -> list[Document]:
    """한국어 규제 문서 기준으로 청크 분할. 문단·문장 경계 우선."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_index(reset=False, use_vision=True):
    if reset and config.CHROMA_DIR.exists():
        print("[*] 기존 색인 삭제...")
        shutil.rmtree(config.CHROMA_DIR)
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[*] 문서 로드 중: {config.DATA_DIR}")
    docs = load_documents(config.DATA_DIR)
    if not docs:
        return
    chunks = split_documents(docs)
    print(f"[*] {len(docs)}개 페이지 → {len(chunks)}개 텍스트 청크")

    # 이미지 설명 청크 추가
    if use_vision:
        print("[*] 이미지 설명 생성 중... (--no-vision 으로 건너뛸 수 있음)")
        img_chunks = extract_image_descriptions(config.DATA_DIR, config.CHAT_MODEL)
        print(f"[*] 이미지 설명 {len(img_chunks)}개 추가")
        chunks += img_chunks

    embeddings = OpenAIEmbeddings(model=config.EMBED_MODEL)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.COLLECTION,
        persist_directory=str(config.CHROMA_DIR),
    )
    print(f"[✓] 색인 완료 → {config.CHROMA_DIR}  (총 {len(chunks)}개 청크)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="기존 색인 삭제 후 재색인")
    parser.add_argument("--no-vision", action="store_true", help="이미지 설명 생성 건너뜀")
    args = parser.parse_args()
    build_index(reset=args.reset, use_vision=not args.no_vision)
