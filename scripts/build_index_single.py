#!/usr/bin/env python3
"""医疗指南 RAG 索引构建 - 单进程稳定版"""
import os, uuid
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
import pypdfium2
from paddleocr import PaddleOCR

DATA_DIR = Path("/home/kehua/storage0/rag/data")
VECTORSTORE_DIR = Path("/home/kehua/storage0/rag/vectorstore")
COLLECTION_NAME = "health_knowledge_base"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
MODEL_PATH = "/home/kehua/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"
BATCH_SIZE = 32

def collect_pdfs(root):
    return sorted([p for ext in [".pdf"] for p in root.rglob("*" + ext)])

def main():
    print("=" * 60)
    print("医疗指南 RAG 索引构建（单进程稳定版）")
    print("=" * 60)

    pdf_files = collect_pdfs(DATA_DIR)
    print(f"发现 {len(pdf_files)} 个 PDF")
    if not pdf_files:
        return

    print("\n加载 Embedding 模型...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16).cuda()
    mdl.eval()
    dim = mdl.config.hidden_size
    print(f"向量维度: {dim}")

    print("\n初始化 ChromaDB...")
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR), settings=Settings(allow_reset=True))
    col = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    print(f"Collection: {col.name}, 当前: {col.count()} 条")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", " "], length_function=len
    )

    ocr = None
    total = 0
    errors = []

    for pdf_path in tqdm(pdf_files, desc="处理"):
        fname = pdf_path.name
        rel = str(pdf_path.relative_to(DATA_DIR.parent))
        try:
            pdf = pypdfium2.PdfDocument(str(pdf_path))
            parts = []
            for pi in range(len(pdf)):
                t = pdf[pi].get_textpage().get_text_bounded()
                if t and t.strip():
                    parts.append(t)
            pdf.close()
            text = "\n\n".join(parts)

            if not text or len(text.strip()) < 20:
                if ocr is None:
                    print("\n初始化 OCR..."); ocr = PaddleOCR(lang="ch", device="metax_gpu", use_textline_orientation=True); print("OK")
                pdf = pypdfium2.PdfDocument(str(pdf_path))
                lines = []
                for pi in range(len(pdf)):
                    bmp = pdf[pi].render(scale=2.0, rotation=0)
                    img = np.array(bmp.to_pil())
                    for block in ocr.predict(img):
                        for ln in (block or []):
                            if ln and len(ln) >= 2:
                                lines.append(ln[1] if isinstance(ln[1], str) else str(ln[1]))
                pdf.close()
                text = "\n".join(lines)

            if not text or len(text.strip()) < 20:
                errors.append(f"{fname}: 无文字")
                continue

            chunks = splitter.split_text(text)
            ids, vecs, docs, metas = [], [], [], []

            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i : i + BATCH_SIZE]
                inp = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inp = {k: v.cuda() for k, v in inp.items()}
                with torch.no_grad():
                    out = mdl(**inp)
                    emb = torch.nn.functional.normalize(out.last_hidden_state[:, 0, :], p=2, dim=1)
                for j, (c, v) in enumerate(zip(batch, emb.cpu().numpy())):
                    ids.append(f"{fname}_{i+j}_{uuid.uuid4().hex[:6]}")
                    vecs.append(v.tolist())
                    docs.append(c)
                    metas.append({"source": fname, "filepath": rel, "chunk_index": i+j, "total_chunks": len(chunks)})

            col.upsert(ids=ids, embeddings=vecs, documents=docs, metadatas=metas)
            total += len(chunks)

        except Exception as e:
            errors.append(f"{fname}: {e}")

    print(f"\n{'='*60}")
    print(f"完成! 成功: {len(pdf_files)-len(errors)}/{len(pdf_files)}, chunks: {col.count()}, dim: {dim}")
    if errors:
        for e in errors[:10]:
            print(f"  {e}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
