#!/usr/bin/env python3
"""
医疗指南 RAG 索引构建脚本 - 8卡并行版
每个进程绑定一个 GPU，各自处理一批 PDF，汇总到 ChromaDB
"""

import os
import sys
import uuid
import signal
import hashlib
import pickle
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Plus

import pypdfium2
from paddleocr import PaddleOCR

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────
DATA_DIR = Path("/home/kehua/storage0/rag/data")
VECTORSTORE_DIR = Path("/home/kehua/storage0/rag/vectorstore")
COLLECTION_NAME = "health_knowledge_base"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
EMBEDDING_MODEL_PATH = "/home/kehua/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"

BM25_INDEX_PATH = Path("/home/kehua/storage1/rag/vectorstore/bm25_index.pkl")
CORPUS_PATH = Path("/home/kehua/storage1/rag/vectorstore/corpus.pkl")

SUPPORTED_EXTS = {".pdf"}


# ──────────────────────────────────────────────
# 每个 GPU 进程的工作函数
# ──────────────────────────────────────────────
def worker_main(gpu_id: int, pdf_files: list[Path]):
    """在指定 GPU 上处理一批 PDF"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    print(f"[Worker-{gpu_id}] 启动，处理 {len(pdf_files)} 个 PDF")

    # ── Embedding 模型（每个进程独占一个 GPU） ──
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        EMBEDDING_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).cuda()
    model.eval()

    # ── ChromaDB client（每个进程独立连接） ──
    client = chromadb.PersistentClient(
        path=str(VECTORSTORE_DIR),
        settings=Settings(allow_reset=False),
    )
    col = client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", " "],
        length_function=len,
    )

    total_chunks = 0
    errors = 0

    pending_ids_all, pending_docs_all = [], []

    for pdf_path in tqdm(pdf_files, desc=f"GPU{gpu_id}", leave=False):
        try:
            # ── 文字提取 ──
            pdf = pypdfium2.PdfDocument(str(pdf_path))
            all_text = []
            needs_ocr = False

            for page_idx in range(len(pdf)):
                page = pdf[page_idx]
                tp = page.get_textpage()
                text = tp.get_text_bounded()
                if text and text.strip():
                    all_text.append(text)
                else:
                    needs_ocr = True
            pdf.close()

            text = "\n\n".join(all_text)

            # 无文字 → 全页 OCR
            if not text or len(text.strip()) < 20 or needs_ocr:
                if needs_ocr and (text and text.strip()):
                    pass  # 已有文字就用已有的
                elif not text or len(text.strip()) < 20:
                    ocr = PaddleOCR(lang="ch", device="metax_gpu", use_textline_orientation=True)
                    pdf = pypdfium2.PdfDocument(str(pdf_path))
                    all_ocr = []
                    for page_idx in range(len(pdf)):
                        page = pdf[page_idx]
                        bitmap = page.render(scale=2.0, rotation=0)
                        pil_img = bitmap.to_pil()
                        img_np = np.array(pil_img)
                        res = ocr.predict(img_np)
                        for block in res:
                            for line in (block or []):
                                if line and len(line) >= 2:
                                    txt = line[1] if isinstance(line[1], str) else str(line[1])
                                    all_ocr.append(txt)
                    pdf.close()
                    text = "\n".join(all_ocr)
                    if not text.strip():
                        errors += 1
                        print(f"[Worker-{gpu_id}] [跳过] {pdf_path.name}: 无法提取文字")
                        continue

            if not text.strip():
                errors += 1
                continue

            # ── 分块 ──
            chunks = splitter.split_text(text)
            filename = pdf_path.name
            relative = str(pdf_path.relative_to(DATA_DIR.parent))

            # ── Embedding 批量 ──
            batch_size = 32
            pending_ids, pending_vecs, pending_docs, pending_metas = [], [], [], []

            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i : i + batch_size]
                inputs = tokenizer(
                    batch_chunks,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                inputs = {k: v.cuda() for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                    emb = outputs.last_hidden_state[:, 0, :]
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                emb_np = emb.cpu().numpy()

                for j, (chunk, vector) in enumerate(zip(batch_chunks, emb_np)):
                    chunk_id = f"{filename}_{i+j}_{uuid.uuid4().hex[:6]}"
                    meta = {
                        "source": filename,
                        "filepath": relative,
                        "chunk_index": i + j,
                        "total_chunks": len(chunks),
                        "gpu_id": gpu_id,
                    }
                    pending_ids.append(chunk_id)
                    pending_vecs.append(vector.tolist())
                    pending_docs.append(chunk)
                    pending_metas.append(meta)
                    pending_ids_all.append(chunk_id)
                    pending_docs_all.append(chunk)

                # 每 500 条写入一次
                if len(pending_ids) >= 500:
                    col.upsert(ids=pending_ids, embeddings=pending_vecs, documents=pending_docs, metadatas=pending_metas)
                    pending_ids, pending_vecs, pending_docs, pending_metas = [], [], [], []

            if pending_ids:
                col.upsert(ids=pending_ids, embeddings=pending_vecs, documents=pending_docs, metadatas=pending_metas)

            total_chunks += len(chunks)

        except Exception as e:
            errors += 1
            print(f"[Worker-{gpu_id}] [错误] {pdf_path.name}: {e}")

    print(f"[Worker-{gpu_id}] 完成: {total_chunks} chunks, {errors} 错误")
    return total_chunks, errors, pending_ids_all, pending_docs_all


# ──────────────────────────────────────────────
# 主入口（分发任务到 8 个进程）
# ──────────────────────────────────────────────
def collect_pdfs(root: Path) -> list[Path]:
    pdfs = []
    for ext in SUPPORTED_EXTS:
        pdfs.extend(root.rglob(f"*{ext}"))
    return sorted(pdfs)


def main():
    import multiprocessing as mp

    print("=" * 60)
    print("医疗指南 RAG 索引构建 (8 GPU 并行)")
    print("=" * 60)

    pdf_files = collect_pdfs(DATA_DIR)
    print(f"\n发现 {len(pdf_files)} 个 PDF 文件")

    if not pdf_files:
        return

    # ── 均分到 8 个 GPU ──
    num_gpus = 8
    chunk_size = (len(pdf_files) + num_gpus - 1) // num_gpus
    gpu_batches = []
    for i in range(num_gpus):
        start = i * chunk_size
        end = min(start + chunk_size, len(pdf_files))
        batch = pdf_files[start:end]
        if batch:
            gpu_batches.append((i, batch))

    print(f"分配到 {len(gpu_batches)} 个 GPU")
    for gpu_id, batch in gpu_batches:
        print(f"  GPU {gpu_id}: {len(batch)} 个 PDF")

    # ── 启动 8 个进程 ──
    ctx = mp.get_context("spawn")
    processes = []
    for gpu_id, batch in gpu_batches:
        p = ctx.Process(target=worker_main, args=(gpu_id, batch))
        p.start()
        processes.append(p)
        print(f"启动 Worker on GPU {gpu_id} (PID {p.pid})")

    print(f"\n等待 {len(processes)} 个进程完成...")

    # ── 等待全部完成，收集结果 ──
    all_results = []
    for p in processes:
        p.join()
        result = p.exitcode  # 临时用 exitcode 传结果，后面改用其他方式
    # 实际上 multiprocessing spawn 无法通过 return 直接获取，需要用 shared dict 或 Queue
    # 这里改为 worker 写入文件，主进程汇总
    print(f"\n主进程汇总 BM25 索引...")

    # ── 从 ChromaDB 读取所有 chunk ID 和文本构建 BM25 ──
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR), settings=Settings(allow_reset=False))
    col = client.get_or_create_collection(COLLECTION_NAME)
    all_ids = []
    all_docs = []
    batch_size = 1000
    offset = 0
    while True:
        r = col.get(limit=batch_size, offset=offset, include=["documents"])
        if not r["ids"]:
            break
        all_ids.extend(r["ids"])
        all_docs.extend(r["documents"])
        offset += batch_size
        if len(r["ids"]) < batch_size:
            break

    print(f"从 ChromaDB 加载了 {len(all_ids)} 条文档，准备分词...")

    # ── 构建 BM25 索引 ──
    import jieba
    print("使用 jieba 分词...")
    tokenized_corpus = [list(jieba.cut(doc)) for doc in tqdm(all_docs, desc="分词")]
    bm25 = BM25Plus(tokenized_corpus)
    print("BM25 索引构建完成！")

    # ── 保存 BM25 索引 ──
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    with open(CORPUS_PATH, "wb") as f:
        pickle.dump({"ids": all_ids, "docs": all_docs}, f)

    print(f"BM25 索引已保存到: {BM25_INDEX_PATH}")
    print(f"Corpus 已保存到: {CORPUS_PATH}")

    print(f"\n{'=' * 60}")
    print(f"索引构建完成！")
    print(f"  总文件: {len(pdf_files)}")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  总 chunks: {col.count()}")
    print(f"  向量维度: 1024")
    print(f"  BM25 索引: {BM25_INDEX_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
