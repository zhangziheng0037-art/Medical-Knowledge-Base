#!/usr/bin/env python3
"""
从已有 ChromaDB Collection 构建 BM25 索引（不依赖 PDF）
用法: python scripts/build_bm25_index.py
"""
import pickle
from pathlib import Path
from tqdm import tqdm

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Plus
import jieba

VECTORSTORE_DIR = Path("/home/kehua/storage1/rag/vectorstore")
COLLECTION_NAME = "health_knowledge_base"
BM25_INDEX_PATH = VECTORSTORE_DIR / "bm25_index.pkl"
CORPUS_PATH = VECTORSTORE_DIR / "corpus.pkl"


def main():
    print("=" * 60)
    print("从 ChromaDB 构建 BM25 索引")
    print("=" * 60)

    # 连接 ChromaDB
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR), settings=Settings(allow_reset=False))
    col = client.get_or_create_collection(COLLECTION_NAME)
    total = col.count()
    print(f"Collection: {col.name}, 总 chunks: {total:,}")

    if total == 0:
        print("没有数据，退出")
        return

    # 批量读取所有文档
    all_ids = []
    all_docs = []
    batch_size = 1000
    offset = 0
    print(f"\n读取文档文本...")
    while True:
        r = col.get(limit=batch_size, offset=offset, include=["documents"])
        if not r["ids"]:
            break
        all_ids.extend(r["ids"])
        all_docs.extend(r["documents"])
        offset += batch_size
        print(f"  已读取 {len(all_ids):,} / {total:,}", end="\r")
        if len(r["ids"]) < batch_size:
            break
    print(f"\n读取完成: {len(all_ids):,} 条文档")

    # 分词
    print("\n使用 jieba 分词...")
    tokenized_corpus = []
    for doc in tqdm(all_docs, desc="分词"):
        tokenized_corpus.append(list(jieba.cut(doc)))

    # 构建 BM25
    print("\n构建 BM25 索引...")
    bm25 = BM25Plus(tokenized_corpus)

    # 保存
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    with open(CORPUS_PATH, "wb") as f:
        pickle.dump({"ids": all_ids, "docs": all_docs}, f)

    print(f"\n{'=' * 60}")
    print(f"BM25 索引构建完成！")
    print(f"  总文档: {len(all_ids):,}")
    print(f"  BM25 索引: {BM25_INDEX_PATH}")
    print(f"  Corpus: {CORPUS_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
