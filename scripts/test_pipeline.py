#!/usr/bin/env python3
"""快速测试脚本：验证索引全流程（单个 PDF）"""
import io
from pathlib import Path

import numpy as np
import torch
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel

import pypdfium2
from paddleocr import PaddleOCR

DATA_DIR = Path("/home/kehua/storage0/rag/data")
TEST_PDF = DATA_DIR / "基层诊疗指南(2024.7)/基层诊疗指南(2024.7)/幽门螺杆菌感染基层诊疗指南（2019年）.pdf"
MODEL_PATH = "/home/kehua/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"

print(f"1. PDF 读取测试: {TEST_PDF.name}")
pdf = pypdfium2.PdfDocument(str(TEST_PDF))
print(f"   页数: {len(pdf)}")
page = pdf[0]
text_page = page.get_textpage()
t = text_page.get_text_bounded()
print(f"   第1页文字长度: {len(t)} 字符")
print(f"   第1页预览: {t[:200]!r}")
pdf.close()

print(f"\n2. 分块测试")
splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
chunks = splitter.split_text(t)
print(f"   分块数: {len(chunks)}")
for i, c in enumerate(chunks[:3]):
    print(f"   chunk[{i}]: {len(c)} 字符: {c[:100]!r}...")

print(f"\n3. Embedding 测试")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16).cuda()
model.eval()
inputs = tokenizer([t[:500]], return_tensors="pt", padding=True, truncation=True, max_length=512)
inputs = {k: v.cuda() for k, v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
    emb = outputs.last_hidden_state[:, 0, :]
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
print(f"   向量维度: {emb.shape[1]}")

print(f"\n4. ChromaDB 测试")
client = chromadb.PersistentClient(
    path=str(DATA_DIR.parent / "vectorstore"),
    settings=Settings(allow_reset=True),
)
col = client.get_or_create_collection("health_knowledge_base", metadata={"hnsw:space": "cosine"})
print(f"   Collection: {col.name}, count: {col.count()}")

print(f"\n5. 写入一条测试数据")
test_id = "test_single_chunk"
col.upsert(ids=[test_id], embeddings=[emb[0].cpu().tolist()], documents=[t[:500]], metadatas=[{"source": "test.pdf"}])
print(f"   写入完成，当前 count: {col.count()}")

print(f"\n6. 检索验证")
results = col.query(query_embeddings=[emb[0].cpu().tolist()], n_results=1)
print(f"   检索结果: {results['documents'][0][0][:100]!r}")

print(f"\n7. 清理测试数据")
col.delete(ids=[test_id])
print(f"   清理完成，count: {col.count()}")

print(f"\n✅ 全流程测试通过！")
