#!/usr/bin/env python3
"""
RAG 查询脚本 - BGE-M3 混合检索（Dense + Sparse + BM25 + RRF 融合）
用法: DASHSCOPE_API_KEY=sk-xxxx python scripts/rag_query.py "你的问题"
"""
import os
import sys
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Plus
from tqdm import tqdm
import jieba

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────
VECTORSTORE_DIR = Path("/home/kehua/storage1/rag/vectorstore")
COLLECTION_NAME = "health_knowledge_base"
MODEL_PATH = "/home/kehua/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"

BM25_INDEX_PATH = VECTORSTORE_DIR / "bm25_index.pkl"
CORPUS_PATH = VECTORSTORE_DIR / "corpus.pkl"
SPARSE_VEC_PATH = VECTORSTORE_DIR / "sparse_vecs.pkl"

# RRF 融合参数
RRF_K = 60


# ══════════════════════════════════════════════════════
# 1. BGE-M3 多向量提取
# ══════════════════════════════════════════════════════

def load_embedding_model():
    """加载 BGE-M3 embedding 模型"""
    print(f"加载 Embedding 模型: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16).cuda()
    model.eval()
    print(f"Embedding 维度: {model.config.hidden_size}")
    return tokenizer, model


def encode_bge_m3(query: str, tokenizer, model):
    """提取 BGE-M3 三种向量：dense + sparse + ColBERT 多向量"""
    inputs = tokenizer(
        [query],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        return_attention_mask=True,
    )
    inputs = {k: v.cuda() for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        last_hidden = outputs.last_hidden_state[0]  # (seq_len, hidden)

    # ── Dense: CLS 向量，L2 归一化 ──
    dense_vec = last_hidden[0]
    dense_vec = torch.nn.functional.normalize(dense_vec.unsqueeze(0), p=2, dim=1)[0]

    # ── Sparse: 基于 attention mask 和 token importance ──
    seq_len = input_ids.shape[1]
    token_importance = torch.zeros(seq_len, device="cuda")
    for layer_idx in range(model.config.num_hidden_layers):
        attn_weight = outputs.attentions[layer_idx][0].mean(dim=0)  # (seq_len, seq_len)
        token_importance += attn_weight.sum(dim=1)  # 每个 token 收到的注意力之和

    token_importance = token_importance[1:-1]  # 去掉 [CLS] 和 [SEP]
    ids = input_ids[0, 1:-1].cpu().tolist()
    sparse_vec = {}
    for idx, imp in zip(ids, token_importance.cpu().tolist()):
        if imp > 0:
            sparse_vec[idx] = imp

    # ── ColBERT: 所有 token 向量（L2 归一化） ──
    colbert_vecs = torch.nn.functional.normalize(last_hidden, p=2, dim=1)

    return dense_vec, sparse_vec, colbert_vecs, inputs["input_ids"][0].cpu().tolist()


def dense_score(query_dense: torch.Tensor, doc_dense: np.ndarray) -> np.ndarray:
    """余弦相似度（ChromaDB 已按 cosine 存储，这里直接计算）"""
    scores = np.dot(doc_dense, query_dense.cpu().numpy())
    return scores.astype(np.float32)


def sparse_score(query_sparse: dict, doc_sparse: dict) -> float:
    """两个稀疏向量的 Jaccard 相似度"""
    if not query_sparse or not doc_sparse:
        return 0.0
    q_keys = set(query_sparse.keys())
    d_keys = set(doc_sparse.keys())
    intersection = q_keys & d_keys
    if not intersection:
        return 0.0
    union = q_keys | d_keys
    score = sum(min(query_sparse[k], doc_sparse[k]) for k in intersection)
    return score / len(union)


def colbert_score(query_colbert: torch.Tensor, doc_colbert: torch.Tensor) -> float:
    """ColBERT 晚期交互：MaxSim"""
    # query_colbert: (Q, hidden), doc_colbert: (D, hidden)
    # 每个 query token 与所有 doc token 计算余弦，取最大值后求和
    sim = torch.matmul(query_colbert, doc_colbert.T)  # (Q, D)
    max_sim_per_query = sim.max(dim=1).values  # (Q,)
    return max_sim_per_query.sum().item()


def reciprocal_rank_fusion(results_list: list[dict], k: int = 60) -> list:
    """
    RRF (Reciprocal Rank Fusion) 多路召回融合
    results_list: [{"id": str, "score": float}, ...] 列表，每个元素是一路检索结果
    返回按 RRF 分数排序的 id 列表
    """
    rrf_scores = {}
    for results in results_list:
        for rank, item in enumerate(results):
            doc_id = item["id"]
            rrf = 1.0 / (k + rank + 1)
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {"rrf": 0.0, "dense": 0.0, "sparse": 0.0, "bm25": 0.0}
            rrf_scores[doc_id]["rrf"] += rrf

    # 按 RRF 分数排序
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x]["rrf"], reverse=True)
    for doc_id in sorted_ids:
        rrf_scores[doc_id]["id"] = doc_id
    return [rrf_scores[doc_id] for doc_id in sorted_ids]


# ══════════════════════════════════════════════════════
# 2. 加载索引
# ══════════════════════════════════════════════════════

def load_bm25_and_corpus():
    """加载 BM25 索引和文档语料"""
    if not BM25_INDEX_PATH.exists() or not CORPUS_PATH.exists():
        return None, None
    print(f"加载 BM25 索引: {BM25_INDEX_PATH}")
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25 = pickle.load(f)
    with open(CORPUS_PATH, "rb") as f:
        corpus = pickle.load(f)
    return bm25, corpus


def load_sparse_vectors():
    """加载预计算的稀疏向量"""
    if not SPARSE_VEC_PATH.exists():
        return None
    print(f"加载 Sparse 向量: {SPARSE_VEC_PATH}")
    with open(SPARSE_VEC_PATH, "rb") as f:
        return pickle.load(f)


def load_chroma():
    """加载 ChromaDB"""
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR), settings=Settings(allow_reset=False))
    col = client.get_or_create_collection(COLLECTION_NAME)
    print(f"向量库统计:")
    print(f"  Collection: {col.name}")
    print(f"  总 chunks: {col.count()}")
    return col


# ══════════════════════════════════════════════════════
# 3. 混合检索
# ══════════════════════════════════════════════════════

def search_dense(query_dense: torch.Tensor, col, top_k: int = 50):
    """Dense 检索：ChromaDB HNSW"""
    results = col.query(
        query_embeddings=[query_dense.cpu().tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "embeddings", "distances"]
    )
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    # ChromaDB cosine 距离转相似度: sim = 1 - dist/2
    dists = results["distances"][0]
    embs = results["embeddings"][0]
    scores = [1.0 - d / 2.0 for d in dists]

    return [
        {"id": ids[i], "doc": docs[i], "meta": metas[i], "dense_score": scores[i], "embedding": embs[i]}
        for i in range(len(ids))
    ]


def search_bm25(query: str, bm25, corpus: dict, top_k: int = 50):
    """BM25 关键词检索"""
    if bm25 is None or corpus is None:
        return []
    tokenized_query = list(jieba.cut(query))
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "id": corpus["ids"][idx],
                "doc": corpus["docs"][idx],
                "bm25_score": float(scores[idx])
            })
    return results


def search_sparse(query_sparse: dict, sparse_vecs: dict, corpus: dict, top_k: int = 50):
    """Sparse 检索：与预计算的文档稀疏向量做 Jaccard"""
    if sparse_vecs is None:
        return []
    results = []
    for idx, doc_sparse in enumerate(sparse_vecs):
        score = sparse_score(query_sparse, doc_sparse)
        if score > 0:
            results.append({
                "id": corpus["ids"][idx],
                "doc": corpus["docs"][idx],
                "sparse_score": score
            })
    results.sort(key=lambda x: x["sparse_score"], reverse=True)
    return results[:top_k]


def search_colbert(query_colbert: torch.Tensor, top_candidates: list, col, corpus: dict, sparse_vecs: dict):
    """ColBERT 交互检索：对 top 候选做精细化重排"""
    if not top_candidates or not sparse_vecs:
        return []
    # 获取候选的 colbert 向量（需要在索引时预存，这里用 sparse_vecs 中间字段模拟）
    # 实际上 ColBERT 需要预存 token 向量，这里做占位：真实实现需要重建索引时保存 token embeddings
    # 改为用 Sparse + Dense 的加权融合作为 ColBERT proxy
    results = []
    for cand in top_candidates:
        doc_id = cand["id"]
        try:
            idx = corpus["ids"].index(doc_id)
        except ValueError:
            continue
        # proxy: dense * 0.5 + sparse * 0.5
        dense = cand.get("dense_score", 0)
        sparse = cand.get("sparse_score", 0)
        colbert_proxy = dense * 0.5 + sparse * 0.5
        results.append({
            "id": doc_id,
            "doc": cand["doc"],
            "meta": cand.get("meta"),
            "colbert_proxy": colbert_proxy,
            "dense_score": dense,
            "sparse_score": sparse,
            "bm25_score": cand.get("bm25_score", 0),
        })
    results.sort(key=lambda x: x["colbert_proxy"], reverse=True)
    return results


def compare_search(query: str, tokenizer, model, col, bm25, corpus, sparse_vecs, top_k: int = 10):
    """
    对比模式：分别执行四路检索并对比结果
    """
    print(f"\n{'=' * 60}")
    print(f"对比模式: {query}")
    print(f"{'=' * 60}\n")

    t_q = time.perf_counter()
    query_dense, query_sparse, query_colbert, query_token_ids = encode_bge_m3(query, tokenizer, model)

    # Dense
    t0 = time.perf_counter()
    dense_results = search_dense(query_dense, col, top_k=top_k)
    t_dense = time.perf_counter() - t0

    # BM25
    t1 = time.perf_counter()
    bm25_results = search_bm25(query, bm25, corpus, top_k=top_k)
    t_bm25 = time.perf_counter() - t1

    # Sparse
    t2 = time.perf_counter()
    sparse_results = search_sparse(query_sparse, sparse_vecs, corpus, top_k=top_k)
    t_sparse = time.perf_counter() - t2

    # Hybrid (RRF)
    t3 = time.perf_counter()
    hybrid_results, _ = hybrid_search(
        query, tokenizer, model, col, bm25, corpus, sparse_vecs, top_k=top_k
    )
    t_hybrid = time.perf_counter() - t3

    total_t = time.perf_counter() - t_q

    print(f"{'─' * 60}")
    print(f"{'排名':<4} {'Dense':<36} {'BM25':<36} {'Sparse':<36} {'混合':<36}")
    print(f"{'─' * 60}")

    def short_doc(doc, length=14):
        s = doc.replace("\n", " ").strip()
        return s[:length] + "..." if len(s) > length else s

    def fmt_score(score):
        if score is None:
            return "N/A"
        return f"{score:.4f}"

    def get_id(results, i):
        if i < len(results):
            return results[i]["id"]
        return "-"

    def get_score(results, i, key):
        if i < len(results):
            s = results[i].get(key, 0)
            return f"{s:.4f}" if s > 0 else "N/A"
        return "-"

    # 对齐所有路的结果数量
    max_rank = max(
        len(dense_results), len(bm25_results),
        len(sparse_results), len(hybrid_results)
    )

    for i in range(max_rank):
        d_id  = get_id(dense_results, i)
        b_id  = get_id(bm25_results, i)
        s_id  = get_id(sparse_results, i)
        h_id  = get_id(hybrid_results, i)
        d_sc  = get_score(dense_results, i, "dense_score")
        b_sc  = get_score(bm25_results, i, "bm25_score")
        sp_sc = get_score(sparse_results, i, "sparse_score")
        h_sc  = get_score(hybrid_results, i, "dense_score")

        mark_h = " ◀" if h_id in [d_id, b_id, s_id] else ""
        print(f"{i+1:<4} {d_id:<20} {d_sc:<15} {b_id:<20} {b_sc:<15} {s_id:<20} {sp_sc:<15} {h_id:<20} {h_sc:<15}{mark_h}")

    print(f"{'─' * 60}")
    print(f"Dense: {t_dense:.2f}s | BM25: {t_bm25:.2f}s | Sparse: {t_sparse:.2f}s | 混合: {t_hybrid:.2f}s | 总计: {total_t:.2f}s")

    # 展示详细内容
    for label, results in [("Dense", dense_results), ("BM25", bm25_results), ("Sparse", sparse_results), ("混合(RRF)", hybrid_results)]:
        print(f"\n{'=' * 40}")
        print(f"  {label} 检索结果 (top {len(results)})")
        print(f"{'=' * 40}")
        score_key = {
            "Dense": "dense_score", "BM25": "bm25_score",
            "Sparse": "sparse_score", "混合(RRF)": "dense_score"
        }[label]
        for i, r in enumerate(results):
            doc = r.get("doc") or (corpus["texts"][corpus["ids"].index(r["id"])] if corpus and r["id"] in corpus.get("ids", []) else "[内容缺失]")
            print(f"\n  [{i+1}] ID: {r['id']}  Score: {r.get(score_key, 0):.4f}")
            print(f"  内容: {doc[:300]}")

    return dense_results, bm25_results, sparse_results, hybrid_results


def hybrid_search(query: str, tokenizer, model, col, bm25, corpus, sparse_vecs, top_k: int = 20):
    """
    混合检索主函数：三路召回 + RRF 融合
    """
    t_q = time.perf_counter()

    # ── Step 1: 查询编码（一次性提取三种向量） ──
    query_dense, query_sparse, query_colbert, query_token_ids = encode_bge_m3(query, tokenizer, model)
    t_encode = time.perf_counter() - t_q
    print(f"  向量编码: {t_encode:.2f}s")

    # ── Step 2: 三路并行检索 ──
    t0 = time.perf_counter()
    # 2a. Dense (ChromaDB, top 50)
    dense_results = search_dense(query_dense, col, top_k=50)
    t_dense = time.perf_counter() - t0
    print(f"  Dense 检索: {t_dense:.2f}s ({len(dense_results)} 结果)")

    t1 = time.perf_counter()
    # 2b. BM25 (top 50)
    bm25_results = search_bm25(query, bm25, corpus, top_k=50)
    t_bm25 = time.perf_counter() - t1
    print(f"  BM25 检索: {t_bm25:.2f}s ({len(bm25_results)} 结果)")

    t2 = time.perf_counter()
    # 2c. Sparse (top 50)
    sparse_results = search_sparse(query_sparse, sparse_vecs, corpus, top_k=50)
    t_sparse = time.perf_counter() - t2
    print(f"  Sparse 检索: {t_sparse:.2f}s ({len(sparse_results)} 结果)")

    # ── Step 3: 合并候选集（取 Dense 和 BM25 前 100 的并集） ──
    all_candidate_ids = set()
    for r in dense_results[:100]:
        all_candidate_ids.add(r["id"])
    for r in bm25_results[:100]:
        all_candidate_ids.add(r["id"])
    for r in sparse_results[:100]:
        all_candidate_ids.add(r["id"])

    # ── Step 4: 构建候选详情 ──
    candidate_map = {}
    for r in dense_results:
        if r["id"] in all_candidate_ids:
            candidate_map[r["id"]] = {
                "id": r["id"],
                "doc": r["doc"],
                "meta": r.get("meta"),
                "dense_score": r.get("dense_score", 0),
                "sparse_score": 0.0,
                "bm25_score": 0.0,
            }
    for r in sparse_results:
        if r["id"] in all_candidate_ids and r["id"] in candidate_map:
            candidate_map[r["id"]]["sparse_score"] = r.get("sparse_score", 0)
    for r in bm25_results:
        if r["id"] in all_candidate_ids and r["id"] in candidate_map:
            candidate_map[r["id"]]["bm25_score"] = r.get("bm25_score", 0)

    # ── Step 5: 准备 RRF 三路输入 ──
    # Dense 排名
    dense_ranked = [
        {"id": r["id"], "score": r.get("dense_score", 0)}
        for r in dense_results
    ]
    # Sparse 排名
    sparse_ranked = [
        {"id": r["id"], "score": r.get("sparse_score", 0)}
        for r in sparse_results
    ]
    # BM25 排名
    bm25_ranked = [
        {"id": r["id"], "score": r.get("bm25_score", 0)}
        for r in bm25_results
    ]

    # ── Step 6: RRF 融合 ──
    fused = reciprocal_rank_fusion([dense_ranked, sparse_ranked, bm25_ranked], k=RRF_K)

    # 合并分数
    for item in fused:
        cid = item["id"]
        if cid in candidate_map:
            item["doc"] = candidate_map[cid]["doc"]
            item["meta"] = candidate_map[cid].get("meta")
            item["dense_score"] = candidate_map[cid].get("dense_score", 0)
            item["sparse_score"] = candidate_map[cid].get("sparse_score", 0)
            item["bm25_score"] = candidate_map[cid].get("bm25_score", 0)

    t_rrf = time.perf_counter() - t_q
    print(f"  融合耗时: {t_rrf:.2f}s")

    return fused[:top_k], t_encode


def format_context(results: list, top_k: int = 5, corpus: dict = None) -> str:
    """格式化检索结果为上下文"""
    context_parts = []
    for i, item in enumerate(results[:top_k]):
        doc = item.get("doc")
        #兜底：尝试从 corpus 中取
        if not doc and corpus and item["id"] in corpus.get("ids", []):
            idx = corpus["ids"].index(item["id"])
            doc = corpus["texts"][idx]
        if not doc:
            doc = "[内容缺失]"

        meta = item.get("meta") or {}
        source = meta.get("source", "未知")
        context_parts.append(
            f"[来源 {i+1}] {source}\n"
            f"Dense: {item.get('dense_score', 0):.4f} | "
            f"Sparse: {item.get('sparse_score', 0):.4f} | "
            f"BM25: {item.get('bm25_score', 0):.4f}\n"
            f"{doc}"
        )
    return "\n\n---\n\n".join(context_parts)


# ══════════════════════════════════════════════════════
# 4. LLM 调用
# ══════════════════════════════════════════════════════

def call_qwen_api(question: str, context: str, api_key: str, model_name: str = "qwen-plus") -> str:
    """调用通义千问 API"""
    try:
        import openai
    except ImportError:
        print("错误: 请安装 openai 包: pip install openai")
        sys.exit(1)

    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    system_prompt = """你是一个专业的医疗指南助手。根据提供的上下文信息回答用户的问题。

要求:
1. 优先根据上下文中的信息回答，适当引用信息来源
2. 如果上下文中没有相关信息或信息不足，请结合你自身的医学知识进行补充回答
3. 回答要专业、准确
"""

    user_prompt = f"""上下文信息:
{context}

用户问题: {question}

请根据上述上下文信息回答用户的问题。"""

    print(f"\n正在调用通义千问 API ({model_name})...")
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=2000
    )

    return response.choices[0].message.content


# ══════════════════════════════════════════════════════
# 5. 主流程
# ══════════════════════════════════════════════════════

def run_query(question: str, tokenizer, model, col, bm25, corpus, sparse_vecs):
    """执行一次完整的混合检索 RAG 查询"""
    print(f"\n{'=' * 60}")
    print(f"问题: {question}")
    print(f"{'=' * 60}")

    # 检索
    t0 = time.perf_counter()
    results, t_encode = hybrid_search(question, tokenizer, model, col, bm25, corpus, sparse_vecs, top_k=10)
    t_search = time.perf_counter() - t0

    # 检查是否命中（用于提示，不影响是否回答）
    has_results = len(results) > 0 and (results[0].get("dense_score", 0) > 0.1 or results[0].get("bm25_score", 0) > 1.0)

    # 格式化
    context = format_context(results, corpus=corpus)
    print(f"\n检索到 {len(results)} 条相关结果（融合后）")

    # 显示结果详情
    print("\n" + "=" * 60)
    print("检索到的相关内容:")
    print("=" * 60)
    print(context)

    # LLM 回答
    print("\n" + "=" * 60)
    print("LLM 回答:")
    print("=" * 60)
    t2 = time.perf_counter()
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    # 始终传入上下文；知识库无结果时追加说明，提示大模型用自身能力补充
    if not has_results:
        context_with_note = (
            "【说明】知识库中未检索到与该问题直接相关的内容，"
            "请结合你自身的医学知识进行专业、准确的回答。\n\n"
            f"已检索的内容（相关性较低，仅供参考）：\n{context}"
        )
        answer = call_qwen_api(question, context_with_note, api_key)
    else:
        answer = call_qwen_api(question, context, api_key)
    t_api = time.perf_counter() - t2
    print(answer)

    # 耗时统计
    t_total = t_search + t_api
    print(f"\n{'─' * 40}")
    print(f"耗时统计:")
    print(f"  编码+检索: {t_search:>7.2f}s  ({t_search/t_total*100:>5.1f}%)")
    print(f"  LLM API:   {t_api:>7.2f}s  ({t_api/t_total*100:>5.1f}%)")
    print(f"  {'总计':>7}:  {t_total:>7.2f}s")
    print(f"{'─' * 40}")

    return answer


def build_sparse_vectors_for_existing(col, corpus: dict, tokenizer, model):
    """
    为现有 ChromaDB 索引中的文档批量计算 BGE-M3 Sparse 向量
    避免重建索引，只追加计算稀疏向量
    """
    print(f"\n正在为现有 {col.count()} 条文档计算 Sparse 向量...")
    print("（此过程只需执行一次，后续查询直接加载）")

    sparse_vecs = []
    batch_size = 32
    ids = corpus["ids"]
    docs = corpus["docs"]

    for i in tqdm(range(0, len(docs), batch_size)):
        batch = docs[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        batch_sparse = []
        for j in range(len(batch)):
            input_ids = inputs["input_ids"][j]
            attn_mask = inputs["attention_mask"][j]
            seq_len = attn_mask.sum().item()

            # 多层 attention 累加（attention 在 GPU 上）
            pad_seq_len = input_ids.size(0)
            token_importance = torch.zeros(pad_seq_len, device="cuda")
            for layer_idx in range(model.config.num_hidden_layers):
                attn = outputs.attentions[layer_idx][j].mean(dim=0)  # [pad_seq_len, pad_seq_len]
                token_importance += attn.sum(dim=1)

            ids_tokens = input_ids[1:seq_len-1].cpu().tolist()
            importances = token_importance[1:seq_len-1].cpu().tolist()

            sparse = {}
            for tid, imp in zip(ids_tokens, importances):
                if imp > 0:
                    sparse[tid] = imp
            batch_sparse.append(sparse)

        sparse_vecs.extend(batch_sparse)

    # 保存
    with open(SPARSE_VEC_PATH, "wb") as f:
        pickle.dump(sparse_vecs, f)
    print(f"Sparse 向量已保存到: {SPARSE_VEC_PATH} ({len(sparse_vecs)} 条)")
    return sparse_vecs


def main():
    parser = argparse.ArgumentParser(description="RAG 混合检索查询")
    parser.add_argument("question", help="要查询的问题", nargs="?")
    parser.add_argument("--no-loop", action="store_true", help="单次查询后退出")
    parser.add_argument("--rebuild-sparse", action="store_true", help="重新计算 Sparse 向量")
    parser.add_argument("--bm25-only", action="store_true", help="仅用 BM25 检索（调试用）")
    parser.add_argument("--dense-only", action="store_true", help="仅用 Dense 检索（调试用）")
    parser.add_argument("--compare", action="store_true", help="对比模式：三路 + 混合同时展示")

    args = parser.parse_args()

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误: 请设置 DASHSCOPE_API_KEY 环境变量")
        print("用法: DASHSCOPE_API_KEY=sk-xxxx python scripts/rag_query.py \"你的问题\"")
        sys.exit(1)

    print(f"{'=' * 60}")
    print("RAG 医疗指南问答系统 (BGE-M3 混合检索)")
    print(f"{'=' * 60}")

    # 加载模型
    tokenizer, model = load_embedding_model()

    # 加载向量库
    col = load_chroma()

    # 加载 BM25
    bm25, corpus = load_bm25_and_corpus()
    if bm25 is None:
        print("警告: BM25 索引不存在，请先运行 build_index.py 重建索引")
        bm25 = None
        corpus = None

    # 加载 Sparse 向量（可选，有则加载，无则跳过）
    sparse_vecs = None
    if args.rebuild_sparse:
        if corpus is None:
            print("错误: 需要先有 corpus.pkl，请运行 build_bm25_index.py")
            sys.exit(1)
        print(f"\n正在为现有 {col.count()} 条文档计算 Sparse 向量（GPU）...")
        print("预计耗时 3-5 分钟，此过程只需执行一次，后续查询直接加载")
        sparse_vecs = build_sparse_vectors_for_existing(col, corpus, tokenizer, model)
    elif SPARSE_VEC_PATH.exists():
        sparse_vecs = load_sparse_vectors()
        if sparse_vecs is not None:
            print(f"Sparse 向量: {len(sparse_vecs)} 条已加载")
        else:
            print("警告: Sparse 向量加载失败，将使用 Dense+BM25 两路检索")
    else:
        print("Sparse 向量未建立（跳过），使用 Dense+BM25 两路检索")
        print("如需三路检索，运行: python scripts/rag_query.py --rebuild-sparse ...")

    if args.compare:
        question = args.question or "高血压的诊断标准是什么"
        compare_search(question, tokenizer, model, col, bm25, corpus, sparse_vecs)
        return

    # 调试模式
    if args.bm25_only:
        print("\n[调试模式] 仅使用 BM25 检索")
        question = args.question or "高血压的诊断标准是什么"
        results = search_bm25(question, bm25, corpus, top_k=5)
        for i, r in enumerate(results):
            print(f"\n[BM25 {i+1}] {r['id']}")
            print(f"  Score: {r['bm25_score']:.4f}")
            print(f"  内容: {r['doc'][:200]}...")
        return

    if args.dense_only:
        print("\n[调试模式] 仅使用 Dense 检索")
        question = args.question or "高血压的诊断标准是什么"
        query_dense, _, _, _ = encode_bge_m3(question, tokenizer, model)
        results = search_dense(query_dense, col, top_k=5)
        for i, r in enumerate(results):
            print(f"\n[Dense {i+1}] {r['id']}")
            print(f"  Score: {r['dense_score']:.4f}")
            print(f"  内容: {r['doc'][:200]}...")
        return

    if args.question:
        # 单次查询
        run_query(args.question, tokenizer, model, col, bm25, corpus, sparse_vecs)
    else:
        # 交互式循环
        print("\n请输入问题（输入 q 或 quit 退出）:\n")
        while True:
            try:
                question = input("> ").strip()
                if question.lower() in ("q", "quit", "exit"):
                    print("再见!")
                    break
                if not question:
                    continue
                run_query(question, tokenizer, model, col, bm25, corpus, sparse_vecs)
                print()
            except (KeyboardInterrupt, EOFError):
                print("\n再见!")
                break


if __name__ == "__main__":
    main()
