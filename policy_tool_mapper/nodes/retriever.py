"""
Retriever node — mode: retrieval.

Three-stage recall-optimised retrieval for each tool profile:
  1. BM25 (lexical)         — catches exact terminology matches
  2. Bi-encoder (OpenAI)    — semantic top-k by cosine similarity
  3. Cross-encoder reranker — joint (tool, sentence) scoring

BM25 + bi-encoder candidates are unioned (~top-50-60 per tool), then
re-ranked by the cross-encoder. The top `ce_top_k` pairs pass to the judge node.
False positives are killed by the LLM judge, not here.
"""

import asyncio
import re
from typing import Any

from langchain_core.runnables import RunnableConfig

from policy_tool_mapper.state import (
    PipelineState,
    PolicyStatement,
    RetrievalCandidate,
    ToolProfile,
)

# ── Defaults ──────────────────────────────────────────────────────────────────
_TOP_K_BM25    = 30
_TOP_K_BIENC   = 30
_CE_MODEL      = "BAAI/bge-reranker-v2-m3"
_CE_TOP_K      = 20
_EMBED_MODEL   = "text-embedding-3-small"   # best OpenAI embedding for cost/quality

_cross_encoder: Any = None   # cached after first load


def _load_cross_encoder(model: str):
    global _cross_encoder
    if _cross_encoder is None:
        try:
            import os
            # Disable Apple Accelerate multi-threaded BLAS — prevents EXC_ARM_DA_ALIGN
            # (unaligned SIMD access in cblas_sgemm on macOS ARM, input-size dependent)
            os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            from sentence_transformers import CrossEncoder
            print(f"[retriever] Loading cross-encoder: {model} (device=cpu)")
            _cross_encoder = CrossEncoder(model, device="cpu")
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for retrieval mode.\n"
                "Install: pip install 'policy-tool-mapper[retrieval]'"
            ) from exc
    return _cross_encoder


# ── BM25 ──────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


def _bm25_top_k(
    query: str,
    statements: list[PolicyStatement],
    k: int,
) -> list[str]:
    try:
        import numpy as np
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise ImportError(
            "rank-bm25 and numpy are required for retrieval mode.\n"
            "Install: pip install 'policy-tool-mapper[retrieval]'"
        ) from exc

    corpus = [_tokenize(s.text) for s in statements]
    bm25   = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query))
    idx    = np.argsort(scores)[::-1][:k]
    return [statements[i].id for i in idx if scores[i] > 0]


# ── Bi-encoder ────────────────────────────────────────────────────────────────

def _cosine_top_k(
    query_emb:   list[float],
    stmt_embs:   dict[str, list[float]],
    all_ids:     list[str],
    k:           int,
) -> list[str]:
    import numpy as np
    q = np.array(query_emb, dtype=np.float32)
    q /= np.linalg.norm(q) + 1e-10
    scores: dict[str, float] = {}
    for sid in all_ids:
        v = np.array(stmt_embs[sid], dtype=np.float32)
        v /= np.linalg.norm(v) + 1e-10
        scores[sid] = float(np.dot(q, v))
    return sorted(scores, key=scores.__getitem__, reverse=True)[:k]


# ── Cross-encoder ─────────────────────────────────────────────────────────────

def _cross_encode_sync(
    tool_text:  str,
    candidates: list[str],
    stmt_by_id: dict[str, PolicyStatement],
    ce_model:   str,
    top_k:      int,
) -> list[str]:
    if not candidates:
        return []
    ce     = _load_cross_encoder(ce_model)
    pairs  = [(tool_text, stmt_by_id[sid].text) for sid in candidates]
    scores = ce.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [sid for sid, _score in ranked[:top_k]]


# ── Node ──────────────────────────────────────────────────────────────────────

async def retriever_node(state: PipelineState, config: RunnableConfig) -> dict:
    cfg          = config.get("configurable", {})
    embed_model  = cfg.get("embed_model",  _EMBED_MODEL)
    ce_model     = cfg.get("ce_model",     _CE_MODEL)
    ce_top_k     = cfg.get("ce_top_k",     _CE_TOP_K)

    statements: list[PolicyStatement] = state["policy_statements"]
    profiles:   list[ToolProfile]     = state["tool_profiles"]
    stmt_by_id  = {s.id: s for s in statements}
    all_ids     = [s.id for s in statements]

    # ── Embed everything with OpenAI bi-encoder ───────────────────────────────
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:
        raise ImportError(
            "langchain-openai is required for retrieval mode.\n"
            "Install: pip install 'policy-tool-mapper[openai]'"
        ) from exc

    embedder = OpenAIEmbeddings(model=embed_model)

    print(f"[retriever] Embedding {len(statements)} statements with {embed_model} ...")
    stmt_embs_list = await embedder.aembed_documents([s.text for s in statements])
    stmt_embs      = {s.id: emb for s, emb in zip(statements, stmt_embs_list)}

    profile_texts = [
        f"{p.name}: {p.description}\n{p.semantic_profile}" for p in profiles
    ]
    print(f"[retriever] Embedding {len(profiles)} tool profiles ...")
    profile_embs_list = await embedder.aembed_documents(profile_texts)
    profile_embs      = {p.tool_id: emb for p, emb in zip(profiles, profile_embs_list)}

    # ── Per-tool: BM25 ∪ bi-encoder → cross-encoder rerank ───────────────────
    candidates: list[RetrievalCandidate] = []

    for profile in profiles:
        tool_text = f"{profile.name}: {profile.description}\n{profile.semantic_profile}"

        bm25_ids  = _bm25_top_k(tool_text, statements, _TOP_K_BM25)
        bienc_ids = _cosine_top_k(profile_embs[profile.tool_id], stmt_embs, all_ids, _TOP_K_BIENC)

        # Union, preserving order (BM25 first, then bi-encoder extras)
        seen: set[str] = set()
        union_ids: list[str] = []
        for sid in bm25_ids + bienc_ids:
            if sid not in seen:
                seen.add(sid)
                union_ids.append(sid)

        # Cross-encoder reranking — skipped when ce_model is "none"
        if ce_model and ce_model.lower() != "none":
            kept = _cross_encode_sync(tool_text, union_ids, stmt_by_id, ce_model, ce_top_k)
        else:
            kept = union_ids   # pass full union to judge; judge handles precision

        print(
            f"[retriever] {profile.tool_id}: "
            f"BM25={len(bm25_ids)} bienc={len(bienc_ids)} "
            f"union={len(union_ids)} → CE top-k kept={len(kept)}"
        )
        candidates.append(RetrievalCandidate(tool_id=profile.tool_id, statement_ids=kept))

    total_cands = sum(len(c.statement_ids) for c in candidates)
    print(f"[retriever] Total candidates after cross-encoder: {total_cands}")
    return {"retrieval_candidates": candidates}
