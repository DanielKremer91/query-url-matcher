from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import labels as L
from .chunk import chunk_words
from .embeddings.base import l2_normalize


@dataclass
class MatchResult:
    queries: list
    urls: list
    chunks: list  # je URL die Liste ihrer Chunks
    chunk_scores: np.ndarray
    full_scores: np.ndarray
    best_chunk_idx: np.ndarray
    full_method: list

    def combined(self, weight: float) -> np.ndarray:
        return weight * self.chunk_scores + (1 - weight) * self.full_scores

    def lead(self, basis: str, weight: float = 0.7) -> np.ndarray:
        if basis == "chunk":
            return self.chunk_scores
        if basis == "full":
            return self.full_scores
        if basis == "combined":
            return self.combined(weight)
        raise ValueError(f"Unbekannte Bewertungsgrundlage: {basis}")

    def best_chunk(self, i: int, j: int) -> str:
        return self.chunks[j][int(self.best_chunk_idx[i, j])]


def estimate_chunks(contents, chunk_size, chunk_overlap) -> int:
    return sum(len(chunk_words(c, chunk_size, chunk_overlap)) for c in contents)


def run_matching(queries, urls, contents, embedder, chunk_size, chunk_overlap) -> MatchResult:
    chunks = [chunk_words(c, chunk_size, chunk_overlap) for c in contents]
    flat = [chunk for per_url in chunks for chunk in per_url]
    offsets = np.cumsum([0] + [len(per_url) for per_url in chunks])

    q = embedder.embed(list(queries), "query")
    c = embedder.embed(flat, "passage")
    sims = q @ c.T

    n_q, n_u = len(queries), len(urls)
    chunk_scores = np.zeros((n_q, n_u), dtype=np.float32)
    best = np.zeros((n_q, n_u), dtype=int)
    full_vecs = np.zeros((n_u, c.shape[1]), dtype=np.float32)
    methods, as_fulltext = [], []
    for u in range(n_u):
        lo, hi = offsets[u], offsets[u + 1]
        block = sims[:, lo:hi]
        best[:, u] = block.argmax(axis=1)
        chunk_scores[:, u] = block.max(axis=1)
        if hi - lo == 1:
            full_vecs[u] = c[lo]
            methods.append(L.FULLTEXT)
        elif embedder.fits_context(contents[u]):
            as_fulltext.append(u)
            methods.append(L.FULLTEXT)
        else:
            full_vecs[u] = c[lo:hi].mean(axis=0)
            methods.append(L.CHUNK_MEAN)
    if as_fulltext:
        full_vecs[as_fulltext] = embedder.embed([" ".join(contents[u].split()) for u in as_fulltext], "passage")
    full_scores = q @ l2_normalize(full_vecs).T

    return MatchResult(list(queries), list(urls), chunks, chunk_scores, full_scores, best, methods)


def ranks(scores: np.ndarray) -> np.ndarray:
    return (-scores).argsort(axis=1, kind="stable").argsort(axis=1, kind="stable") + 1


def top_hits(result: MatchResult, basis: str = "chunk", weight: float = 0.7, top_n: int = 5) -> pd.DataFrame:
    combined = result.combined(weight)
    lead = result.lead(basis, weight)
    r_chunk, r_full, r_combi = ranks(result.chunk_scores), ranks(result.full_scores), ranks(combined)
    rows = []
    for i, query in enumerate(result.queries):
        for j in np.argsort(-lead[i], kind="stable")[:top_n]:
            rows.append(
                {
                    L.C_QUERY: query,
                    L.C_URL: result.urls[j],
                    L.C_CHUNK: result.best_chunk(i, j),
                    L.C_S_CHUNK: round(float(result.chunk_scores[i, j]), 4),
                    L.C_S_FULL: round(float(result.full_scores[i, j]), 4),
                    L.C_S_COMBI: round(float(combined[i, j]), 4),
                    L.C_R_CHUNK: int(r_chunk[i, j]),
                    L.C_R_FULL: int(r_full[i, j]),
                    L.C_R_COMBI: int(r_combi[i, j]),
                    L.C_METHOD: result.full_method[j],
                }
            )
    return pd.DataFrame(rows)
