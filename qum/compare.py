"""Modellvergleich an einer Wahrheitsliste: Wie oft landet die erwartete URL vorn?

Aufruf: python -m qum.compare --truth wahrheit.csv --content frog.csv
"""

import argparse
import dataclasses
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .embeddings import make_embedder
from .ingest import KEYWORD_ALIASES, URL_ALIASES, IngestError, find_column, load_content, read_table
from .match import ranks, run_matching
from .models import ModelSpec, get_model
from .normalize import normalize_url

# Name -> (Modellschlüssel, Abweichungen von der Modelldefinition)
CONFIGS = {
    "paraphrase": ("paraphrase-mpnet", {}),
    "e5": ("e5-large", {}),
    "e5-ohne-prefix": ("e5-large", {"query_prefix": "", "passage_prefix": ""}),  # e5 nicht so angesteuert wie trainiert
    "openai": ("openai", {}),
    "openai-prefix": ("openai", {"query_prefix": "query: ", "passage_prefix": "passage: "}),
}


def spec_for(config: str) -> ModelSpec:
    key, overrides = CONFIGS[config]
    return dataclasses.replace(get_model(key), key=config, **overrides)


def evaluate(truth, urls, contents, embedder, chunk_size, chunk_overlap, basis="chunk", weight=0.7) -> dict:
    u_index = {normalize_url(u): j for j, u in enumerate(urls)}
    pairs = [(query, u_index[normalize_url(url)]) for query, url in truth if normalize_url(url) in u_index]
    if not pairs:
        raise ValueError("Keine URL der Wahrheitsliste steht im Frog-Export.")
    result = run_matching([query for query, _ in pairs], urls, contents, embedder, chunk_size, chunk_overlap)
    rank = ranks(result.lead(basis, weight))
    positions = np.array([rank[i, j] for i, (_, j) in enumerate(pairs)])
    return {
        "n": len(pairs),
        "fehlend": len(truth) - len(pairs),
        "hit1": float((positions == 1).mean()),
        "hit3": float((positions <= 3).mean()),
        "mrr": float((1 / positions).mean()),
    }


def _load_truth(path: Path) -> list:
    df = read_table(path.read_bytes(), path.name)
    kcol, ucol = find_column(df, KEYWORD_ALIASES), find_column(df, URL_ALIASES)
    if kcol is None or ucol is None:
        raise IngestError(f"Wahrheitsliste braucht eine Query- und eine URL-Spalte. Gefunden: {list(df.columns)}")
    df = df.dropna(subset=[kcol, ucol])
    return [(str(q).strip(), str(u).strip()) for q, u in zip(df[kcol], df[ucol])]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Modellvergleich an einer Wahrheitsliste")
    parser.add_argument("--truth", required=True, type=Path, help="CSV/XLSX mit Query und erwarteter URL")
    parser.add_argument("--content", required=True, type=Path, help="Frog-Export mit URL und Main Content")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("--basis", default="chunk", choices=["chunk", "full", "combined"])
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    parser.add_argument("--out", type=Path, default=Path("modellvergleich.csv"))
    args = parser.parse_args(argv)

    truth = _load_truth(args.truth)
    content = load_content(read_table(args.content.read_bytes(), args.content.name))
    rows = []
    for config in args.configs:
        spec = spec_for(config)
        api_key = os.environ.get("OPENAI_API_KEY") if spec.provider == "openai" else None
        embedder = make_embedder(spec, api_key=api_key, cache_dir=args.cache_dir)
        # gleiche Chunks für alle Konfigurationen, damit nur das Modell den Unterschied macht
        metrics = evaluate(truth, content.urls, content.contents, embedder, args.chunk_size, args.overlap, args.basis)
        rows.append({"Konfiguration": config, "Modell": spec.model_id, **metrics})
        print(f"{config}: Rang 1 {metrics['hit1']:.0%}, Top 3 {metrics['hit3']:.0%}, MRR {metrics['mrr']:.3f}")
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Gespeichert: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
