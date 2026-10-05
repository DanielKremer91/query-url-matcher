import pandas as pd

from . import labels as L
from .ingest import KEYWORD_ALIASES, URL_ALIASES, _require
from .match import ranks, run_matching
from .normalize import normalize_url

OUTPUT_COLUMNS = [L.C_S_CHUNK, L.C_S_FULL, L.C_S_COMBI, L.C_PAIR_RANK, L.C_BEST_URL, L.C_NOTE]


def stale_columns(df, keep=()) -> list:
    """Ergebnisspalten, die schon in der Eingabe stehen (etwa aus einem früheren Paare-Export)."""
    return [column for column in OUTPUT_COLUMNS if column in df.columns and column not in keep]


def score_pairs(
    df, urls, contents, embedder, chunk_size, chunk_overlap, basis="chunk", weight=0.7, keyword_col=None, url_col=None
) -> pd.DataFrame:
    kcol = _require(df, keyword_col, KEYWORD_ALIASES, "Keyword")
    ucol = _require(df, url_col, URL_ALIASES, "URL")

    work = df.drop(columns=stale_columns(df, keep=(kcol, ucol)))
    work = work.dropna(subset=[kcol, ucol]).reset_index(drop=True)
    if work.empty:
        return work.assign(**{column: [] for column in OUTPUT_COLUMNS})

    keywords = list(dict.fromkeys(str(k).strip() for k in work[kcol]))
    result = run_matching(keywords, urls, contents, embedder, chunk_size, chunk_overlap)
    lead = result.lead(basis, weight)
    combined = result.combined(weight)
    rank = ranks(lead)
    k_index = {k: i for i, k in enumerate(keywords)}
    u_index = {normalize_url(u): j for j, u in enumerate(urls)}

    rows = []
    for keyword, url in zip(work[kcol], work[ucol]):
        i = k_index[str(keyword).strip()]
        j = u_index.get(normalize_url(url))
        best = result.urls[int(lead[i].argmax())]
        if j is None:
            rows.append((None, None, None, None, best, L.NOTE_URL_MISSING))
        else:
            rows.append(
                (
                    round(float(result.chunk_scores[i, j]), 4),
                    round(float(result.full_scores[i, j]), 4),
                    round(float(combined[i, j]), 4),
                    int(rank[i, j]),
                    best,
                    "",
                )
            )
    scored = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    scored[L.C_PAIR_RANK] = scored[L.C_PAIR_RANK].astype("Int64")
    return pd.concat([work, scored], axis=1)
