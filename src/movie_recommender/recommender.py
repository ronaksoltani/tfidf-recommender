from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REQUIRED_COLUMNS = {"title", "genres", "overview"}


def load_catalog(path: Path | None = None) -> pd.DataFrame:
    source = path if path else files("movie_recommender").joinpath("catalog.csv")
    frame = pd.read_csv(source).fillna("")
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"catalog is missing required columns: {', '.join(sorted(missing))}")
    frame["features"] = (frame["genres"].astype(str) + " " + frame["overview"].astype(str)).str.strip()
    return frame


def recommend(title: str, catalog: pd.DataFrame, limit: int = 5) -> list[dict[str, object]]:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    matches = catalog.index[catalog["title"].str.casefold() == title.strip().casefold()].tolist()
    if not matches:
        choices = ", ".join(catalog["title"].astype(str).head(5))
        raise ValueError(f"unknown title {title!r}; examples: {choices}")
    matrix = TfidfVectorizer(stop_words="english").fit_transform(catalog["features"])
    scores = cosine_similarity(matrix[matches[0]], matrix).ravel()
    ranked = sorted(((index, score) for index, score in enumerate(scores) if index != matches[0]),
                    key=lambda item: (-item[1], str(catalog.iloc[item[0]]["title"]).casefold()))
    return [{"title": str(catalog.iloc[index]["title"]), "genres": str(catalog.iloc[index]["genres"]),
             "similarity": round(float(score), 4)} for index, score in ranked[:limit]]
