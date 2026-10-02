# TF-IDF Content Recommender

Recommend movies from a small included catalog by comparing plot descriptions with TF-IDF vectors and cosine similarity. The ranking is transparent and runs entirely on the local machine.

## Quick start

```bash
python -m venv .venv
python -m pip install -e .
movie-recs "Spirited Away" --limit 5
```

Use `--catalog path/to/movies.csv` to bring your own CSV with `title`, `genres`, and `overview` columns. The included catalog is original demo content, not a scraped commercial dataset.

## Learning notes

TF-IDF gives more weight to informative terms and less to common terms. Cosine similarity compares the resulting vectors by direction, so longer descriptions do not automatically dominate. This is a content-based baseline, not collaborative filtering.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
```

## License

MIT. See [LICENSE](LICENSE).
