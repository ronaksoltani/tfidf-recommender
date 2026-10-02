import argparse
from pathlib import Path

from .recommender import load_catalog, recommend


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recommend movies from plot and genre similarity.")
    parser.add_argument("title")
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        results = recommend(args.title, load_catalog(args.catalog), args.limit)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Because you liked {args.title}:\n")
    for rank, item in enumerate(results, start=1):
        print(f"{rank}. {item['title']} — {item['genres']} (similarity {item['similarity']:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
