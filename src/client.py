from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

import requests

DEFAULT_ENDPOINT = os.getenv("RECOMMENDER_URL", "http://localhost:50023/api/recommend")

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "songs",
        nargs="+",
        help="Seed songs used to generate the playlist recommendations.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_ENDPOINT,
        help=f"Recommendation endpoint (default: {DEFAULT_ENDPOINT}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of songs returned by the API.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print the raw JSON response including evidence details.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    payload = {"songs": args.songs}
    if args.limit is not None:
        payload["limit"] = args.limit

    try:
        response = requests.post(args.url, json=payload, timeout=15)
    except requests.RequestException as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1

    if response.status_code != 200:
        print(
            f"Request failed with status {response.status_code}: {response.text}",
            file=sys.stderr,
        )
        return 1

    data = response.json()
    songs = data.get("songs", [])

    if args.details:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print("Recommendations:")
        if not songs:
            print("  (no recommendations available)")
        for index, song in enumerate(songs, start=1):
            print(f"  {index:2d}. {song}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())