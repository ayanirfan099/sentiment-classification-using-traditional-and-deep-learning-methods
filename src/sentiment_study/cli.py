import argparse
import json

from .core import CHECKPOINT, train


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--output", default="results/imdb")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--transformer", action="store_true")
    p.add_argument("--fixture", action="store_true")
    p.add_argument("--checkpoint", default=CHECKPOINT)
    print(json.dumps(train(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
