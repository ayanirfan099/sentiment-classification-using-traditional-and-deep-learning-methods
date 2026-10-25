"""Convert already acquired Stanford IMDb tree; no unsafe tar extraction."""

import argparse
from pathlib import Path

import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("root")
p.add_argument("--output", default="data/imdb.csv")
a = p.parse_args()
rows = []
for split in ["train", "test"]:
    for sentiment, label in [("neg", 0), ("pos", 1)]:
        paths = sorted((Path(a.root) / split / sentiment).glob("*.txt"))
        if not paths:
            raise ValueError(f"No reviews at {split}/{sentiment}")
        for path in paths:
            rows.append({"text": path.read_text(encoding="utf-8"), "label": label, "split": split})
out = Path(a.output)
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print(len(rows), "rows; inspect exact duplicate policy before full benchmark")
