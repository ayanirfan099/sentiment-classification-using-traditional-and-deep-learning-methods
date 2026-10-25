import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

CHECKPOINT = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    if not {"text", "label", "split"} <= set(frame):
        raise ValueError("Required columns: text,label,split")
    if frame.empty or frame.text.isna().any() or frame.text.astype(str).str.strip().eq("").any():
        raise ValueError("Empty text")
    if not frame.label.isin([0, 1]).all() or not frame.split.isin(["train", "test"]).all():
        raise ValueError("Invalid labels or splits")
    frame = frame.copy()
    frame["text"] = frame.text.astype(str)
    normalized = frame.text.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    if normalized.duplicated().any():
        raise ValueError("Duplicate texts can contaminate evaluation")
    if set(frame.split) != {"train", "test"}:
        raise ValueError("Both official train and test splits required")
    for split in ["train", "test"]:
        if frame.loc[frame.split == split, "label"].nunique() != 2:
            raise ValueError("Both classes required in each split")
    return frame


def measure(y, pred) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1,
        "macro_precision": precision,
        "macro_recall": recall,
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


class TransformerSentiment:
    """Pretrained inference only. No silent substitution or training claim."""

    def __init__(self, checkpoint=CHECKPOINT, local_only=False):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(2)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=local_only)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, local_files_only=local_only
        ).eval()
        if self.model.config.num_labels != 2:
            raise ValueError("Binary sentiment checkpoint required")
        # Explicit SST-2 convention; arbitrary checkpoints must document matching labels.
        self.checkpoint = str(checkpoint)

    def predict(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        import torch

        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Nonempty texts required")
        probabilities = []
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.inference_mode():
                probabilities.extend(self.model(**batch).logits.softmax(-1)[:, 1].tolist())
        return np.asarray(probabilities)


def train(csv, output, seed=42, transformer=False, checkpoint=CHECKPOINT, fixture=False):
    frame = validate(pd.read_csv(csv))
    train = frame[frame.split == "train"]
    test = frame[frame.split == "test"]
    a, b = train_test_split(
        np.arange(len(train)), test_size=0.25, stratify=train.label, random_state=seed
    )
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    candidates = {
        "majority": Pipeline(
            [("tfidf", TfidfVectorizer()), ("model", DummyClassifier(strategy="most_frequent"))]
        )
    }
    for c in [0.1, 1, 10]:
        for kind, model in [
            ("logistic", LogisticRegression(C=c, max_iter=2000, random_state=seed)),
            ("svm", LinearSVC(C=c, random_state=seed)),
        ]:
            candidates[f"{kind}_C{c}"] = Pipeline(
                [
                    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000)),
                    ("model", model),
                ]
            )
    validation = {}
    for name, model in candidates.items():
        model.fit(train.text.iloc[a], train.label.iloc[a])
        validation[name] = measure(train.label.iloc[b], model.predict(train.text.iloc[b]))
    # Select C within each model family on validation; fit on all official training data only.
    selected = {"majority": "majority"}
    for kind in ["logistic", "svm"]:
        selected[kind] = max(
            [n for n in candidates if n.startswith(kind)], key=lambda n: validation[n]["macro_f1"]
        )
    results = {}
    length = test.text.str.split().str.len()
    for family, name in selected.items():
        start = perf_counter()
        model = candidates[name].fit(train.text, train.label)
        fit_seconds = perf_counter() - start
        start = perf_counter()
        pred = model.predict(test.text)
        seconds = perf_counter() - start
        path = out / f"{family}.joblib"
        joblib.dump(model, path)
        result = measure(test.label, pred)
        result.update(
            {
                "selected": name,
                "fit_seconds": fit_seconds,
                "inference_seconds": seconds,
                "serialized_bytes": path.stat().st_size,
                "errors": [
                    {"test_position": i, "label": int(y), "prediction": int(p)}
                    for i, (y, p) in enumerate(zip(test.label, pred, strict=True))
                    if y != p
                ][:20],
            }
        )
        result["length_groups"] = {}
        for label, mask in [
            ("short_40_words_or_less", length <= 40),
            ("long_over_200_words", length > 200),
        ]:
            result["length_groups"][label] = {
                "count": int(mask.sum()),
                "metrics": measure(test.label[mask], pred[mask]) if mask.any() else None,
            }
        results[family] = result
    if transformer:
        start = perf_counter()
        deep = TransformerSentiment(checkpoint)
        load_seconds = perf_counter() - start
        start = perf_counter()
        prob = deep.predict(test.text.tolist())
        seconds = perf_counter() - start
        results["pretrained_distilbert"] = {
            **measure(test.label, prob >= 0.5),
            "checkpoint": checkpoint,
            "trained_here": False,
            "training_corpus": "SST-2 for default checkpoint, not IMDb",
            "load_seconds": load_seconds,
            "inference_seconds": seconds,
            "parameter_bytes": sum(p.numel() * p.element_size() for p in deep.model.parameters()),
        }
    result = {
        "dataset": "handwritten software fixture; not IMDb"
        if fixture
        else "IMDb official boundaries supplied by caller",
        "seed": seed,
        "train_count": len(train),
        "test_count": len(test),
        "validation": validation,
        "models": results,
        "document_lengths": {
            "train": train.text.str.split().str.len().describe().to_dict(),
            "test": length.describe().to_dict(),
        },
        "transformer_status": "executed"
        if transformer
        else "not executed; requires accessible checkpoint",
        "fine_tuning": "not implemented; optional future work",
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.bar(list(results), [v["macro_f1"] for v in results.values()])
    ax.set(ylabel="Macro F1", ylim=(0, 1), title=result["dataset"])
    fig.savefig(out / "comparison.png")
    plt.close(fig)
    return result
