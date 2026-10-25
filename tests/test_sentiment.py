import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentiment_study.core import TransformerSentiment, train, validate

FIXTURE = Path(__file__).parent / "fixtures/reviews.csv"


def test_validation():
    f = pd.read_csv(FIXTURE)
    assert len(validate(f)) == 32
    with pytest.raises(ValueError):
        validate(pd.concat([f, f.iloc[:1]]))
    f.loc[0, "label"] = 3
    with pytest.raises(ValueError):
        validate(f)


def test_classical(tmp_path):
    a = train(FIXTURE, tmp_path / "a", fixture=True)
    b = train(FIXTURE, tmp_path / "b", fixture=True)
    for name in a["models"]:
        assert a["models"][name]["accuracy"] == b["models"][name]["accuracy"]
    import joblib

    m = joblib.load(tmp_path / "a/logistic.joblib")
    assert m.predict(["unseen ambiguous wording"]).shape == (1,)
    assert "unseen" not in m.named_steps["tfidf"].vocabulary_


def test_transformer_offline(tmp_path):
    # Random tiny weights test serialization/tokenization, not sentiment quality.
    import torch
    from transformers import BertTokenizer, DistilBertConfig, DistilBertForSequenceClassification

    torch.manual_seed(42)
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ngood\nbad\nfilm\n")
    tokenizer = BertTokenizer(vocab_file=str(vocab))
    tokenizer.save_pretrained(tmp_path)
    model = DistilBertForSequenceClassification(
        DistilBertConfig(vocab_size=8, dim=16, hidden_dim=32, n_layers=1, n_heads=2, num_labels=2)
    )
    model.save_pretrained(tmp_path)
    wrapper = TransformerSentiment(tmp_path, local_only=True)
    prob = wrapper.predict(["good film", "bad film"])
    assert prob.shape == (2,) and np.isfinite(prob).all()
    np.testing.assert_array_equal(prob, wrapper.predict(["good film", "bad film"]))
    with pytest.raises(ValueError):
        wrapper.predict([""])


def test_cli():
    assert (
        subprocess.run(
            [sys.executable, "-m", "sentiment_study.cli", "--help"],
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )
