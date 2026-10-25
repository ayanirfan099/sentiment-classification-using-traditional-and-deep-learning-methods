# Reproduce

Python >=3.11; executed in Python 3.12 on CPU. From repository root:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
# Acquire Stanford IMDb from https://ai.stanford.edu/~amaas/data/sentiment/
python scripts/prepare_imdb.py data/aclImdb --output data/imdb.csv
python -m sentiment_study.cli --csv data/imdb.csv --transformer --output results/imdb
# Fully offline fixture run:
python -m sentiment_study.cli --csv tests/fixtures/reviews.csv --fixture --output results/fixture
streamlit run src/sentiment_study/app.py
pytest -q
ruff check .
ruff format --check .
```

Use CPU PyTorch wheels from https://download.pytorch.org/whl/cpu if installing the deep learning dependencies on a CPU machine. Remote datasets and weights require separate access. Do not load untrusted pickle/joblib/checkpoint artifacts.
