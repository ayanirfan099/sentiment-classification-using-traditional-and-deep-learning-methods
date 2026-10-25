from pathlib import Path

import joblib
import streamlit as st

from sentiment_study.core import TransformerSentiment

st.title("Sentiment model comparison")
st.caption(
    "Research demo. Fixture-trained models are not IMDb benchmarks; pretrained SST-2 models were not trained here."
)
folder = Path(st.text_input("Artifact directory", "results/fixture"))
text = st.text_area("Review")
if st.button("Compare") and text.strip():
    available = 0
    for name in ["majority", "logistic", "svm"]:
        p = folder / f"{name}.joblib"
        if p.exists():
            model = joblib.load(p)
            st.write(name, "positive" if model.predict([text])[0] else "negative")
            available += 1
    if not available:
        st.info("No trained classical pipelines found. Run training first.")
if st.checkbox("Use accessible pretrained DistilBERT (download may be required)") and text.strip():
    try:
        st.write(
            "Pretrained positive probability", float(TransformerSentiment().predict([text])[0])
        )
    except (OSError, ValueError) as exc:
        st.error(f"Pretrained model unavailable: {exc}")
