# Implementation status

Implemented but not fully benchmarked

A tested comparison pipeline for majority baseline, TF-IDF logistic regression, TF-IDF linear SVM and explicitly pretrained DistilBERT inference. Models are shown in the demo only when their artifacts or inference path are available.

Four tests passed: invalid schema/labels/duplicates, seeded classical reproduction, no unseen-token vocabulary leakage, full fixture training/reload, CLI help, and offline tokenizer/checkpoint DistilBERT inference with repeated probability equality. Streamlit initial page rendering checked separately.

IMDb and Hugging Face access returned proxy tunnel 403, so real IMDb and pretrained inference metrics are pending. Four transformer-path assertions use random tiny local DistilBERT weights, not a trained sentiment model. Optional fine-tuning is not implemented. Handwritten fixture has 24 train and 8 test reviews with deliberately simple vocabulary; its metrics cannot establish sentiment generalization. Short/long bins are descriptive, not tuned on test. Duplicate input texts are rejected conservatively; investigate provenance and deduplicate official training only before full IMDb execution without altering official test boundaries. No GPU or broad domain benchmark. User-upload browser interaction is untested.

Exact commands: [reproducibility](reproducibility.md). Push status and commit identity are recorded in the portfolio summary; remote CI success is not implied.
