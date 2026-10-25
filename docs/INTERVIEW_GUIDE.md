# Study guide

These notes explain implemented methods; do not claim independent authorship or experiments beyond the recorded results.

## Where can TF-IDF leakage occur?

Fitting vocabulary or IDF on validation/test text changes the feature representation using evaluation data. Fit it inside the training pipeline.

## Was DistilBERT trained here?

No. The default inference checkpoint was fine-tuned upstream on SST-2; local tests use tiny random weights solely to test the execution path.

## How does truncation affect long reviews?

Only the first 512 tokens contribute in this inference path. Length-sliced evaluation and an explicit limitation are necessary; no full-document claim is made.
