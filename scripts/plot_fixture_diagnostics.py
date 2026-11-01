"""Visualize the committed fixture measurements, without inventing IMDb results."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
report = json.loads((ROOT / "results/fixture/metrics.json").read_text())
models = report["models"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
names = [name for name, value in models.items() if "inference_seconds" in value]
for name in names:
    value = models[name]
    axes[0].scatter(value["inference_seconds"] * 1000, value["macro_f1"], label=name)
axes[0].set(
    xlabel="Recorded inference milliseconds (8-review batch)",
    ylabel="Macro F1",
    ylim=(0, 1),
    title="Fixture quality versus recorded latency",
)
axes[0].legend()
matrix = models["svm"]["confusion_matrix"]
axes[1].imshow(matrix, cmap="Blues", vmin=0)
for i, row in enumerate(matrix):
    for j, count in enumerate(row):
        axes[1].text(j, i, str(count), ha="center", va="center")
axes[1].set(
    xticks=[0, 1],
    yticks=[0, 1],
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"],
    xlabel="Predicted",
    ylabel="Actual",
    title="Selected SVM: 8 fixture reviews",
)
fig.suptitle("Handwritten software fixture — not IMDb or a pretrained-model benchmark")
fig.savefig(ROOT / "results/fixture/diagnostics.png", dpi=150)
plt.close(fig)
