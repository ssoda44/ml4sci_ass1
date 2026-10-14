import json
from pathlib import Path
import matplotlib.pyplot as plt

# Reuse only the ground-truth summaries; no prediction generation is needed.
eda_result_path = Path("part2_continued/e85b5013cdce/gaussian_evaluation.json")
if not eda_result_path.exists():
    eda_result_path = Path("assignments") / eda_result_path
eda_result = json.loads(eda_result_path.read_text(encoding="utf-8"))
eda_histograms = eda_result["histograms"]["ground_truth"]
eda_labels = {
    "speed": ("Speed", "Speed (m/s)"),
    "acceleration": ("Acceleration magnitude", "Acceleration magnitude (m/s²)"),
    "tortuosity": ("Tortuosity", "Path length / net displacement"),
    "nearest_neighbor": ("Closest-neighbour distance", "Distance (m)"),
}

eda_fig, eda_axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
for eda_ax, (eda_key, (eda_title, eda_xlabel)) in zip(eda_axes.flat, eda_labels.items()):
    eda_hist = eda_histograms[eda_key]
    eda_ax.stairs(eda_hist["density"], eda_hist["edges"],
                  label="Ground truth", color="black", linewidth=1.8)
    eda_ax.set(
        title=f"{eda_title}\n{100 * eda_hist['outside_fraction']:.2f}% outside shown range",
        xlabel=eda_xlabel, ylabel="Density",
        xlim=(eda_hist["edges"][0], eda_hist["edges"][-1]),
    )
    eda_ax.grid(alpha=0.2)
    eda_ax.legend()

eda_fig.suptitle(
    f"Ground-truth EDA: {eda_result['num_scenes']:,} held-out scenes, frames 10–29"
)
eda_fig.savefig(eda_result_path.parent / "ground_truth_eda.png", dpi=180)
plt.show()
