import ast
import json
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager

root = Path(r"C:/Users/19670/Desktop/ml4sci_ass1")
scratch = root / "tmp/part2_eda"
original_text = (scratch / "before.ipynb").read_text(encoding="utf-8")
original = json.loads(original_text)
code = (scratch / "plot_eda.py").read_text(encoding="utf-8")
heading = (scratch / "heading.md").read_text(encoding="utf-8")
ast.parse(code)
run_dir = root / "assignments/part2_continued/e85b5013cdce"
ground_truth = json.loads((run_dir / "gaussian_evaluation.json").read_text(encoding="utf-8"))["histograms"]["ground_truth"]
other_ground_truth = json.loads((run_dir / "informed_evaluation.json").read_text(encoding="utf-8"))["histograms"]["ground_truth"]
assert ground_truth == other_ground_truth
assert set(ground_truth) == {"speed", "acceleration", "tortuosity", "nearest_neighbor"}

os.environ["IPYTHONDIR"] = str(scratch / "ipython")
kernel_dir = scratch / "kernels" / "part2-eda"
kernel_dir.mkdir(parents=True, exist_ok=True)
(kernel_dir / "kernel.json").write_text(json.dumps({
    "argv": [r"D:/anaconda/envs/MLE/python.exe", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "Part 2 EDA plot", "language": "python"
}), encoding="utf-8")
manager = AsyncKernelManager(
    kernel_name="part2-eda",
    kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(scratch / "kernels")]),
)
execution = nbformat.v4.new_notebook(cells=[
    nbformat.v4.new_code_cell(code, id="part2-ground-truth-eda")
])
client = NotebookClient(execution, km=manager, timeout=120,
                        resources={"metadata": {"path": str(root / "assignments")}})
client.owns_km = True
execution = client.execute()
plot_cell = dict(execution.cells[0])
assert not any(output.output_type == "error" for output in execution.cells[0].outputs)
assert sum("image/png" in output.get("data", {}) for output in execution.cells[0].outputs) == 1
image_path = run_dir / "ground_truth_eda.png"
assert image_path.is_file() and image_path.stat().st_size > 1000
plot_cell["source"] = code.splitlines(keepends=True)
md_cell = dict(nbformat.v4.new_markdown_cell(heading, id="part2-ground-truth-eda-heading"))
md_cell["source"] = heading.splitlines(keepends=True)
updated = json.loads(json.dumps(original))
insertion = next(i for i, cell in enumerate(updated["cells"])
                 if cell["cell_type"] == "markdown"
                 and "".join(cell["source"]).startswith("## Model architecture"))
assert not any(cell.get("id") == "part2-ground-truth-eda" for cell in original["cells"])
updated["cells"][insertion:insertion] = [md_cell, plot_cell]
remaining = [cell for cell in updated["cells"] if cell.get("id") not in
             ("part2-ground-truth-eda", "part2-ground-truth-eda-heading")]
assert remaining == original["cells"]
assert {k: v for k, v in updated.items() if k != "cells"} == {
    k: v for k, v in original.items() if k != "cells"}
nbformat.validate(nbformat.reads(json.dumps(updated), as_version=4))
path = root / "assignments/Assignment_2_part2.ipynb"
assert path.read_text(encoding="utf-8") == original_text, "Notebook changed during plotting"
path.write_text(json.dumps(updated, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
summary = dict(
    eda_code_cells_added=1, heading_cells_added=1,
    insertion_before_model_architecture=True,
    original_cells_and_outputs_preserved=True,
    plotted_sources=["ground_truth"], histogram_panels=4,
    cached_scenes=2000, frame_range=[10,29],
    embedded_plot_outputs=1, image_path=str(image_path),
    model_predictions_recomputed=False,
)
(scratch / "verification_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
