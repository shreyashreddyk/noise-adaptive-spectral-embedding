from __future__ import annotations

from pathlib import Path

import numpy as np

from nase.plots.ablations import plot_case_metric_comparison
from nase.plots.stability_heatmap import plot_stability_heatmap


def test_stability_heatmap_plot_written(tmp_path: Path) -> None:
    matrix = np.array([[1.0, 0.8], [0.8, 1.0]])
    out = tmp_path / "heatmap.png"
    plot_stability_heatmap(matrix=matrix, epsilon_grid=[0.5, 1.0], out_path=out, dpi=120)
    assert out.exists()
    assert out.stat().st_size > 0


def test_case_metric_comparison_plot_written(tmp_path: Path) -> None:
    out = tmp_path / "comparison.png"
    plot_case_metric_comparison(
        labels=["a", "b"], values=[1.0, 2.0], metric_name="Mean selected k*", out_path=out, dpi=120
    )
    assert out.exists()
    assert out.stat().st_size > 0
