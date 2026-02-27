from __future__ import annotations

from pathlib import Path

import numpy as np

from nase.plots.ablations import plot_case_metric_comparison
from nase.plots.embeddings import plot_embedding_2d, plot_embedding_3d
from nase.plots.spectrum import plot_eigengap, plot_spectrum
from nase.plots.stability import plot_stability_scores
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


def test_spectrum_and_eigengap_plots_written(tmp_path: Path) -> None:
    eigenvalues = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    spectrum_out = tmp_path / "spectrum.png"
    eigengap_out = tmp_path / "eigengap.png"
    plot_spectrum(eigenvalues=eigenvalues, out_path=spectrum_out, dpi=120, selected_k=2)
    plot_eigengap(eigenvalues=eigenvalues, out_path=eigengap_out, dpi=120, selected_k=2)
    assert spectrum_out.exists()
    assert spectrum_out.stat().st_size > 0
    assert eigengap_out.exists()
    assert eigengap_out.stat().st_size > 0


def test_stability_plot_with_selected_k_written(tmp_path: Path) -> None:
    out = tmp_path / "stability.png"
    plot_stability_scores(
        scores={1: 0.98, 2: 0.95, 3: 0.75},
        out_path=out,
        dpi=120,
        selected_k=2,
        threshold=0.9,
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_embedding_2d_and_3d_plots_written(tmp_path: Path) -> None:
    embedding = np.array(
        [
            [0.0, 0.1, 0.2],
            [0.2, 0.0, 0.1],
            [0.4, 0.3, 0.0],
            [0.5, 0.5, 0.2],
        ]
    )
    continuous = np.array([0.0, 1.0, 2.0, 3.0])
    labels = np.array(["a", "a", "b", "b"])

    out_2d = tmp_path / "embedding_2d.png"
    out_3d = tmp_path / "embedding_3d.png"
    plot_embedding_2d(embedding=embedding[:, :2], colour=continuous, out_path=out_2d, dpi=120)
    plot_embedding_3d(embedding=embedding, colour=labels, out_path=out_3d, dpi=120)
    assert out_2d.exists()
    assert out_2d.stat().st_size > 0
    assert out_3d.exists()
    assert out_3d.stat().st_size > 0
