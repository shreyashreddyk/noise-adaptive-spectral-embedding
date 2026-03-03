from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class DataConfig:
    manifold: str = "swiss_roll"
    n_samples: int = 500
    intrinsic_dim: int = 2
    ambient_dim: int = 3
    noise_std: float = 0.05
    seed: int = 42


@dataclass(slots=True)
class GraphConfig:
    epsilon: float = 1.0
    epsilon_grid: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    use_knn: bool = False
    knn_k: int = 20
    sparse_threshold_n: int = 2000
    enable_dss: bool = False
    dss_max_iter: int = 500
    dss_tol: float = 1e-6
    dss_min_value: float = 1e-12


@dataclass(slots=True)
class SpectralConfig:
    n_eigs: int = 25
    diffusion_time: int = 1
    alpha: float = 0.5


@dataclass(slots=True)
class CutoffConfig:
    method: str = "bandwidth_stability"
    eigengap_min_k: int = 1
    eigengap_max_k: int = 20
    stability_min_k: int = 1
    stability_max_k: int = 20
    stability_threshold: float = 0.9
    r_constant: float = 1.0
    r_estimation_k: int = 2
    use_estimated_r: bool = False


@dataclass(slots=True)
class PlotConfig:
    dpi: int = 150
    formats: list[str] = field(default_factory=lambda: ["png", "svg"])


@dataclass(slots=True)
class EstimatorConfig:
    enable_intrinsic_dim: bool = False
    intrinsic_dim_k: int = 10
    intrinsic_dim_estimate_clean: bool = True


@dataclass(slots=True)
class ExperimentConfig:
    name: str = "nase_experiment"
    output_root: Path = Path("results")
    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    cutoff: CutoffConfig = field(default_factory=CutoffConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    estimators: EstimatorConfig = field(default_factory=EstimatorConfig)
