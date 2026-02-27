from __future__ import annotations

import numpy as np

from nase.spectral.diffusion_maps import diffusion_map_embedding, diffusion_operator


def test_diffusion_map_embedding_shapes() -> None:
    affinity = np.array(
        [
            [0.0, 1.0, 0.2],
            [1.0, 0.0, 0.5],
            [0.2, 0.5, 0.0],
        ]
    )
    op = diffusion_operator(affinity, alpha=0.5)
    emb, evals, evecs = diffusion_map_embedding(op, n_eigs=3, time=1)
    assert evals.shape == (3,)
    assert evecs.shape == (3, 3)
    assert emb.shape[0] == 3
