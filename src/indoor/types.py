"""typed output for indoor methods."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from indoor.config import IndoorProblemConfig


class IndoorField(BaseModel):
    """temperature + velocity field for one indoor method run."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    T: NDArray[np.floating[Any]]
    x_grid: NDArray[np.floating[Any]]
    z_grid: NDArray[np.floating[Any]]
    method_name: str
    runtime_seconds: float
    config: IndoorProblemConfig
    u: NDArray[np.floating[Any]] | None = None
    w: NDArray[np.floating[Any]] | None = None

    def model_post_init(self, __context: object) -> None:
        object.__setattr__(self, "T", np.asarray(self.T, dtype=np.float64))
        object.__setattr__(self, "x_grid", np.asarray(self.x_grid, dtype=np.float64))
        object.__setattr__(self, "z_grid", np.asarray(self.z_grid, dtype=np.float64))
        if self.u is not None:
            object.__setattr__(self, "u", np.asarray(self.u, dtype=np.float64))
        if self.w is not None:
            object.__setattr__(self, "w", np.asarray(self.w, dtype=np.float64))
