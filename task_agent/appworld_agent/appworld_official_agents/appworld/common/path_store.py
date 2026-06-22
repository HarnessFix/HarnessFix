from __future__ import annotations

from pathlib import Path


class _PathStore:
    def __init__(self) -> None:
        self.cache = str(Path(".appworld_cache").resolve())
        self.experiment_outputs = str(Path(".appworld_experiment_outputs").resolve())

    def update_root(self, root: str) -> None:
        root_path = Path(root)
        self.cache = str(root_path / "cache")
        self.experiment_outputs = str(root_path / "experiments")


path_store = _PathStore()
