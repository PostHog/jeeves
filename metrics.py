from __future__ import annotations

import torch

from distributed import all_reduce_


class Metrics:
    def __init__(self, names: tuple[str, ...], device):
        self.names = names
        self.values = torch.zeros(len(names), device=device)
        self.count = 0

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.values[self.names.index(name)]

    def add(self, **values) -> None:
        if values.keys() != set(self.names):
            raise ValueError(f"expected metrics {self.names}, got {tuple(values)}")
        with torch.no_grad():
            self.values.add_(torch.stack([torch.as_tensor(values[name], device=self.values.device).detach() for name in self.names]))
        self.count += 1

    def totals(self, world: int = 1) -> dict[str, float]:
        values = all_reduce_(self.values.clone(), world, average=True).tolist()
        return dict(zip(self.names, values))

    def means(self, world: int = 1) -> dict[str, float]:
        return {name: value / max(1, self.count) for name, value in self.totals(world).items()}

    def reset(self) -> None:
        self.values.zero_()
        self.count = 0
