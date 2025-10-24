import pandas as pd


class MetricTracker:
    def __init__(self, *names: str):
        self._data = pd.DataFrame(index=names, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, name: str, value: float):
        self._data.loc[name, 'total'] += value
        self._data.loc[name, 'counts'] += 1
        self._data.loc[name, 'average'] = self._data.total[name] / self._data.counts[name]

    def __getitem__(self, name: str) -> float:
        return self._data.average[name]

    def result(self) -> dict[str, float]:
        return dict(self._data.average)

    def names(self) -> list[str]:
        return list(self._data.total.keys())
