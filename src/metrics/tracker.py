import pandas as pd


class MetricTracker:
    """
    Tracks and computes running averages of multiple metrics.

    Parameters
    ----------
    *names : str
        Names of metrics to track.
    """

    def __init__(self, *names: str):
        self._data = pd.DataFrame(index=names, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        """Reset all metric values to zero."""
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, name: str, value: float):
        """
        Update a metric with a new value.

        Parameters
        ----------
        name : str
            Name of the metric to update.
        value : float
            New value to add to the running average.
        """
        self._data.loc[name, 'total'] += value
        self._data.loc[name, 'counts'] += 1
        self._data.loc[name, 'average'] = self._data.total[name] / self._data.counts[name]

    def __getitem__(self, name: str) -> float:
        """
        Get the average value of a metric.

        Parameters
        ----------
        name : str
            Name of the metric.

        Returns
        -------
        float
            Average value of the metric.
        """
        return self._data.average[name]

    def result(self) -> dict[str, float]:
        """
        Get all metric averages as a dictionary.

        Returns
        -------
        dict[str, float]
            Dictionary mapping metric names to their averages.
        """
        return dict(self._data.average)

    def names(self) -> list[str]:
        """
        Get list of all tracked metric names.

        Returns
        -------
        list[str]
            List of metric names.
        """
        return list(self._data.total.keys())
