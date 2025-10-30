class BaseMetric:
    """
    Base class for all metrics.

    Parameters
    ----------
    name : str or None, optional
        Name of the metric. If None, uses class name.
    *args : tuple
        Additional positional arguments.
    **kwargs : dict
        Additional keyword arguments.
    """

    def __init__(self, name: str | None = None, *args, **kwargs):
        self.name = name if name is not None else type(self).__name__

    def update(self, **batch):
        """
        Update metric state with batch data.

        Parameters
        ----------
        **batch : dict
            Batch data for metric computation.

        Raises
        ------
        NotImplementedError
            Must be implemented in subclass.
        """
        raise NotImplementedError('This method must be implemented in the nested class.')

    def __call__(self, **batch) -> float:
        """
        Compute and return metric value.

        Parameters
        ----------
        **batch : dict
            Batch data for metric computation.

        Returns
        -------
        float
            Computed metric value.

        Raises
        ------
        NotImplementedError
            Must be implemented in subclass.
        """
        raise NotImplementedError('This method must be implemented in the nested class.')

    def __repr__(self) -> str:
        """
        Return string representation of the metric.

        Returns
        -------
        str
            String representation.
        """
        return f'{type(self).__name__}()'
