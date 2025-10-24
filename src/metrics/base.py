class BaseMetric:
    def __init__(self, name: str | None = None, *args, **kwargs):
        self.name = name if name is not None else type(self).__name__

    def update(self, **batch):
        raise NotImplementedError('This method must be implemented in the nested class.')

    def __call__(self, **batch) -> float:
        raise NotImplementedError('This method must be implemented in the nested class.')

    def __repr__(self) -> str:
        return f'{type(self).__name__}()'
