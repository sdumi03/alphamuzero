from typing import Optional
import numpy as np


class MinMaxStats(object):

    def __init__(self, minimum: Optional[float] = None, maximum: Optional[float] = None) -> None:
        self.default_max = self.maximum = maximum if maximum is not None else -np.inf
        self.default_min = self.minimum = minimum if minimum is not None else np.inf

    def refresh(self) -> None:
        self.maximum = self.default_max
        self.minimum = self.default_min

    def update(self, value: float) -> None:
        self.maximum = np.max([self.maximum, value])
        self.minimum = np.min([self.minimum, value])

    def normalize(self, value: float) -> float:
        if self.maximum >= self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum + 1e-8)
        return value
