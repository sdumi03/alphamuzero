from typing import Optional, Callable

from config import ConfigDict

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


class ParameterScheduler:

    def __init__(self, config: ConfigDict):
        self.config = config

    def build(self):
        indices, values = list(zip(*self.config.temp_schedule_points))

        schedulers = {
            'linear': self.linear_schedule,
            'stepwise': self.step_wise_decrease
        }

        return schedulers[self.config.temp_schedule_method](np.array(indices), np.array(values))

    @staticmethod
    def linear_schedule(indices: np.ndarray, values: np.ndarray) -> Callable:
        def scheduler(training_steps):
            return np.interp(training_steps, indices, values)

        return scheduler

    @staticmethod
    def step_wise_decrease(indices: np.ndarray, values: np.ndarray) -> Callable:
        def scheduler(training_steps):
            current_pos = np.sum(np.cumsum(training_steps > indices))
            return values[current_pos] if current_pos < len(values) else values[-1]

        return scheduler

