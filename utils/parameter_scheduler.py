from typing import Callable
from config import ConfigDict
import numpy as np


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

