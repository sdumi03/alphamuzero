from abc import ABC, abstractmethod
from typing import Union, List

import tensorflow as tf
import numpy as np


class Monitor(ABC):

    def __init__(self, net_instance, config):
        self.net_instance = net_instance
        self.config = config

    def log(self, tensor: Union[tf.Tensor, float], name: str) -> None:
        if self.net_instance.steps % self.net_instance.config.LOG_RATE == 0:
            tf.summary.scalar(name, data=tensor, step=self.net_instance.steps)

    def log_distribution(self, tensor: Union[tf.Tensor, np.ndarray], name: str) -> None:
        if self.net_instance.steps % self.net_instance.config.LOG_RATE == 0:
            tf.summary.histogram(name, data=tensor, step=self.net_instance.steps)

    @abstractmethod
    def log_batch(self, data_batch: List) -> None:
        pass
