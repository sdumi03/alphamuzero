from typing import Generic, List, Tuple

from config import ConfigDict

from alphazero.monitor import AlphaZeroMonitor
from architectures import AlphaZeroArch

from utils.loss import scalar_to_support, support_to_scalar

import os
import numpy as np
import tensorflow as tf


class AlphaZeroNet:

    def __init__(self, game: Generic, config: ConfigDict) -> None:
        self.single_player = (game.n_players == 1)
        self.action_size = game.get_action_size()
        self.net = AlphaZeroArch[config.architecture](game, config)
        self.steps = 0
        self.monitor = AlphaZeroMonitor(self, config)

    def train(self, examples: List) -> None:
        observations, targets, loss_scale = list(zip(*examples))
        target_pis, target_vs = list(map(np.asarray, zip(*targets)))

        observations = np.asarray(observations)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        priorities = np.asarray(loss_scale)

        target_vs = scalar_to_support(target_vs, self.net.config.support_size)

        # total_loss, pi_loss, v_loss
        total_loss = self.net.model.train_on_batch(
            x=observations,
            y=[target_pis, target_vs],
            sample_weight=[priorities, priorities]
        )

        l2_norm = tf.reduce_sum([tf.nn.l2_loss(x) for x in self.net.model.get_weights()])

        # self.monitor.log(pi_loss, "pi_loss")
        # self.monitor.log(v_loss, "v_loss")

        # self.monitor.log(total_loss, "total loss")
        # self.monitor.log(l2_norm, "l2 norm")

        self.steps += 1

    def predict(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        # print(observations.shape)
        observations = observations[np.newaxis, ...]
        # print(observations.shape)

        pi, v = self.net.model.predict(observations, verbose=0)

        v_real = support_to_scalar(v, self.net.config.support_size)

        return pi[0], np.ndarray.item(v_real)

    def save_checkpoint(self, folder: str = 'checkpoint', filename: str = 'checkpoint.pth.tar') -> None:
        filepath = os.path.join(folder, filename)

        if not os.path.exists(folder):
            os.mkdir(folder)

        self.net.model.save_weights(filepath)

    def load_checkpoint(self, folder: str = 'checkpoint', filename: str = 'checkpoint.pth.tar') -> None:
        filepath = os.path.join(folder, filename)

        try:
            self.net.model.load_weights(filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"No AlphaZero Model in path {filepath}")
