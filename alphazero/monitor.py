from typing import List

from monitor import Monitor


class AlphaZeroMonitor(Monitor):

    def __init__(self, net_instance, config):
        super().__init__(net_instance, config)

    def log_batch(self, data_batch: List) -> None:
        """
        Log a large amount of neural network statistics based on the given batch.
        Functionality can be toggled on by specifying '--debug' as a console argument to Main.py.
        Note: toggling this functionality on will produce significantly larger tensorboard event files!

        Statistics include:
         - Priority sampling sample probabilities.
         - Values of each target/ prediction for the data batch.
         - Loss discrepancy between cross-entropy and MSE for the reward/ value predictions.
        """
        if self.net_instance.config.DEBUG_MODE and \
           self.net_instance.steps % self.net_instance.config.LOG_RATE == 0:

            observations, targets, sample_weight = list(zip(*data_batch))
            target_pis, target_vs = list(map(np.asarray, zip(*targets)))
            observations = np.asarray(observations)

            priority = sample_weight * len(data_batch)  # Undo 1/n scaling to get priority
            tf.summary.histogram(f"sample probability", data=priority, step=self.net_instance.steps)

            pis, vs = self.net_instance.net.model.predict_on_batch(observations)
            v_reals = support_to_scalar(vs, self.net_instance.config.support_size).ravel()  # as scalars

            tf.summary.histogram(f"v_targets", data=target_vs, step=self.net_instance.steps)
            tf.summary.histogram(f"v_predict", data=v_reals, step=self.net_instance.steps)

            mse = np.mean((v_reals - target_vs) ** 2)
            tf.summary.scalar("v_mse", data=mse, step=self.net_instance.steps)