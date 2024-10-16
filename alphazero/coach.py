from typing import Generic, Optional, List

from coach import Coach
from config import ConfigDict

from alphazero.mcts import AlphaZeroMCTS
from alphazero.player import DefaultAlphaZeroPlayer

from games.history import GameHistory

from utils.sample_batch import _sample_batch


class AlphaZeroCoach(Coach):

    def __init__(self, game: Generic, net: Generic, config: ConfigDict, run_name: Optional[str] = None) -> None:
        super().__init__(game, net, config, AlphaZeroMCTS, DefaultAlphaZeroPlayer)

        # Initialize tensorboard logging.
        # if run_name is None:
        #     run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

        # self.log_dir = f"out/logs/AlphaZero/{self.neural_net.architecture}/" + run_name
        # self.file_writer = tf.summary.create_file_writer(self.log_dir + "/metrics")
        # self.file_writer.set_as_default()

    def sample_batch(self, histories: List[GameHistory]) -> List:
        sample_coordinates, sample_weight = _sample_batch(
            list_of_histories=histories,
            n=self.config.batch_size,
            prioritize=self.config.prioritize,
            alpha=self.config.prioritize_alpha,
            beta=self.config.prioritize_beta
        )

        examples = [(
            histories[h_i].stack_observations(self.config.observation_length, t=i),
            (histories[h_i].probabilities[i], histories[h_i].observed_returns[i]),
            loss_scale
        )
            for (h_i, i), loss_scale in zip(sample_coordinates, sample_weight)
        ]

        return examples