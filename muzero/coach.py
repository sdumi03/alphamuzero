from typing import Generic, Optional, List
from datetime import datetime

from games.history import GameHistory
from coach import Coach

from config import ConfigDict

from utils.sample_batch import _sample_batch


class MuZeroCoach(Coach):

    def __init__(self, game: Generic, net: Generic, config: ConfigDict, run_name: Optional[str] = None) -> None:
        super().__init__(game, net, config, MuZeroMCTS, DefaultMuZeroPlayer)

        # Initialize tensorboard logging
        # if run_name is None:
        #     run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

        # self.log_dir = f"out/logs/MuZero/{self.net.architecture}/" + run_name
        # self.file_writer = tf.summary.create_file_writer(self.log_dir + "/metrics")
        # self.file_writer.set_as_default()

        # Define helper variables
        # self.return_forward_observations = (net.config.dynamics_penalty > 0 or net.config.latent_decoder)
        self.observation_stack_length = net.config.observation_length

    def sample_batch(self, histories: List[GameHistory]) -> List:

        # Generate coordinates within the replay buffer to sample from. Also generate the loss scale of said samples
        sample_coordinates, sample_weight = _sample_batch(
            list_of_histories=histories,
            n=self.neural_net.net_args.batch_size,
            prioritize=self.args.prioritize,
            alpha=self.args.prioritize_alpha,
            beta=self.args.prioritize_beta
        )

        # Collect training examples for MuZero: (input, action, (targets), forward_observations, loss_scale)
        examples = [(
            histories[h_i].stack_observations(self.observation_stack_length, t=i),
            *self.build_hypothetical_steps(histories[h_i], t=i, k=self.args.K),
            loss_scale
        )
            for (h_i, i), loss_scale in zip(sample_coordinates, sample_weight)
        ]

        return examples
