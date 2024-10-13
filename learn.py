from typing import Generic

from config import ConfigDict

from alphazero.net import AlphaZeroNet
from alphazero.coach import AlphaZeroCoach

from muzero.net import MuZeroNet
from muzero.coach import MuZeroCoach


def learn_alphazero(game: Generic, config: ConfigDict) -> None:
    net = AlphaZeroNet(game, config)
    if config.load_model:
        net.load_checkpoint()

    coach = AlphaZeroCoach(game, net, config)
    if config.load_model:
        coach.load_replay_buffer()

    coach.learn()


def learn_muzero(game: Generic, config: ConfigDict) -> None:
    net = MuZeroNet(game, config)
    if config.load_model:
        net.load_checkpoint()

    coach = MuZeroCoach(game, net, config)
    if config.load_model:
        coach.load_replay_buffer()

    coach.learn()