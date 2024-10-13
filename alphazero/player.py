from typing import Optional

from player import Player
from games.state import GameState

import numpy as np


class DefaultAlphaZeroPlayer(Player):

    def __init__(self, game, name: str = '') -> None:
        super().__init__(game, name, parametric=True)

    def set_variables(self, model, mcts, name: str) -> None:
        self.model = model
        self.mcts = mcts
        self.name = name

    def refresh(self, hard_reset: bool = False):
        super().refresh(hard_reset)
        self.mcts.clear_tree()

    def act(self, state: GameState) -> int:
        pi, _ = self.mcts.run_mcts(state, self.history, temp=0)
        return np.argmax(pi).item()
