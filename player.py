from abc import ABC, abstractmethod
from typing import Generic, Optional

from games.state import GameState
from games.history import GameHistory

import numpy as np


class Player(ABC):

    def __init__(self, game: Generic, name: str = '', parametric: bool = False) -> None:
        self.game = game
        self.name = name
        self.parametric = parametric
        self.histories = list()
        self.history = GameHistory()

    def bind_history(self, history: GameHistory) -> None:
        self.history = history

    def refresh(self, hard_reset: bool = False) -> None:
        if hard_reset:
            self.histories = list()
            self.history.refresh()
        else:
            self.histories.append(self.history)
            self.history = GameHistory()

    def observe(self, state: GameState) -> None:
        self.history.capture(state, np.array([]), 0, 0)

    def clone(self):
        return self.__class__(self.game)

    @abstractmethod
    def act(self, state: GameState) -> int:
        pass
