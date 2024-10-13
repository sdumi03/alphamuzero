from abc import ABC, abstractmethod
from typing import Tuple, Union, List

from games.state import GameState

import numpy as np


class Game(ABC):

    def __init__(self, n_players: int = 1) -> None:
        self.n_players = n_players
        self.n_symmetries = 1
        if self.n_players > 2:
            raise NotImplementedError(f"Environments for more than 2 agents are not yet supported, {n_players} > 2")

    @abstractmethod
    def get_initial_state(self) -> GameState:
        pass

    @abstractmethod
    def get_dimensions(self) -> Tuple[int, ...]:
        pass

    @abstractmethod
    def get_action_size(self) -> int:
        pass

    @abstractmethod
    def get_next_state(self, state: GameState, action: int, **kwargs) -> Tuple[GameState, float]:
        pass

    @abstractmethod
    def get_legal_moves(self, state: GameState) -> np.ndarray:
        pass

    @abstractmethod
    def get_game_ended(self, state: GameState, **kwargs) -> Union[float, int]:
        pass

    @abstractmethod
    def build_observation(self, state: GameState) -> np.ndarray:
        pass

    @abstractmethod
    def get_symmetries(self, state: GameState, pi: np.ndarray) -> List:
        pass

    @abstractmethod
    def get_hash(self, state: GameState) -> Union[str, bytes, int]:
        pass

    def close(self, state: GameState) -> None:
        pass

    def render(self, state: GameState):
        raise NotImplementedError(f"Render method not implemented for Game: {self}")