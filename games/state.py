from dataclasses import dataclass
from typing import Generic


@dataclass
class GameState:
    canonical_state: Generic
    observation: np.ndarray
    player: int
    action: int
    done: bool
