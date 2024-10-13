from dataclasses import dataclass
from typing import Generic

import numpy as np


@dataclass
class GameState:
    canonical_state: Generic
    observation: np.ndarray
    player: int
    action: int
    done: bool
