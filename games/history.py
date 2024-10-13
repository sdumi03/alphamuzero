from dataclasses import dataclass, field
from typing import Optional

from games.state import GameState

import numpy as np


@dataclass
class GameHistory:
    observations: list      = field(default_factory=list)   # o_t: State Observations
    players: list           = field(default_factory=list)   # p_t: Current player
    probabilities: list     = field(default_factory=list)   # pi_t: Probability vector of MCTS for the action
    search_returns: list    = field(default_factory=list)   # v_t: MCTS value estimation
    rewards: list           = field(default_factory=list)   # u_t+1: Observed reward after performing a_t+1
    actions: list           = field(default_factory=list)   # a_t+1: Action leading to transition s_t -> s_t+1
    observed_returns: list  = field(default_factory=list)   # z_t: Training targets for the value function
    terminated: bool        = False                         # Whether the environment has terminated

    def __len__(self) -> int:
        return len(self.observations)

    def capture(self, state: GameState, pi: np.ndarray, r: float, v: float) -> None:
        self.observations.append(state.observation)
        self.players.append(state.player)
        self.actions.append(state.action)
        self.probabilities.append(pi)
        self.rewards.append(r)
        self.search_returns.append(v)

    def terminate(self) -> None:
        self.probabilities.append(np.zeros_like(self.probabilities[-1]))
        self.rewards.append(0)         # Reward past u_T
        self.search_returns.append(0)  # Bootstrap: Future possible reward = 0
        self.terminated = True

    def refresh(self) -> None:
        all([x.clear() for x in vars(self).values() if type(x) == list])
        self.terminated = False

    def compute_returns(self, gamma: float = 1, n: Optional[int] = None) -> None:
        self.observed_returns = list()

        if n is None:
            self.observed_returns.append(self.rewards[-2])

            for i in range(1, len(self)):
                self.observed_returns.append(-self.observed_returns[i - 1])

            self.observed_returns = self.observed_returns[::-1] + [0]

        else:
            for t in range(len(self.rewards)):
                horizon = np.min([t + n, len(self.rewards) - 1])

                discounted_rewards = [np.power(gamma, k - t) * self.rewards[k] for k in range(t, horizon)]
                bootstrap = (np.power(gamma, horizon - t) * self.search_returns[horizon]) if horizon <= t + n else 0

                self.observed_returns.append(sum(discounted_rewards) + bootstrap)

    def stack_observations(self, length: int, current_observation: Optional[np.ndarray] = None, t: Optional[int] = None) -> np.ndarray:
        if length <= 1:
            if current_observation is not None:
                return current_observation
            elif t is not None:
                return self.observations[np.min([t, len(self) - 1])]
            else:
                return self.observations[-1]

        if t is None:
            # If current observation is also None, then t needs to both index and slice self.observations:
            # for len(self) indexing will throw an out of bounds error when current_observation is None.
            # for len(self) - 1, if current_observation is NOT None, then the trajectory wil omit a step.
            # Proof: t = len(self) - 1 --> list[:t] in {i, ..., t-1}.
            t = len(self) - (1 if current_observation is None else 0)

        if current_observation is None:
            current_observation = self.observations[t]

        # Get a trajectory of states of 'length' most recent observations until time-point t.
        # Trajectories sampled beyond the end of the game are simply repeats of the terminal observation.
        if t > len(self):
            terminal_repeat = [current_observation] * (t - len(self))
            trajectory = self.observations[:t][-(length - len(terminal_repeat)):] + terminal_repeat
        else:
            trajectory = self.observations[:t][-(length - 1):] + [current_observation]

        if len(trajectory) < length:
            prefix = [np.zeros_like(current_observation) for _ in range(length - len(trajectory))]
            trajectory = prefix + trajectory

        return np.concatenate(trajectory, axis=-1)  # Concatenate along channel dimension.

