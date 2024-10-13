from typing import Tuple, List
from dataclasses import dataclass
from copy import deepcopy

from games.state import GameState
from games.game import Game

import numpy as np
import gym


@dataclass
class GymState(GameState):
    env: Env


class GymGame(Game):

    def __init__(self, env_name: str, wrappers: Tuple = ()) -> None:
        super().__init__(n_players=1)
        self.env_name = env_name
        self.wrappers = wrappers

        dummy = gym.make(env_name)
        for w in wrappers:
            dummy = w(dummy)

        self.dimensions = dummy.observation_space.shape
        self.actions = dummy.action_space.n

    def get_dimensions(self, **kwargs) -> Tuple[int, ...]:
        return self.dimensions if len(self.dimensions) > 1 else (1, 1, *self.dimensions)

    def get_action_size(self) -> int:
        return self.actions

    def get_initial_state(self) -> GymState:
        env = gym.make(self.env_name)
        for w in self.wrappers:
            env = w(env)

        next_state = GymState(canonical_state=env.reset(), observation=None, action=-1, done=False, player=1, env=env)
        next_state.observation = self.build_observation(next_state)

        return next_state

    def get_next_state(self, state: GymState, action: int, **kwargs) -> Tuple[GymState, float]:
        def next_env(old_state: GymState, clone: bool = False):  # Macro for cloning the state
            return deepcopy(old_state.env) if clone else old_state.env

        env = next_env(state, **kwargs)
        raw_observation, reward, done, info = env.step(action)

        next_state = GymState(
            canonical_state=raw_observation,
            observation=None,
            action=action,
            done=done,
            player=1,
            env=env
        )
        next_state.observation = self.build_observation(next_state)

        return next_state, reward

    def get_legal_moves(self, state: GymState) -> np.ndarray:
        return np.ones(self.actions)

    def get_game_ended(self, state: GymState, **kwargs) -> int:
        return int(state.done)

    def get_symmetries(self, state: GymState, pi: np.ndarray, **kwargs) -> List:
        return [(state.observation, pi)]

    def build_observation(self, state: GymState, **kwargs) -> np.ndarray:
        return state.canonical_state if len(self.dimensions) > 1 else np.asarray([[state.canonical_state]])

    def get_hash(self, state: GymState) -> bytes:
        return np.asarray(state.canonical_state).tobytes()

    def close(self, state: GymState) -> None:
        state.env.close()

    def render(self, state: GymState) -> None:
        state.env.render()