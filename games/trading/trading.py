from typing import Tuple
from dataclasses import dataclass

from games.game import Game
from games.state import GameState

import yfinance as yf
import numpy as np


@dataclass
class TradingState(GameState):
    ix: int


class TradingGame(Game):

    def __init__(self) -> None:
        super().__init__(n_players=1)
        self.pred_len = 20

    def get_dimensions(self) -> Tuple[int, ...]:
        return 1, 1, 6

    def get_action_size(self) -> int:
        return 2

    def get_initial_state(self) -> TradingState:
        df_history = yf.Ticker('MSFT').history(period='max')
        ix = np.random.randint(len(df_history) - (self.pred_len + 1), size=1)[0]

        next_state = TradingState(
            canonical_state=df_history.iloc[ix : ix + self.pred_len],
            observation=None,
            player=1,
            action=-1,
            done=False,
            ix=0
        )
        next_state.observation = self.build_observation(next_state)

        return next_state

    def get_next_state(self, state: TradingState, action: int) -> Tuple[TradingState, float]:
        if action == 0: position = 1
        if action == 1: position = -1

        state.ix += 1

        o = state.canonical_state.iloc[state.ix]['Open']
        c = state.canonical_state.iloc[state.ix]['Close']
        if c > o:
            moved = 1
        else:
            moved = -1

        reward = position * moved

        # no se si asi o con self.pred_len - 1
        done = state.ix >= self.pred_len

        next_state = TradingState(
            canonical_state=state.canonical_state,
            observation=None,
            player=1,
            action=action,
            done=done,
            ix=state.ix
        )
        next_state.observation = self.build_observation(next_state)

        return next_state, reward

    def get_legal_moves(self, state: TradingState) -> np.ndarray:
        return np.ones(2)

    def get_game_ended(self, state: TradingState) -> int:
        return int(state.done)

    def get_symmetries(self, state: TradingState, pi: np.ndarray) -> List:
        return [(state.observation, pi)]

    def build_observation(self, state: TradingState) -> np.ndarray:
        o = state.canonical_state.iloc[state.ix]['Open']
        h = state.canonical_state.iloc[state.ix]['High']
        l = state.canonical_state.iloc[state.ix]['Low']
        c = state.canonical_state.iloc[state.ix]['Close']
        v = state.canonical_state.iloc[state.ix]['Close'] - state.canonical_state.iloc[state.ix]['Open']
        m = state.canonical_state.iloc[state.ix]['High'] - state.canonical_state.iloc[state.ix]['Low']

        return np.asarray([[o, h, l, c, v, m]])

    def get_hash(self, state: TradingState) -> bytes:
        return np.asarray(state.observation).tobytes()

    def close(self, state: TradingState) -> None:
        pass

    def render(self, state: TradingState) -> None:
        pass