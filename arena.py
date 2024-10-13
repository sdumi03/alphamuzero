from typing import Generic, Tuple

from config import ConfigDict

import numpy as np


class Arena:

    def __init__(self, game: Generic, player1, player2, max_trial_length: int = 1_000) -> None:
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.max_trial_length = max_trial_length

    def play_turn_based_game(self, first_player, second_player) -> int:
        players = [second_player, None, first_player]
        state = self.game.get_initial_state()
        r = step = 0

        while not state.done and step < self.max_trial_length:
            state.action = players[state.player + 1].act(state)

            valid_moves = self.game.get_legal_moves(state)
            if not valid_moves[state.action]:
                state.action = len(valid_moves)

            # Capture an observation for both players
            players[state.player + 1].observe(state)
            players[1 - state.player].observe(state)

            state, r = self.game.get_next_state(state, state.action)
            step += 1

        self.game.close(state)

        return -state.player * r

    def play_turn_based_games(self, num_games: int) -> Tuple[int, int, int]:
        results = list()
        for _ in range(num_games):
            self.player1.refresh()
            self.player2.refresh()

            results.append(self.play_turn_based_game(self.player1, self.player2))

        one_won = np.sum(np.array(results) == 1).item()
        two_won = np.sum(np.array(results) == -1).item()

        results = list()
        for _ in range(num_games):
            self.player1.refresh()
            self.player2.refresh()

            results.append(self.play_turn_based_game(self.player2, self.player1))

        one_won += np.sum(np.array(results) == -1).item()
        two_won += np.sum(np.array(results) == 1).item()

        return one_won, two_won, (one_won + two_won - num_games * 2)

    def play_game(self, player) -> float:
        state = self.game.get_initial_state()
        step = score = 0

        while not state.done and step < self.max_trial_length:
            state.action = player.act(state)

            player.observe(state)

            state, r = self.game.get_next_state(state, state.action)

            score += r
            step += 1

        self.game.close(state)

        return score

    def play_games(self, num_games: int, player) -> np.ndarray:
        scores = list()

        for _ in range(num_games):
            player.refresh()
            scores.append(self.play_game(player))

        return np.array(scores)

    def pitting(self, config: ConfigDict, logger: Monitor) -> bool:
        if self.game.n_players == 1:
            p1_score = self.play_games(config.pitting_trials, self.player1)
            p2_score = self.play_games(config.pitting_trials, self.player2)

            wins, draws = np.sum(p1_score > p2_score), np.sum(p1_score == p2_score)
            losses = config.pitting_trials - (wins + draws)

            # logger.log(p1_score.mean(), "Average Trial Reward")
            # logger.log_distribution(p1_score, "Trial Reward")

            # print(f'AVERAGE PLAYER 1 SCORE: {p1_score.mean()} ; AVERAGE PLAYER 2 SCORE: {p2_score.mean()}')
        else:
            losses, wins, draws = self.play_turn_based_games(config.pitting_trials)

        # print(f'CHAMPION/CONTENDER WINS : {wins} / {losses} ; DRAWS : {draws} ; '
        #       f'NEW CHAMPION ACCEPTANCE RATIO : {config.pit_acceptance_ratio}')

        return (
            losses + wins > 0 and wins / (losses + wins) >= config.pit_acceptance_ratio
        ) or config.pit_acceptance_ratio == 0