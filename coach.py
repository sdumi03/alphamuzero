from abc import ABC, abstractmethod
from collections import deque
from typing import List, Generic
from pickle import Pickler, Unpickler, HIGHEST_PROTOCOL
import os
import numpy as np

from config import ConfigDict

from arena import Arena

from games.history import GameHistory
from utils.parameter_scheduler import ParameterScheduler


class Coach(ABC):

    def __init__(self, game: Generic, net: Generic, config: ConfigDict, mcts, player) -> None:
        self.game = game
        self.net = net
        self.config = config

        self.mcts = mcts(self.game, self.net, self.config)
        self.player_arena = player(self.game)
        self.player_arena.set_variables(self.net, self.mcts, 'p1')

        if self.config.pitting:
            self.opponent_net = self.net.__class__(self.game, self.config)
            self.opponent_mcts = mcts(self.game, self.opponent_net, self.config)
            self.opponent_player_arena = player(self.game)
            self.opponent_player_arena.set_variables(self.opponent_net, self.opponent_mcts, 'p2')

        self.replay_buffer = deque(maxlen=self.config.selfplay_buffer_window)

        self.temp_schedule = ParameterScheduler(self.config)
        self.update_temperature = self.temp_schedule.build()

    @abstractmethod
    def sample_batch(self, histories: List[GameHistory]) -> List:
        raise NotImplementedError('Coach.sample_batch not implemented')

    def learn(self) -> None:
        for i in range(self.config.num_learn_iterations):
            print()
            print(f"Iteration: {i}")
            iteration_train_examples = list()

            for _ in range(self.config.num_games_per_iteration):
                self.mcts.clear_tree()
                game_history = self.execute_episode()
                iteration_train_examples.append(game_history)

                if sum(map(len, iteration_train_examples)) > self.config.max_buffer_size:
                    iteration_train_examples.pop(0)

            self.replay_buffer.append(iteration_train_examples)
            self.save_replay_buffer(i)

            replay_buffer_flattened = GameHistory.flatten(self.replay_buffer)

            self.net.save_checkpoint(folder=self.config.checkpoint, filename='temp.weights.h5')

            print('Training net')
            for _ in range(self.config.num_gradient_steps):
                batch = self.sample_batch(replay_buffer_flattened)
                self.net.train(batch)
                # self.net.monitor.log_batch(batch)

            accept = True
            if self.config.pitting:
                self.opponent_net.load_checkpoint(folder=self.config.checkpoint, filename='temp.weights.h5')

                arena = Arena(self.game, self.player_arena, self.opponent_player_arena, self.config.max_trial_moves)
                accept = arena.pitting(self.config, self.net.monitor)

            if accept:
                print('Accepting new model')
                self.net.save_checkpoint(folder=self.config.checkpoint, filename=self.get_checkpoint_file(i))
                self.net.save_checkpoint(folder=self.config.checkpoint, filename=self.config.load_folder_file[-1])
            else:
                print('Rejecting new model')
                self.net.load_checkpoint(folder=self.args.checkpoint, filename='temp.weights.h5')

    def execute_episode(self) -> GameHistory:
        game_history = GameHistory()
        state = self.game.get_initial_state()
        step = 0

        while not state.done and step < self.config.max_game_moves:
            temp = self.update_temperature(self.net.steps if self.temp_schedule.config.temp_schedule_by_weight_update else step)

            pi, v = self.mcts.run_mcts(state, game_history, temp=temp)

            state.action = np.random.choice(len(pi), p=pi)
            next_state, r = self.game.get_next_state(state, state.action)
            game_history.capture(state, pi, r, v)

            state = next_state
            step += 1

        self.game.close(state)
        game_history.terminate()
        game_history.compute_returns(gamma=self.config.gamma, n=(self.config.n_steps if self.game.n_players == 1 else None))

        return game_history

    @staticmethod
    def get_checkpoint_file(iteration: int) -> str:
        return f"checkpoint_{iteration}.weights.h5"

    def save_replay_buffer(self, iteration: int) -> None:
        folder = self.config.checkpoint
        if not os.path.exists(folder): os.makedirs(folder)

        filename = os.path.join(folder, self.get_checkpoint_file(iteration) + '.examples')
        with open(filename, "wb+") as f:
            Pickler(f, protocol=HIGHEST_PROTOCOL).dump(self.replay_buffer)

        # Don't hog up storage space and clean up old (never to be used again) data
        old_checkpoint = os.path.join(folder, self.get_checkpoint_file(iteration - 1) + '.examples')
        if os.path.isfile(old_checkpoint):
            os.remove(old_checkpoint)

    def load_replay_buffer(self) -> None:
        model_file = os.path.join(self.config.load_folder_file[0], self.config.load_folder_file[1])
        examples_file = model_file + '.examples'

        if os.path.isfile(examples_file):
            with open(examples_file, 'rb') as f:
                self.replay_buffer = Unpickler(f).load()
        else:
            raise f"Not found train examples in file {examples_file}"