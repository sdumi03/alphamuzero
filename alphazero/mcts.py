from typing import Generic, Tuple

from alphazero.net import AlphaZeroNet
from config import ConfigDict

from games.state import GameState
from games.history import GameHistory

from utils.minmax import MinMaxStats

import numpy as np


class AlphaZeroMCTS:
    CANONICAL: bool = False  # Whether to compute the UCB formula using AlphaZero's formula (true) or MuZero's formula.

    def __init__(self, game: Generic, net: AlphaZeroNet, config: ConfigDict) -> None:
        self.game = game
        self.net = net
        self.config = config

        self.single_player = game.n_players == 1
        self.action_size = game.get_action_size()

        # Gets reinitialized at every search
        self.minmax = MinMaxStats(self.config.minimum_reward, self.config.maximum_reward)

        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Ssa = {}  # stores state transitions for s, a
        self.Rsa = {}  # stores R values for s,a
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}   # stores #times board s was visited
        self.Ps = {}   # stores initial policy (returned by neural net)
        self.Vs = {}   # stores game.getValidMoves for board s

    def clear_tree(self) -> None:
        self.Qsa, self.Ssa, self.Rsa, self.Nsa, self.Ns, self.Ps, self.Vs = [{} for _ in range(7)]

    def initialize_root(self, state: GameState, trajectory: GameHistory) -> Tuple[bytes, float]:
        network_input = trajectory.stack_observations(self.config.observation_length, state.observation)
        pi_0, v_0 = self.net.predict(network_input)

        # creo que s_0 en get_hash hay que cambiar para que sea el ix, sino deja observations
        s_0 = self.game.get_hash(state)

        # Add Dirichlet Exploration noise.
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(pi_0))
        self.Ps[s_0] = noise * self.config.exploration_fraction + (1 - self.config.exploration_fraction) * pi_0

        # Mask the prior for illegal moves, and re-normalize accordingly.
        self.Vs[s_0] = self.game.get_legal_moves(state)

        self.Ps[s_0] *= self.Vs[s_0]
        self.Ps[s_0] = self.Ps[s_0] / np.sum(self.Ps[s_0])

        # Sum of visit counts of the edges/ children and legal moves.
        self.Ns[s_0] = 0

        return s_0, v_0

    def compute_ucb(self, s: bytes, a: int, exploration_factor: float) -> float:
        """
        Compute the UCB for an edge (s, a) within the MCTS tree:

            PUCT(s, a) = MinMaxNormalize(Q(s, a)) + P(s, a) * sqrt(visits_parent / (1 + visits_s)) * exploration_factor

        Where the exploration factor is either the exploration term of MuZero (default) or a float c_1.

        Illegal edges are returned as zeros. The Q values within the tree are MinMax normalized over the
        accumulated statistics over the current tree search.

        :param s: hash Key of the current state inside the MCTS tree.
        :param a: int Action key representing the path to reach the child node from path (s, a)
        :param exploration_factor: float Pre-computed exploration factor from the MuZero PUCT formula.
        :return: float Upper confidence bound with neural network prior
        """
        if s in self.Vs and not self.Vs[s][a]:
            return 0

        visit_count = self.Nsa[(s, a)] if (s, a) in self.Nsa else 0
        q_value = self.minmax.normalize(self.Qsa[(s, a)]) if (s, a) in self.Qsa else 0
        c_children = np.max([self.Ns[s], 1e-8])  # Ensure that prior doesn't collapse to 0 if s is new.

        # Exploration
        if self.CANONICAL:
            # Standard PUCT formula from the AlphaZero paper
            ucb = self.Ps[s][a] * np.sqrt(c_children) / (1 + visit_count) * self.config.c1
        else:
            # The PUCT formula from the MuZero paper
            ucb = self.Ps[s][a] * np.sqrt(c_children) / (1 + visit_count) * exploration_factor

        ucb += q_value  # Exploitation
        return ucb

    def run_mcts(self, state: GameState, trajectory: GameHistory, temp: int = 1) -> Tuple[np.ndarray, float]:
        """
        This function performs 'num_MCTS_sims' simulations of MCTS starting from the provided root GameState.

        Before the search we only clear statistics stored inside the MinMax tree. In this way we ensure that
        reward bounds get refreshed over time/ don't get affected by strong reward scaling in past searches.
        This implementation, thus, reuses state state transitions from past searches. This may influence memory usage.

        Our estimation of the root-value of the MCTS tree search is based on a sample average of each backed-up
        MCTS value. This means that this estimate represents an on-policy estimate V^pi.

        Illegal moves are masked before computing the action probabilities.

        :param state: GameState Data structure containing the current state of the environment.
        :param trajectory: GameHistory Data structure containing the entire episode trajectory of the agent(s).
        :param temp: float Visit count exponentiation factor. A value of 0 = Greedy, +infinity = uniformly random.
        :return: tuple (pi, v) The move probabilities of MCTS and the estimated root-value of the policy.
        """
        # Refresh value bounds in the tree
        self.minmax.refresh()

        # Initialize the root variables needed for MCTS.
        s_0, v_0 = self.initialize_root(state, trajectory)

        # Aggregate root state value over MCTS back-propagated values
        v_search = sum([self._search(state, trajectory) for _ in range(self.config.num_mcts_sims - 1)])
        v = (v_0 + (v_search if self.single_player else -v_search)) / self.config.num_mcts_sims

        # MCTS Visit count array for each edge 'a' from root node 's_0'.
        counts = np.array([self.Nsa[(s_0, a)] if (s_0, a) in self.Nsa else 0 for a in range(self.action_size)])

        if temp == 0:  # Greedy selection. One hot encode the most visited paths (randomly break ties).
            move_probabilities = np.zeros(len(counts))
            move_probabilities[np.argmax(counts + np.random.randn(len(counts)) * 1e-8)] = 1
        else:
            counts = np.power(counts, 1. / temp)
            move_probabilities = counts / np.sum(counts)

        return move_probabilities, v

    def _search(self, state: GameState, trajectory: GameHistory, path: Tuple[int, ...] = tuple()) -> float:
        """
        Recursively perform MCTS search inside the actual environments with search-paths guided by the PUCT formula.

        Selection chooses an action for expanding/ traversing the edge (s, a) within the tree search.
        The exploration_factor for the PUCT formula is computed within this function for efficiency:

            exploration_factor = c1 * log(visits_s + c2 + 1) - log(c2)

        Setting AlphaMCTS.CANONICAL to true sets exploration_factor just to c1.

        If an edge is expanded, we perform a step within the environment (with action a) and observe the state
        transition, reward, and infer the new move probabilities, and state value. If an edge is traversed, we simply
        look up earlier inferred/ observed values from the class dictionaries.

        During backup we update the current value estimates of an edge Q(s, a) using an average, we additionally
        update the MinMax statistics to get reward/ value boundaries for the PUCT formula. Note that backed-up
        values get discounted for gamma < 1. For adversarial games, we negate the backed up value G_k at each backup.

        The actual search-path 'path' is kept as a debugging-variable, it currently has no practical use. This method
        may raise a recursion error if the environment creates cycles, this should be highly improbable for most
        environments. If this does occur, the environment can be altered to terminate after n visits to some cycle.

        :param state: GameState Numerical prediction of the state by the encoder/ dynamics model.
        :param trajectory: GameHistory Data structure containing all observations until the current search-depth.
        :param path: tuple of integers representing the tree search-path of the current function call.
        :return: float The backed-up discounted/ Monte-Carlo returns (dependent on gamma) of the tree search.
        :raises RecursionError: When cycles occur within the search path, the search can get stuck *ad infinitum*.
        """
        s = self.game.get_hash(state)

        ### SELECTION
        # pick the action with the highest upper confidence bound
        exploration_factor = self.config.c1 + np.log(self.Ns[s] + self.config.c2 + 1) - np.log(self.config.c2)
        confidence_bounds = np.asarray([self.compute_ucb(s, a, exploration_factor) for a in range(self.action_size)])
        a = np.flatnonzero(self.Vs[s])[np.argmax(confidence_bounds[self.Vs[s].astype(bool)])]  # Get masked argmax.

        # Default leaf node value. Future possible future reward is 0. Variable is overwritten if edge is non-terminal.
        value = 0
        if (s, a) not in self.Ssa:  ### ROLLOUT for valid moves
            next_state, reward = self.game.get_next_state(state, a, clone=True)
            s_next = self.game.get_hash(next_state)

            # Transition statistics.
            self.Rsa[(s, a)], self.Ssa[(s, a)], self.Ns[s_next] = reward, next_state, 0

            # Inference for non-terminal nodes.
            if not next_state.done:
                # Build network input for inference.
                network_input = trajectory.stack_observations(
                    self.config.observation_length, state.observation
                )
                prior, value = self.net.predict(network_input)

                # Inference statistics. Alternate value perspective due to adversary (model predicts for next player).
                self.Ps[s_next], self.Vs[s_next] = prior, self.game.get_legal_moves(next_state)
                value = value if self.single_player else -value

        elif not self.Ssa[(s, a)].done:  ### EXPANSION
            trajectory.observations.append(state.observation)   # Build up an observation trajectory inside the tree
            value = self._search(self.Ssa[(s, a)], trajectory, path + (a,))
            trajectory.observations.pop()                       # Clear tree observation trajectory when backing up

        ### BACKUP
        gk = self.Rsa[(s, a)] + self.config.gamma * value  # (Discounted) Value of the current node

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + gk) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = gk
            self.Nsa[(s, a)] = 1

        self.minmax.update(self.Qsa[(s, a)])
        self.Ns[s] += 1

        return gk if self.single_player else -gk
