from typing import List, Tuple

from games.history import GameHistory

import numpy as np


def sample_batch(
    list_of_histories: List[GameHistory],
    n: int,
    prioritize: bool = False,
    alpha: float = 1.0,
    beta: float = 1.0
    ) -> Tuple[List[Tuple[int, int]], List[float]]:
    """
    Generate a sample specification from the list of GameHistory object using uniform or prioritized sampling.
    Along with the generated indices, for each sample/ index a scalar is returned for the loss function during
    backpropagation. For uniform sampling this is simply w_i = 1 / N (Mean) for prioritized sampling this is
    adjusted to

        w_i = (1/ (N * P_i))^beta,

    where P_i is the priority of sample/ index i, defined as

        P_i = (p_i)^alpha / sum_k (p_k)^alpha, with p_i = |v_i - z_i|,

    and v_i being the MCTS search result and z_i being the observed n-step return.
    The w_i is the Importance Sampling ratio and accounts for some of the sampling bias.

    WARNING: Sampling with replacement is performed if the given batch-size exceeds the replay-buffer size.

    :param list_of_histories: List of GameHistory objects to sample indices from.
    :param n: int Number of samples to generate == batch_size.
    :param prioritize: bool Whether to use prioritized sampling
    :param alpha: float Exponentiation factor for computing priorities, high = uniform, low = greedy
    :param beta: float Exponentiation factor for the Importance Sampling ratio.
    :return: List of tuples indicating a sample, the first index in the tuple specifies which GameHistory object
             within list_of_histories is chosen and the second index specifies the time point within that GameHistory.
             List of scalars containing either the Importance Sampling ratio or 1 / N to scale the network loss with.
    """
    lengths = list(map(len, list_of_histories))   # Map the trajectory length of each Game

    sampling_probability = None                   # None defaults to uniform in np.random.choice
    sample_weight = np.ones(np.sum(lengths))      # 1 / N. Uniform weight update strength over batch.

    if prioritize or alpha == 0:
        errors = np.array([np.abs(h.search_returns[i] - h.observed_returns[i])
                           for h in list_of_histories for i in range(len(h))])

        mass = np.power(errors, alpha)
        sampling_probability = mass / np.sum(mass)

        # Adjust weight update strength proportionally to IS-ratio to account for sampling bias.
        sample_weight = np.power(n * sampling_probability, beta)

    # Sample with prioritized / uniform probabilities sample indices over the flattened list of GameHistory objects.
    flat_indices = np.random.choice(a=np.sum(lengths), size=n, replace=(n > np.sum(lengths)), p=sampling_probability)

    # Map the flat indices to the correct histories and history indices.
    history_index_borders = np.cumsum(lengths)
    history_indices = [(np.sum(i >= history_index_borders), i) for i in flat_indices]

    # Of the form [(history_i, t), ...] \equiv history_it
    sample_coordinates = [(h_i, i - np.r_[0, history_index_borders][h_i]) for h_i, i in history_indices]
    # Extract the corresponding IS loss scalars for each sample (or simply N x 1 / N if non-prioritized)
    sample_weights = sample_weight[flat_indices]

    return sample_coordinates, sample_weights