from architectures.gym_arch import AlphaZeroGymArch


AlphaZeroArch = {
    # 'Hex': AlphaZeroHexNetwork,
    # 'Othello': AlphaZeroHexNetwork,
    'gym': AlphaZeroGymArch,
    # "Atari": AlphaZeroAtariNetwork
}

# Add your MuZero neural network architecture here by referencing the imported Class with a string key.
MuZeroArch = {
    # 'Hex': MuZeroHexNetwork,
    # 'Othello': MuZeroHexNetwork,
    # 'Gym': MuZeroGymNetwork,
    # 'Atari': MuZeroAtariNetwork
}


# Add different agent implementations for interacting with environments.
# Players = {
#     "ALPHAZERO": DefaultAlphaZeroPlayer,
#     "MUZERO": DefaultMuZeroPlayer,
#     "BLIND_MUZERO": BlindMuZeroPlayer,
#     "RANDOM": RandomPlayer,
#     "DETERMINISTIC": DeterministicPlayer,
#     "MANUAL": ManualPlayer
# }