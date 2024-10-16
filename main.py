import argparse

from typing import Generic

from config import ConfigDict
from learn import learn_alphazero, learn_muzero

from games import Games

import tensorflow as tf

# para el algoritmo de mcts
from sys import setrecursionlimit
setrecursionlimit(2000)


def main():
    parser = argparse.ArgumentParser(
        description='A MuZero and AlphaZero implementation in Tensorflow'
    )

    modes = ['train', 'test']
    parser.add_argument(
        '--mode',
        choices=modes,
        default='train',
        required=True,
        help='Choose a mode (train or test)'
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Set a json config file'
    )

    args = parser.parse_args()

    config = ConfigDict.load_json(args.config)

    game = Games[config.game]()

    if args.mode == 'train':
        device = tf.DeviceSpec(device_type=config.device, device_index=0)

        with tf.device(device.to_string()):
            if config.algorithm == 'ALPHAZERO':
                learn_alphazero(game, config)

            elif config.algorithm == 'MUZERO':
                learn_muzero(game, config)

            else:
                raise NotImplementedError(f"Cannot train on algorithm '{content.algorithm}'")

    if args.mode == 'test':
        print('Testing')

if __name__ == '__main__':
    main()