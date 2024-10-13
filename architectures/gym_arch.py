from typing import Generic

from config import ConfigDict

from keras.layers import Dense, Input, Reshape, Concatenate, Activation
from keras.models import Model
from keras.optimizers import Adam

from utils.crafter import Crafter
# from utils.minmax import MinMaxScaler


class AlphaZeroGymArch:

    def __init__(self, game: Generic, config: ConfigDict):
        self.x, self.y, self.planes = game.get_dimensions()
        self.action_size = game.get_action_size()
        self.config = config
        self.crafter = Crafter(config)

        # s: batch_size x time x state_x x state_y
        self.observation_history = Input(shape=(
            self.x,
            self.y,
            self.planes * self.config.observation_length
        ))

        # a: one hot encoded vector of shape batch_size x (state_x * state_y)
        self.action_tensor = Input(shape=(self.action_size, ))

        observations = Reshape(
            (self.x * self.y * self.planes * self.config.observation_length, )
        )(self.observation_history)

        self.pi, self.v = self.build_predictor(observations)

        self.model = Model(inputs=self.observation_history, outputs=[self.pi, self.v])

        opt = Adam(self.config.lr_init)
        if self.config.support_size > 0:
            self.model.compile(loss=['categorical_crossentropy'] * 2, optimizer=opt)
        else:
            self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=opt)

    def build_predictor(self, observations):
        fc_sequence = self.crafter.dense_sequence(self.config.num_dense, observations)

        pi = Dense(self.action_size, activation='softmax', name='pi')(fc_sequence)

        if self.config.support_size == 0:
            v = Dense(1, activation='linear', name='v')(fc_sequence)
        else:
            v = Dense(self.config.support_size * 2 + 1, activation='softmax', name='v')(fc_sequence)

        return pi, v
