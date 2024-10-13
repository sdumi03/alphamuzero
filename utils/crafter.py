from keras.layers import LeakyReLU, Activation, BatchNormalization, Dropout, Conv2D, Dense, Flatten, Lambda
from keras import backend as k


class Crafter:

    def __init__(self, config):
        self.config = config
        if config.activation != 'leakyrelu':
            self.activation = lambda: Activation(config.activation)
        else:
            self.activation = lambda: LeakyReLU(alpha=0.2)

    def conv_residual_tower(self, n: int, x, left_n: int = 2, right_n: int = 0, use_bn: bool = True):
        assert left_n > 0, 'Residual network must have at least a conv block larger than 0'

        if n > 0:
            left = self.conv_tower(left_n - 1, x, use_bn)

            if left_n - 1 > 0:
                left = (Conv2D(self.config.num_channels, 3, padding='same', use_bias=(not use_bn))(left))

                if use_bn:
                    left = BatchNormalization()(left)

            right = self.conv_tower(right_n - 1, x, use_bn)

            if right_n - 1 > 0:
                right = (Conv2D(self.config.num_channels, 3, padding='same', use_bias=(not use_bn))(right))

                if use_bn:
                    right = BatchNormalization()(right)

            merged = Lambda(lambda var: k.sum(var, axis=0))([left, right])
            out_tensor = self.activation()(merged)

            return self.conv_residual_tower(n - 1, out_tensor, left_n, right_n, use_bn)

        return x

    def conv_tower(self, n: int, x, use_bn: bool = True):
        if n > 0:
            tensor = Conv2D(self.config.num_channels, 3, padding='same', use_bias=(not use_bn))(x)
            if use_bn:
                tensor = BatchNormalization()(tensor)
            return self.conv_tower(n - 1, self.activation()(tensor), use_bn)
        return x

    def dense_sequence(self, n: int, x):
        if n > 0:
            return self.dense_sequence(n - 1, Dropout(self.config.dropout)(self.activation()(
                Dense(self.config.size_dense)(x))))
        return x

    def build_conv_block(self, tensor_in, use_bn: bool = True):
        conv_block = self.conv_tower(self.config.num_towers, tensor_in, use_bn)
        flattened = Flatten()(conv_block)
        fc_sequence = self.dense_sequence(self.config.num_dense, flattened)
        return fc_sequence