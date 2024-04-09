import tensorflow as tf
import numpy as np


class EMANormalization(tf.keras.layers.Layer):
    def __init__(self, decay=0.99999, axis=-1, eps=1e-5):
        self.decay = decay
        self.axis = axis
        self.eps = eps
        super().__init__(trainable=False)
        
    def build(self, shape):
        shape = (shape[self.axis],)
        self.mean = self.add_weight(shape=shape, initializer='zeros', trainable=False, name='mean')
        self.std = self.add_weight(shape=shape, initializer='ones', trainable=False, name='std')
        self.debias = self.add_weight(initializer='zeros', trainable=False, name='debias')

    def call(self, x):
        # Calculate across unpreserved dimensions
        preserved_axis = self.axis if self.axis >= 0 else self.axis + len(x.shape)
        axes_to_reduce = [axis for axis in range(len(x.shape)) if not axis == preserved_axis]
        mean = tf.reduce_mean(x, axis=axes_to_reduce)
        sq = tf.reduce_mean(tf.square(x), axis=axes_to_reduce)
        std = tf.sqrt(sq - tf.square(mean))
        
        # Interpolate to get new values
        self.mean.assign(self.mean * (1 - self.decay) + mean * self.decay)
        self.std.assign(self.std * (1 - self.decay) + std * self.decay)
        self.debias.assign(self.debias * (1 - self.decay) + 1.0)

        # Debias values to normalize and truncate
        debiased_mean = self.mean / (self.debias + self.eps)
        debiased_std = self.std / (self.debias + self.eps) + self.eps
        normalized_inputs = (x - debiased_mean) / debiased_std
        return tf.clip_by_value(normalized_inputs, -5.0, 5.0)
        

class CircularConv1D(tf.keras.layers.Conv1D):    
    def call(self, x):
        num_pad = self.kernel_size[0] // 2
        x_reshaped = tf.reshape(x, shape=(x.shape[0] * x.shape[1],) + x.shape[2:])
        x_padded = tf.concat([x_reshaped[..., -num_pad:, :], x_reshaped, x_reshaped[..., :num_pad, :]], -2)
        out = super().call(x_padded)
        out = tf.reshape(out, shape=x.shape[:-1] + [self.filters])
        return out


class ResidualSABlock(tf.keras.layers.MultiHeadAttention):
    def call(self, x, attention_mask=None):
        x_reshaped = tf.reshape(x, (x.shape[0],) + (x.shape[1] * x.shape[2],) + (x.shape[3],))
        out = super().call(x_reshaped, x_reshaped, attention_mask=attention_mask)
        return tf.reshape(out, x.shape)


class LSTM(tf.keras.layers.LSTMCell):    
    def call(self, x, states=None):
        states = tf.unstack(states, 2, axis=0)
        x_reshaped = tf.reshape(x, (x.shape[0],) + (np.prod(x.shape[1:]),))
        out, states_out = super().call(x_reshaped, states)
        return tf.reshape(out, x.shape), tf.stack(states_out)


class FlattenOuter(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    
    def call(self, x):
        return tf.reshape(x, x.shape[:2] + (np.prod(x.shape[2:]),))


class Concatenate(tf.keras.layers.Concatenate):
    def build(self, shapes):
        shapes = self.get_shapes(shapes)
        super().build(shapes)
    
    def call(self, inputs):
        inputs = self.get_inputs(inputs)
        out = super().call(inputs)
        return out
    
    def get_shapes(self, shapes):
        if len(shapes[0]) < len(shapes[1]):
            shapes = [shapes[1], shapes[0]]
        while len(shapes[0]) > len(shapes[1]):
            shapes[1] = shapes[1][:-1] + shapes[0][-2] + shapes[1][-1:]
        return shapes
    
    def get_inputs(self, inputs):
        if len(inputs[0].shape) < len(inputs[1].shape):
            inputs = [inputs[1], inputs[0]]
        while len(inputs[0].shape) > len(inputs[1].shape):
            inputs[1] = tf.expand_dims(inputs[1], -2)
            tile_dims = [1 for _ in range(len(inputs[1].shape))]
            tile_dims[-2] = inputs[0].shape[-2]
            inputs[1] = tf.tile(inputs[1], tile_dims)
        return inputs

class EntityConcatenate(tf.keras.layers.Concatenate):
    def build(self, input_shapes):
        dims = np.max([len(shape) for shape in input_shapes])
        input_shapes = [shape if len(shape) == dims else shape[:2] + (1,) + shape[2:] for shape in input_shapes]
        super().build(input_shapes)
    
    def call(self, inputs, masks_in = []):
        dims = np.max([len(x.shape) for x in inputs])
        inputs = [x if len(x.shape) == dims else tf.expand_dims(x, 2) for x in inputs]
        out = super().call(inputs)
        mask_out = self.get_mask(masks_in, inputs)
        return out, mask_out
    
    def get_mask(self, masks, inputs):
        new_masks = []
        for mask, x in zip(masks, inputs):
            if mask is None:
                if len(x.shape) == 4:
                    new_masks.append(tf.ones(x.shape[:3], dtype=bool))
                elif len(x.shape) == 3:
                    new_masks.append(tf.ones(x.shape[:2] + [1], dtype=bool))
            else:
                new_masks.append(mask)
        return tf.concat(new_masks, axis=-1)
    

class Pooling(tf.keras.layers.Layer):
    def __init__(self, pool_type='avg_pooling'):
        self.pool_type = pool_type

        if self.pool_type == 'avg_pooling':
            self.pool_layer = tf.keras.layers.GlobalAvgPool1D()
        elif self.pool_type == 'max_pooling':
            self.pool_layer = tf.keras.layers.GlobalMaxPool1D()
        super().__init__()
    
    def build(self, shape):
        self.pool_layer.build(shape[0] * shape[1] + shape[2:])

    def call(self, x, mask=None):
        if mask is not None:
            mask = tf.expand_dims(mask, -1)
            x = x * tf.cast(mask, dtype=tf.float32)
        out = self.pool_layer(tf.reshape(x, x.shape[0] * x.shape[1] + x.shape[2:]))
        return tf.reshape(out, (x.shape[0], x.shape[1]) + x.shape[3:])