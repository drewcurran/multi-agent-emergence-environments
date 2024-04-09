import tensorflow as tf

EPS = 1e-6


class EMAMeanStd:
    '''
        Calculates an Exponential Moving Average for each argument with
        exponential coefficient `beta`. The forward relation is:
            mean = beta * old_mean + (1.0 - beta) * observation
        The algorithm removes the bias introduced from setting ema[-1] = 0.0

        Note: `beta` parameter is defined with respect to a single observation within a batch
        if `per_element_update=True` (if a batch has 1000 elements of an observation, this is
        considered to be a 1000 updates), else it is considered to be the size of an update for a full
        batch (1 update if `per_element_update=False`).
    '''
    def __init__(self, beta, weights=None, shape=(), scope="ema", version = 1, per_element_update=False, reuse=None):
        self.one_minus_beta = 1.0 - beta
        self.shape = shape
        self.version = version
        self.per_element_update = per_element_update
        
        if weights is None:
            with tf.compat.v1.variable_scope(scope, reuse=reuse):
                # Expected value of x
                self.biased_mean = tf.compat.v1.get_variable(
                    dtype=tf.float32,
                    shape=shape,
                    initializer=tf.compat.v1.constant_initializer(0.0),
                    name="mean",
                    trainable=False)
                
                # Expected value of x^2
                self.biased_sq = tf.compat.v1.get_variable(
                    dtype=tf.float32,
                    shape=shape,
                    initializer=tf.compat.v1.constant_initializer(0.0),
                    name="sq",
                    trainable=False)

                # Weight placed on ema[-1] == 0.0 which is divided out to debias
                self.debiasing_term = tf.compat.v1.get_variable(
                    dtype=tf.float32,
                    shape=shape,
                    initializer=tf.compat.v1.constant_initializer(0.0),
                    name="debiasing_term",
                    trainable=False)
        
        else:
            self.biased_mean, self.biased_sq, self.debiasing_term = weights
        
        # Mean and standard deviation after correction due to debiasing term
        self.mean = self.biased_mean / tf.maximum(self.debiasing_term, EPS)
        self.std = self.std_from_mean_and_square(mean=self.mean, square=self.biased_sq / tf.maximum(self.debiasing_term, EPS))

    def update_op(self, x, axes=(0,)):
        scaled_weight = tf.cast(self.one_minus_beta, tf.float64)

        # Approximation for scaled weight ((1-beta)^N ~= (1 - beta) * N)
        if self.per_element_update:
            size = self.mean_std_update_size(x, axes)
            scaled_weight *= tf.cast(size, tf.float64)
        
        one = tf.constant(1.0, dtype=tf.float64)
        old_weight = one - scaled_weight
        old_weight_fp32 = tf.cast(old_weight, dtype=tf.float32)
        scaled_weight_fp32 = tf.cast(scaled_weight, dtype=tf.float32)

        return tf.group(
            # Increment the running debiasing term by the contribution of the initial observation that was discounted
            tf.Variable.assign(self.debiasing_term, tf.cast(self.interpolate(old=tf.cast(self.debiasing_term, tf.float64), new=one, old_weight=old_weight, scaled_weight=scaled_weight), dtype=tf.float32)),
            
            # Interpolation on the expected value of X
            tf.Variable.assign(self.biased_mean, self.interpolate(old=self.biased_mean, new=tf.reduce_mean(input_tensor=tf.cast(x, dtype=tf.float32), axis=axes), old_weight=old_weight_fp32, scaled_weight=scaled_weight_fp32)),
            
            # Interpolation on the expected value of X^2
            tf.Variable.assign(self.biased_sq, self.interpolate(old=self.biased_sq, new=tf.reduce_mean(input_tensor=tf.square(tf.cast(x, dtype=tf.float32)), axis=axes), old_weight=old_weight_fp32, scaled_weight=scaled_weight_fp32)),
        )
    
    def mean_std_update_size(self, x, axes):
        x_shape = tf.shape(input=x)
        x_dims_to_reduce = tf.gather(x_shape, axes)
        size = tf.reduce_prod(input_tensor=x_dims_to_reduce)
        return size

    def interpolate(self, old, new, old_weight, scaled_weight):
        return old * old_weight + new * scaled_weight

    def std_from_mean_and_square(self, mean, square):
        var_est = tf.cast(square, dtype=tf.float32) - tf.square(mean)
        return tf.sqrt(tf.maximum(var_est, 1e-2))
