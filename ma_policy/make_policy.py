import os
import numpy as np
import tensorflow as tf
import logging
import sys
import traceback


def shape_list(x):
    ps = x.get_shape().as_list()
    ts = tf.shape(input=x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

def replace_base_scope(var_name, new_base_scope):
    split = var_name.split('/')
    split[0] = new_base_scope
    return os.path.normpath('/'.join(split))

def adjust_weight_shape(var, weight):
    var_name = os.path.normpath(var.name)
    var_shape = tuple(shape_list(var))
    new_weight = weight
    for dim in range(len(var_shape)):
        if var_shape[dim] > weight.shape[dim]:
            logging.warning(f"{var_name}: Weights of shape {weight.shape} extended to match {var_shape}.")
            added_weight = np.zeros(weight.shape[:dim] + (var_shape[dim] - weight.shape[dim],) + weight.shape[dim + 1:])
            #added_weight = np.random.randn(*(weight.shape[:dim] + (var_shape[dim] - weight.shape[dim],) + weight.shape[dim + 1:]))
            new_weight = np.append(weight, added_weight, axis=dim)
        if var_shape[dim] < weight.shape[dim]:
            logging.warning(f"{var_name}: Weights of shape {weight.shape} truncated to match {var_shape}.")
            new_weight = np.delete(weight, slice(var.shape[0], weight.shape[0]), axis=0)
    return new_weight

def load_variables(policy, weights):
    weights = {os.path.normpath(key): value for key, value in weights.items()}
    weights = {replace_base_scope(key, policy.scope): value for key, value in weights.items()}
    assign_ops = []
    for var in policy.get_variables():
        var_name = os.path.normpath(var.name)
        if var_name not in weights:
            logging.warning(f"{var_name}: Variable not found in weights dict. This will be reinitialized.")
            tf.compat.v1.get_default_session().run(var.initializer)
        else:
            try:
                assert len(np.array(shape_list(var))) == len(np.array(weights[var_name].shape))
                weights[var_name] = adjust_weight_shape(var, weights[var_name])
                assign_ops.append(var.assign(weights[var_name]))
            except Exception:
                traceback.print_exc(file=sys.stdout)
                print(f"{var_name}: Error assigning weights of shape {weights[var_name].shape} to {tuple(shape_list(var))}")
                sys.exit()
    tf.compat.v1.get_default_session().run(assign_ops)
