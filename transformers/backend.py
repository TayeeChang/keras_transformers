import os
import sys
import numpy as np
import tensorflow as tf

if 'TF_KERAS' in os.environ and os.environ['TF_KERAS'] != '0':
    import tensorflow
    sys.modules['keras'] = tensorflow.keras

import keras
import keras.backend as K


def gelu_tanh(x):
    """gelu激活函数的近似算法.
     # 引用:
        https://arxiv.org/pdf/1606.08415.pdf
    """
    return 0.5 * x * (1.0 + K.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x * x * x)))


def gelu_erf(x):
    """gelu激活函数的精确算法.
     # 引用:
        https://arxiv.org/pdf/1606.08415.pdf
    """
    return 0.5 * x * (1.0 + tf.math.erf(x / np.sqrt(2)))


def set_gelu(name):
    """选择gelu精确算法还是近似算法
    """
    name = name.lower()
    assert name in ['tanh', 'erf'], "set_gelu name must be 'tanh' or 'erf'"
    if name == 'erf':
        keras.utils.get_custom_objects()['gelu'] = gelu_erf
    else:
        keras.utils.get_custom_objects()['gelu'] = gelu_tanh


def softmax(x, axis=-1):
    """自定义softmax激活函数.
    """
    x = x - np.max(x, axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def mask_sequences(x, mask, axis=1, value=None):
    """用于mask序列.
    """
    if mask is None:
        return x
    mask = K.cast(mask, K.dtype(x))
    if value == '-inf':
        value = -K.infinity()
    if axis < 0:
        axis = K.ndim(x) + axis
    for _ in range(axis-1):
        mask = K.expand_dims(mask, 1)
    for _ in range(K.ndim(x) - K.ndim(mask)):
        mask = K.expand_dims(mask, -1)
    x = x * mask + value * (1 - mask)
    return x


class Sinusoidal(keras.initializers.Initializer):
    """Sin-Cos 位置嵌入初始化器.
    # 引用:
        https://arxiv.org/abs/1706.03762
    """
    def __call__(self, shape, dtype=None):
        """Sin-Cos形式的位置向量
        """
        seq_len, output_dim = shape
        embeddings = np.zeros(shape)
        for pos in range(seq_len):
            for i in range(output_dim // 2):
                theta = pos / np.power(10000, 2. * i / output_dim)
                embeddings[pos, 2 * i] = np.sin(theta)
                embeddings[pos, 2 * i + 1] = np.cos(theta)
        return embeddings


_INFINITY = 1e12


def infinity():
    return _INFINITY


def set_infinity(value):
    global _INFINITY
    _INFINITY = value


def identity(x):
    return x


symbolic = identity
if hasattr(K, 'symbolic'):
    symbolic = K.symbolic

K.infinity = infinity
K.set_infinity = set_infinity
K.symbolic = symbolic

custom_objects = {
    "gelu": gelu_erf,
    "gelu_tanh": gelu_tanh,
    "gelu_erf": gelu_erf,
    "Sinusoidal": Sinusoidal,
}
keras.utils.get_custom_objects().update(custom_objects)

