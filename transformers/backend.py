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


def sinusoidal_embeddings(pos, dim, base=10000):
    assert dim % 2 == 0
    indices = K.arange(0, dim // 2, dtype=K.floatx())
    indices = K.pow(K.cast(base, K.floatx()), -2 * indices / dim)
    embeddings = tf.einsum('b m, d -> b m d', pos, indices)
    embeddings = K.stack((K.sin(embeddings), K.cos(embeddings)), axis=-1)
    return K.reshape(embeddings, (-1, K.int_shape(embeddings)[1], np.prod(K.int_shape(embeddings)[-2:])))


class Sinusoidal(keras.initializers.Initializer):
    """Sinusoidal 位置编码
    https://arxiv.org/abs/1706.03762
    """
    def __call__(self, shape, dtype=None):
        size, dim = shape
        return sinusoidal_embeddings(K.arange(size, dtype=K.floatx())[None], dim)[0]


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

K.logsumexp = getattr(K, 'logsumexp', None) or tf.reduce_logsumexp
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

