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


def pad_sequences(sequences,
                  axis=1,
                  maxlen=None,
                  padding='post',
                  truncating='post',
                  value=0.):
    """用于序列填充或者截断.
    """
    if axis < 0:
        axis = np.ndim(sequences[0]) + 1 + axis
    assert axis >= 1, 'Check the padding axis, the first dimention means the batch dim, ' \
                      'which can not be padded.'
    sample_num = len(sequences)
    padding_dims = [np.shape(x)[axis-1] for x in sequences]
    if maxlen is None:
        maxlen = max(padding_dims)
    shape = np.shape(sequences[0])
    shape = (sample_num,) +  shape[:axis-1] + (maxlen, ) +  shape[axis:]
    pad_widths = [[[0, 0] for _ in range(len(shape)-1)]
                 for _ in range(sample_num)]
    assert truncating in ['pre', 'post'], "truncating mode must be 'pre' or 'post'"
    assert padding in ['pre', 'post'], "padding mode must be 'pre' or 'post'"
    pad_seq = []
    for i in range(sample_num):
        pad_dim = padding_dims[i]
        seq = np.asarray(sequences[i])
        if pad_dim > maxlen:
            seq = np.reshape(seq, (int(np.prod(shape[1:axis])), -1) + shape[axis + 1:])
            if truncating == 'pre':
                seq = np.reshape(seq[:, -maxlen:], shape[1: axis] + (maxlen,) + shape[axis + 1:])
            elif truncating == 'post':
                seq = np.reshape(seq[:, :maxlen], (shape[1: axis]) + (maxlen,) + shape[axis + 1:])
            pad_seq.append(seq)
        else:
            pad_width = pad_widths[i]
            if padding == 'pre':
                pad_width[axis-1][0] = maxlen - pad_dim
            elif padding == 'post':
                pad_width[axis-1][1] = maxlen - pad_dim
            seq = np.pad(seq, pad_width=pad_width, mode='constant', constant_values=value)
            pad_seq.append(seq)
    return np.asarray(pad_seq, dtype=object)


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

