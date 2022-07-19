import numpy as np
from transformers.backend import softmax


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
    return np.asarray(pad_seq)


class DataGenerator(object):
    """定义数据加载器，用于生成训练批数据。
    """
    def __init__(self,
                 data,
                 batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.indexs = np.arange(len(self.data))
        self.steps = int(np.ceil(len(self.data) / self.batch_size))

    def __len__(self):
        return self.steps

    def __iter__(self):
        raise NotImplementedError

    def _batch_builder(self, index):
        indexs = self.indexs[index * self.batch_size:(index+1) * self.batch_size]
        data_batch = [self.data[index] for index in indexs]
        return data_batch

    @property
    def batch_generator(self):
        for i in range(self.steps):
            yield self._batch_builder(i)

    def fit_generator(self, random=True):
        while True:
            if random:
                np.random.shuffle(self.indexs)
            for d in self.__iter__():
                yield d

                
class AutoRegressiveDecoder(object):
    """自回归解码器
        * 波束搜索
        * 随机采样
    """
    def __init__(self,
                 start_id=None,
                 end_id=None,
                 max_step=None):
        self.start_id = start_id
        self.end = end_id
        self.max_step = max_step

    def predict(self, inputs, outputs):
        raise NotImplementedError

    def beam_search(self,
                    inputs,
                    beam_size=1):
        outputs = np.array([[self.start_id]]) if self.start_id is not None else np.empty((1, 0), dtype=int)
        scores = np.zeros(1)
        complete_seqs = []
        for step in range(self.max_step):
            probs = self.predict(inputs, outputs) # (V, )
            V = probs.shape[-1]
            if step == 0:
                inputs = [np.repeat(input, beam_size, axis=0) for input in inputs] # (B, S)
            scores = scores + np.log(probs + 1e-12).reshape(1, -1) # (1, v), (B, V)
            indices = np.argpartition(scores.reshape(-1), kth=-beam_size, axis=None)[-beam_size:] # (B, )
            row = indices // V # (B, )
            col = indices % V # (B, )
            scores = np.take_along_axis(scores.reshape(-1), indices, axis=-1).reshape(-1, 1) # (B, 1)
            outputs = np.concatenate([outputs[row], np.reshape(col, (-1, 1))], axis=-1) # (B, t)
            if any(np.equal(col, self.end)):
                if col[np.argmax(scores)] == self.end:
                    complete_seqs.append((outputs[np.argmax(scores)], scores[np.argmax(scores)]))
                    break
                indexs = np.where(col == self.end)
                complete_seqs.append((outputs[indexs], scores[indexs]))
                outputs = np.delete(outputs, indexs, axis=0)
                scores = np.delete(scores, indexs, axis=0)
                beam_size -= len(indexs)
                if beam_size == 0:
                    break
        if complete_seqs:
            return sorted(complete_seqs, key=lambda x: -x[1])[0]

        for output, score in zip(outputs, scores):
            complete_seqs.append((output, score))
        return sorted(complete_seqs, key=lambda x: -x[1])[0]

    def random_sample(self,
                      inputs,
                      n,
                      topk=None,
                      topp=None):
        outputs = np.array([[self.start_id]]) if self.start_id is not None else np.empty((1, 0), dtype=int)
        complete_seqs = []
        indices = None
        assert topk is not None or topp is not None, 'topk and topp can not be None meanwhile!'
        for step in range(self.max_step):
            probs = self.predict(inputs, outputs)  # (v, )
            if step == 0:
                inputs = [np.repeat(input, n, axis=0) for input in inputs] # (n, S)
                outputs = [np.repeat(outputs, n, axis=0)] # (n, t)
                probs = np.repeat(np.reshape(probs, (1, -1)), n, axis=0) # (n, v)
            if topk:
                indices = np.argpartition(probs, kth=-topk, axis=-1)
                probs = np.take_along_axis(probs, indices, axis=-1)
                probs[:, :-topk] = float('-inf')
                probs = softmax(probs, axis=-1)
            if topp:
                indices = np.argsort(probs, axis=-1)
                probs = np.take_along_axis(probs, indices, axis=-1)
                filter = np.cumsum(probs >= topp, axis=-1)
                filter = np.roll(filter, 1, axis=-1)
                filter[:, 0] = False
                probs[filter] = -np.float('-inf')
                probs = softmax(probs, axis=-1)

            sample_func = lambda p: np.random.choice(len(p), p=p)
            sample_ids = np.apply_along_axis(sample_func, -1, probs)
            sample_ids = np.reshape(sample_ids, (-1, 1))
            sample_ids = np.take_along_axis(indices, sample_ids, axis=-1)
            outputs = np.concatenate([outputs, sample_ids], axis=-1)
            is_end = outputs[:, -1] == self.end
            if sum(is_end) > 0:
                complete_seqs.append(outputs[is_end])
                outputs = np.delete(outputs, is_end, axis=0)
                n -= sum(is_end)
                if len(outputs) == 0:
                    break
                if n == 0:
                    break
        for seq in outputs:
            complete_seqs.append(seq)
        return complete_seqs

    
def viterbi_decode(nodes,
                   trans,
                   start_id=None,
                   end_id=None):
    """viterbi算法求最优路径
    node.shape=(seq_len, num_labels)
    trans.shape=(num_labels, num_labels)
    本质是动态规划.
    """
    if start_id:
        nodes[0, :start_id] = -1e8
        nodes[0, start_id + 1:] = -1e8
    if end_id:
        nodes[-1, :end_id] = -1e8
        nodes[-1, end_id + 1:] = -1e8

    seq_len, num_labels = len(nodes), len(trans)
    scores = nodes[0].reshape((-1, 1))
    labels = np.arange(num_labels).reshape((-1, 1))
    paths = labels
    for i in range(1, seq_len):
        scores = scores + trans + nodes[i].reshape((1, -1))
        idxs = np.argmax(scores, axis=0)
        scores = np.max(scores, axis=0).reshape(-1, 1)
        paths = np.concatenate([paths[idxs], labels], axis=-1)
    return paths[np.argmax(scores)]


if __name__ == "__main__":
    pass
