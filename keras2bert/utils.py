import numpy as np
from keras2bert.backend import softmax


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
                 end_id=None,
                 max_step=None):
        self.end = end_id
        self.max_step = max_step

    def beam_search(self,
                    inputs,
                    beam_size=1):
        V = np.shape(inputs)[-1]
        beams = None
        scores = None
        complete_seqs = []
        for step in range(self.max_step):
            if step == 0:
                probs = inputs[0]
                beams = np.argpartition(probs, kth=-beam_size, axis=-1)[-beam_size:]
                beams = np.reshape(beams, (-1, 1))
                scores = np.log(inputs[0][beams])
            else:
                scores = scores + np.log(inputs[step]).reshape(1, -1)
                indices = np.argpartition(scores.reshape(-1), kth=-beam_size, axis=-1)[-beam_size:]
                row = indices // V
                col = indices % V
                scores = np.take_along_axis(scores.reshape(-1), indices, axis=-1).reshape(-1, 1)
                beams = np.concatenate([beams[row], np.reshape(col, (-1, 1))], axis=-1)
                if any(col == self.end):
                    if col[np.argmax(scores)] == self.end:
                        complete_seqs.append((beams[np.argmax(scores)], scores[np.argmax(scores)]))
                        break
                    indexs = np.where(col == self.end)
                    complete_seqs.append((beams[indexs], scores[indexs]))
                    beams = np.delete(beams, indexs, axis=0)
                    scores = np.delete(scores, indexs, axis=0)
                    beam_size -= len(indexs)
                    if beam_size == 0:
                        break
        if complete_seqs:
            return sorted(complete_seqs, key=lambda x: -x[1])[0]

        for beam, score in zip(beams, scores):
            complete_seqs.append((beam, score))
        return sorted(complete_seqs, key=lambda x: -x[1])[0]

    def random_sample(self,
                      inputs,
                      n,
                      topk=None,
                      topp=None):
        complete_seqs = []
        cur_seqs = np.empty((n, 0))
        indices = None
        assert topk is not None or topp is not None, 'topk and topp can not be None meanwhile!'
        for step in range(self.max_step):
            probs = inputs[step]
            probs = np.repeat(np.reshape(probs, (1, -1)), n, axis=0)
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
            cur_seqs = np.concatenate([cur_seqs, sample_ids], axis=-1)
            is_end = cur_seqs[:, -1] == self.end
            if sum(is_end) > 0:
                complete_seqs.append(cur_seqs[is_end])
                cur_seqs = np.delete(cur_seqs, is_end, axis=0)
                n -= sum(is_end)
                if len(cur_seqs) == 0:
                    break
                if n == 0:
                    break
        for seq in cur_seqs:
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
