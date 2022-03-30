import unicodedata
import codecs
import numpy as np


def load_vocab(vocab_path):
    """加载词典
    """
    token_dict = {}
    with codecs.open(vocab_path, 'r', 'utf-8') as infile:
        for line in infile:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict


class Tokenizer(object):
    """Bert分词器
    """
    TOKEN_CLS = '[CLS]'
    TOKEN_SEP = '[SEP]'
    TOKEN_UNK = '[UNK]'
    TOKEN_PAD = '[PAD]'

    def __init__(self,
                 vocab_path,
                 token_cls=TOKEN_CLS,
                 token_sep=TOKEN_SEP,
                 token_unk=TOKEN_UNK,
                 token_pad=TOKEN_PAD,
                 do_lower_case=True):
        self._token_dict = load_vocab(vocab_path)
        self._token_dict_inv = {v: k for k, v in self._token_dict.items()}
        self._token_cls = token_cls
        self._token_sep = token_sep
        self._token_unk = token_unk
        self._token_pad = token_pad
        self._do_lower_case = do_lower_case

    @staticmethod
    def _truncate(first_tokens, second_tokens=None, maxlen=None):
        if maxlen is None:
            return
        if second_tokens is not None:
            while True:
                total_len = len(first_tokens) + len(second_tokens)
                if total_len <= maxlen - 3:
                    return
                if len(first_tokens) <= len(second_tokens):
                    second_tokens.pop()
                else:
                    first_tokens.pop()
        else:
            del first_tokens[maxlen - 2:]

    def _pack(self, first_tokens, second_tokens=None):
        first_packed_tokens = [self._token_cls] + first_tokens + [self._token_sep]
        if second_tokens is not None:
            second_packed_tokens = second_tokens + [self._token_sep]
            return first_packed_tokens + second_packed_tokens, len(first_packed_tokens), len(second_packed_tokens)
        else:
            return first_packed_tokens, len(first_packed_tokens), 0

    def convert_tokens_to_ids(self, tokens):
        unk_id = self._token_dict.get(self._token_unk)
        return [self._token_dict.get(token, unk_id) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        unk = self._token_unk
        return [self._token_dict_inv.get(id, unk) for id in ids]

    def convert_token_to_id(self, token):
        return self._token_dict.get(token, self._token_dict[self.TOKEN_UNK])

    def convert_id_to_token(self, id):
        return self._token_dict_inv.get(id, self.TOKEN_UNK)

    def tokenize(self, first, second=None):
        """实现分词功能
        """
        first_tokens = self._tokenize(first)
        second_tokens = self._tokenize(second) if second is not None else None
        tokens, _, _ = self._pack(first_tokens, second_tokens)
        return tokens

    def _encode(self, first, second=None, max_len=None):
        first_tokens = self._tokenize(first)
        second_tokens = self._tokenize(second) if second is not None else None
        self._truncate(first_tokens, second_tokens, max_len)
        tokens, first_len, second_len = self._pack(first_tokens, second_tokens)
        token_ids = self.convert_tokens_to_ids(tokens)
        segment_ids = [0] * first_len + [1] * second_len
        return token_ids, segment_ids

    def encode(self, text1, text2=None, max_len=None):
        if text2 is not None: # 输入为 first, second形式
            return self._encode(text1, text2, max_len=max_len)
        else:
            if isinstance(text1, str):
                return self._encode(text1, max_len=max_len)
            token_ids, segment_ids = [], []
            for text in text1: # 输入为[(first1, second1), (first2, second2)]形式
                if isinstance(text, (list, tuple)):
                    if len(text) == 2:
                        first, second = text[0], text[1]
                    else:
                        first, second = text[0], None
                else:
                    first, second = text, None
                token_id, segment_id = self._encode(first, second, max_len=max_len)
                token_ids.append(token_id)
                segment_ids.append(segment_id)
            return token_ids, segment_ids

    def decode(self, ids):
        sep = ids.index(self._token_dict[self._token_sep])
        try:
            stop = ids.index(self._token_dict.get(self.TOKEN_PAD, 0))
        except ValueError as e:
            stop = len(ids)
        tokens = [self._token_dict_inv[i] for i in ids]
        first = tokens[1:sep]
        if sep < stop - 1:
            second = tokens[sep + 1:stop - 1]
            return first, second
        return first

    def _tokenize(self, text):
        if self._do_lower_case:
            text = unicodedata.normalize('NFD', text)
            text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
            text = text.lower()
        spaced = ''
        for ch in text:
            if self._is_punctuation(ch) or self._is_cjk_character(ch):
                spaced += ' ' + ch + ' '
            elif self._is_space(ch):
                spaced += ' '
            elif ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch
        tokens = []
        for word in spaced.strip().split():
            tokens += self._word_piece_tokenize(word)
        return tokens

    def _word_piece_tokenize(self, word):
        """基于word-piece的最大正向匹配法
        """
        if word in self._token_dict:
            return [word]
        tokens = []
        start = 0
        while start < len(word):
            end = len(word)
            while start < end:
                sub = word[start:end]
                if start > 0:
                    sub = '##' + sub
                if sub in self._token_dict:
                    break
                end -= 1
            if start == end:
                end += 1
            tokens.append(sub)
            start = end
        return tokens

    @staticmethod
    def _is_punctuation(ch):
        code = ord(ch)
        return 33 <= code <= 47 or \
               58 <= code <= 64 or \
               91 <= code <= 96 or \
               123 <= code <= 126 or \
               unicodedata.category(ch).startswith('P')

    @staticmethod
    def _is_cjk_character(ch):
        code = ord(ch)
        return 0x4E00 <= code <= 0x9FFF or \
               0x3400 <= code <= 0x4DBF or \
               0x20000 <= code <= 0x2A6DF or \
               0x2A700 <= code <= 0x2B73F or \
               0x2B740 <= code <= 0x2B81F or \
               0x2B820 <= code <= 0x2CEAF or \
               0xF900 <= code <= 0xFAFF or \
               0x2F800 <= code <= 0x2FA1F

    @staticmethod
    def _is_space(ch):
        return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or \
               unicodedata.category(ch) == 'Zs'

    @staticmethod
    def _is_control(ch):
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def rematch(text, tokens, do_lower_case=True, unknown_token=TOKEN_UNK):
        """找到token在原文中的下标。
        返回每个token的下标元组。
        """
        decoded, token_offsets = '', []
        for token in tokens:
            token_offsets.append([len(decoded), 0])
            if token == unknown_token:
                token = '#'
            if do_lower_case:
                token = token.lower()
            if len(token) > 2 and token.startswith('##'):
                token = token[2:]
            elif len(decoded) > 0:
                token = ' ' + token
                token_offsets[-1][0] += 1
            decoded += token
            token_offsets[-1][1] = len(decoded)

        heading = 0
        text = text.rstrip()
        for i in range(len(text)):
            if not Tokenizer._is_space(text[i]):
                break
            heading += 1
        text = text[heading:]
        len_text, len_decode = len(text), len(decoded)
        costs = [[0] * (len_decode + 1) for _ in range(2)]
        paths = [[(-1, -1)] * (len_decode + 1) for _ in range(len_text + 1)]
        curr, prev = 0, 1

        for j in range(len_decode + 1):
            costs[curr][j] = j
        for i in range(1, len_text + 1):
            curr, prev = prev, curr
            costs[curr][0] = i
            ch = text[i - 1]
            if do_lower_case:
                ch = ch.lower()
            for j in range(1, len_decode + 1):
                costs[curr][j] = costs[prev][j - 1]
                paths[i][j] = (i - 1, j - 1)
                if ch != decoded[j - 1]:
                    costs[curr][j] = costs[prev][j - 1]
                    paths[i][j] = (i - 1, j - 1)
                    if costs[prev][j] < costs[curr][j]:
                        costs[curr][j] = costs[prev][j]
                        paths[i][j] = (i - 1, j)
                    if costs[curr][j - 1] < costs[curr][j]:
                        costs[curr][j] = costs[curr][j - 1]
                        paths[i][j] = (i, j - 1)
                    costs[curr][j] += 1

        matches = [0] * (len_decode + 1)
        position = (len_text, len_decode)
        while position != (-1, -1):
            i, j = position
            matches[j] = i
            position = paths[i][j]

        intervals = [[matches[offset[0]], matches[offset[1]]] for offset in token_offsets]
        for i, interval in enumerate(intervals):
            token_a, token_b = text[interval[0]:interval[1]], tokens[i]
            if len(token_b) > 2 and token_b.startswith('##'):
                token_b = token_b[2:]
            if do_lower_case:
                token_a, token_b = token_a.lower(), token_b.lower()
            if token_a == token_b:
                continue
            if i == 0:
                border = 0
            else:
                border = intervals[i - 1][1]
            for j in range(interval[0] - 1, border - 1, -1):
                if Tokenizer._is_space(text[j]):
                    break
                interval[0] -= 1
            if i + 1 == len(intervals):
                border = len_text
            else:
                border = intervals[i + 1][0]
            for j in range(interval[1], border):
                if Tokenizer._is_space(text[j]):
                    break
                interval[1] += 1
        intervals = [(interval[0] + heading, interval[1] + heading) for interval in intervals]
        return intervals
