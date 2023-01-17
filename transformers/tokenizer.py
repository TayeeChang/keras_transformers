import unicodedata
import codecs
import json
import regex as re
from functools import lru_cache


def load_vocab(vocab_path):
    """加载词典
    """
    token_dict = {}
    with codecs.open(vocab_path, 'r', 'utf-8') as infile:
        for line in infile:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict


@lru_cache()
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word for BPE.
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


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
        self._token_cls_id = self._token_dict[self._token_cls]
        self._token_sep_id = self._token_dict[self._token_sep]
        self._token_unk_id = self._token_dict[self._token_unk]
        self._token_pad_id = self._token_dict[self._token_pad]
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

    def tokenize(self, first, second=None, max_len=None):
        """实现分词功能
        """
        first_tokens = self._tokenize(first)
        second_tokens = self._tokenize(second) if second is not None else None
        self._truncate(first_tokens, second_tokens, max_len)
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
            for text in text1:
                if isinstance(text, (list, tuple)):
                    if len(text) == 2: # 输入为[(first1, second1), (first2, second2), ...]形式
                        first, second = text[0], text[1]
                    else: # 输入为[(first1,), (first2, ), ...]形式
                        first, second = text[0], None
                else: # 输入为[first1, first2, ...]形式
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
        """核心分词函数
        """
        if self._do_lower_case:
            text = text.lower()
            text = unicodedata.normalize('NFD', text)
            text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])

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
                return [word]
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

    def _is_special(self, ch):
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    def _stem(self, ch):
        return ch[2:] if len(ch) > 2 and ch.startswith('##') else ch

    def rematch(self, text, tokens):
        """建立token和text间的映射关系
        """
        cleaned_text = ''
        char_indexs = []
        for i, ch in enumerate(text):
            if self._do_lower_case:
                ch = ch.lower()
                ch = unicodedata.normalize('NFD', ch)
                ch = ''.join([s for s in ch if unicodedata.category(s) != 'Mn'])
            ch = ''.join([s for s in ch if not (ord(s) == 0 or ord(s) == 0xfffd or Tokenizer._is_control(s))])
            cleaned_text += ch
            char_indexs.extend([i] * len(ch))

        offsets_mapping = []
        offsets = 0
        text = cleaned_text
        for token in tokens:
            if self._is_special(token):
                offsets_mapping.append(())
                continue
            token = self._stem(token)
            start = text[offsets:].index(token) + offsets
            end = start + len(token)
            offsets_mapping.append((char_indexs[start], char_indexs[end-1]))
            offsets = end
        return offsets_mapping


class BytePairEncoding(object):
    def __init__(self, encoder, bpe_merges, errors='replace'):
        self.encoder = encoder
        self.decoder = {v:k for k,v in self.encoder.items()}
        self.errors = errors # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word


class RobertaTokenizer(Tokenizer, BytePairEncoding):
    TOKEN_BOS = "<s>"
    TOKEN_EOS = "</s>"
    TOKEN_UNK = "<unk>"
    TOKEN_PAD = '<pad>'

    def __init__(self,
                 roberta_vocab_file,
                 gpt_vocab_file,
                 gpt_merge_file,
                 token_bos=TOKEN_BOS,
                 token_eos=TOKEN_EOS,
                 token_unk=TOKEN_UNK,
                 token_pad=TOKEN_PAD,
                 do_lower_case=False):

        self._token_dict = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
        self._token_bos = self._token_cls = token_bos
        self._token_eos = self._token_sep = token_eos
        self._token_unk = token_unk
        self._token_pad = token_pad
        self._token_bos_id = self._token_cls_id = self._token_dict[self._token_bos]
        self._token_eos_id = self._token_sep_id = self._token_dict[self._token_eos]
        self._token_unk_id = self._token_dict[self._token_unk]
        self._token_pad_id = self._token_dict[self._token_pad]
        self._do_lower_case = do_lower_case

        with open(gpt_vocab_file, 'r') as f:
            gpt_vocab = json.load(f)
        with open(gpt_merge_file, 'r', encoding="utf-8") as f:
            gpt_bpe_data = f.read()
        bpe_merges = [tuple(merge_str.split()) for merge_str in gpt_bpe_data.split('\n')[1:-1]]
        self.oldid2newid = {}
        with open(roberta_vocab_file, 'r') as f:
            for i, line in enumerate(f):
                new_id = i + 4
                old_id, count = line.strip().split()
                if old_id.isnumeric():
                    self.oldid2newid[int(old_id)] = new_id

        for word in gpt_vocab:
            self._token_dict[word] = self.oldid2newid[gpt_vocab[word]]
        self._token_dict_inv = {v: k for k, v in self._token_dict.items()}

        BytePairEncoding.__init__(self, gpt_vocab, bpe_merges)

    def _tokenize(self, text):
        """使用BPE分词
        """
        if self._do_lower_case:
            text = text.lower()
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def _is_special(self, ch):
        return bool(ch) and (ch[0] == '<') and (ch[-1] == '>')

    def _stem(self, ch):
        i = 0
        for i in range(len(ch)):
            if ch[i] != ' ':
                break
        return ch[i:]

    def decode(self, token_ids):
        if token_ids[0] == self._token_cls_id:
            token_ids = token_ids[1:]
        sentences = []
        sent = []
        for tid in token_ids:
            if tid != self._token_eos_id:
                sent.append(tid)
            else:
                sentences.append(sent)
                sent = []
        text = []
        for sentence in sentences:
            sentence = ''.join(self.convert_ids_to_tokens(sentence))
            sentence = bytearray([self.byte_decoder[c] for c in sentence]).decode('utf-8', errors=self.errors)
            text.append(sentence)
        return text

    def rematch(self, text, tokens):
        """建立token和text间的映射关系
        """
        cleaned_text = ''
        char_indexs = []
        for i, ch in enumerate(text):
            if self._do_lower_case:
                ch = ch.lower()
                ch = unicodedata.normalize('NFD', ch)
                ch = ''.join([s for s in ch if unicodedata.category(s) != 'Mn'])
            ch = ''.join([s for s in ch if not (ord(s) == 0 or ord(s) == 0xfffd or Tokenizer._is_control(s))])
            cleaned_text += ch
            char_indexs.append(i)

        offsets_mapping = []
        offsets = 0
        text = cleaned_text

        tokens = [
            bytearray([self.byte_decoder[c] for c in token]).decode('utf-8', errors=self.errors)
            for token in tokens
        ]
        for token in tokens:
            if self._is_special(token):
                offsets_mapping.append(())
                continue
            token = self._stem(token)
            if token not in text[offsets:]:  # 考虑特殊类Mn字符如"\u0300"
                start = offsets
                end = start + 1
            else:
                start = text[offsets:].index(token) + offsets
                end = start + len(token)
                offsets = end
            offsets_mapping.append((char_indexs[start], char_indexs[end-1]))

        return offsets_mapping
