from transformers.tokenizer import Tokenizer
from transformers.models import build_bert_model
import numpy as np

# 模型文件
config_path = '/kg/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/kg/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/kg/bert/chinese_L-12_H-768_A-12/vocab.txt'

# 构建MLM模型
bert = build_bert_model(config_path, checkpoint_path, with_mlm=True)

# tokenize
tokenizer = Tokenizer(dict_path, do_lower_case=True)
text = '实验（英语：experiment）是在设定的条件下，用来检验某种假设，或者验证或质疑某种已经存在的理论而进行的操作'
tokens = tokenizer.tokenize(text)

# 对 “实验” 进行mask,并使用MLM预测
tokens[1] = '[MASK]'
tokens[2] = '[MASK]'
token_ids = tokenizer.convert_tokens_to_ids(tokens)
token_ids = np.array([token_ids])
segment_ids = np.zeros_like(token_ids)

# 输出
output = bert.predict([token_ids, segment_ids])[0][1:3]
idxs = output.argmax(axis=-1)
pred_tokens = tokenizer.convert_ids_to_tokens(idxs)
print(pred_tokens)
"""
pred_tokens = '实验'
"""

# 构建NSP模型
bert = build_bert_model(config_path, checkpoint_path, with_nsp=True)

text1 = '实践是检验真理的唯一标准。'
text2 = '人类登上月球，柏林墙倒下，世界因我们的想象与科学联系在一起。'

token_ids, segment_ids = tokenizer.encode(text1, text2)

token_ids = np.array([token_ids])
segment_ids = np.array([segment_ids])

is_next = bool(bert.predict([token_ids, segment_ids])[0].argmax(axis=-1))
print('is_next = ', is_next)
"""
is_next = False
"""