# Keras Transformer Family

![Authour](https://img.shields.io/badge/Author-Tayee%20Chang-blue.svg) 
![Python](https://img.shields.io/badge/Python-3.6+-brightgreen.svg)
![Tensorflow](https://img.shields.io/badge/TensorFlow-1.13.0-yellowgreen.svg)
![NLP](https://img.shields.io/badge/NLP-Transformers-redgreen.svg)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://github.com/TayeeChang/transformers/blob/master/LICENSE)


🚀🚀🚀  Transformer家族模型Keras版实现，可加载官方预训练权重来支持下游任务。

目前支持的Transformer模型：   
- BERT——[下载](https://github.com/google-research/bert)  
- Roberta——[下载|brightmart版](https://github.com/brightmart/roberta_zh) [下载|哈工大版](https://github.com/ymcui/Chinese-BERT-wwm)
- Albert——[下载](https://github.com/brightmart/albert_zh)  
- Nezha——[下载](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-TensorFlow)   
- Unilm——[下载](https://github.com/google-research/bert)  
- Electra——[下载|google版](https://github.com/google-research/electra) [下载|哈工大版](https://github.com/ymcui/Chinese-ELECTRA)
- GPT——[下载](https://github.com/bojone/CDial-GPT-tf)
- GPT2——[下载](https://github.com/imcaspar/gpt2-ml)
- T5.1.1——[下载](https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md#t511)  
- T5——[下载](https://github.com/google-research/text-to-text-transfer-transformer)

后续会陆续添加其他Transformer模型。

继续完善中...

欢迎使用

## 说明

   环境使用  
   - keras >= 2.3.1  
   - tensorflow == 1.13 or 1.14 (建议)
   
## 安装
```shell   
pip install git+https://github.com/TayeeChang/transformers.git
```
或者
```shell
python setup.py install
```

## 使用
 
 具体参考 [example](https://github.com/TayeeChang/transformers/tree/master/example)
 
## 引用
1. <a href="https://arxiv.org/pdf/1810.04805.pdf&usg=ALkJrhhzxlCL6yTht2BRmH9atgvKFxHsxQ">BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding</a>
2. <a href="https://arxiv.org/pdf/1907.11692.pdf%5C">RoBERTa: A Robustly Optimized BERT Pretraining Approach</a>
3. <a href="https://arxiv.org/pdf/1909.11942.pdf?ref=https://githubhelp.com">ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS</a>
4. <a href="https://arxiv.org/pdf/1909.00204.pdf">NEZHA: NEURAL CONTEXTUALIZED REPRESENTATION FOR CHINESE LANGUAGE UNDERSTANDING</a>
5. <a href="https://arxiv.org/abs/1905.03197">Unified Language Model Pre-training for Natural Language Understanding and Generation</a>
6. <a href="https://arxiv.org/abs/2003.10555">ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators</a>
7. <a href="https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf">Improving Language Understanding by Generative Pre-Training</a>
8. <a href="http://www.persagen.com/files/misc/radford2019language.pdf">Language Models are Unsupervised Multitask Learners</a>
9. <a href="https://arxiv.org/abs/1910.10683">Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer</a>
10. <a href="https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf">Attention Is All You Need</a>
11. <a href="https://github.com/CyberZHG/keras-bert">https://github.com/CyberZHG/keras-bert</a>
