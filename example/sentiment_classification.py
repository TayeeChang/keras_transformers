import codecs
from keras_transformers.backend import keras
from keras_transformers.models import build_bert_model
from keras_transformers.utils import DataGenerator, pad_sequences
from keras_transformers.tokenizer import Tokenizer
from keras.optimizers import Adam
from keras_transformers.optimizers import (
    wrap_optimizer_with_warmup,
    wrap_optimizer_with_accumulate_grads,
    wrap_optimizer_with_weight_decay
)

# 模型文件
config_path = '/kg/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/kg/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/kg/bert/chinese_L-12_H-768_A-12/vocab.txt'

# 自定义超参数
num_labels = 2
epochs = 10
batch_size = 32
maxlen = 128

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)  


def load_data(file_path):
    """读取数据
    """
    data = []
    with codecs.open(file_path, 'r', 'utf-8') as infile:
        for line in infile:
            text, label = line.strip().split('\t')
            data.append((text, int(label)))
    return data


class DataLoader(DataGenerator):
    """数据加载器，生成训练批数据
    """
    def __iter__(self):
        for data in self.batch_generator:
            text, labels = zip(*data)
            batch_token_ids, batch_segment_ids = tokenizer.encode(text, max_len=maxlen)
            batch_labels = [[label] for label in labels]
            batch_token_ids = pad_sequences(batch_token_ids, axis=-1)
            batch_segment_ids = pad_sequences(batch_segment_ids, axis=-1)
            batch_labels = pad_sequences(batch_labels, axis=-1)
            yield [batch_token_ids, batch_segment_ids], batch_labels


# 构建数据集
train_generator = DataLoader(load_data('data/sentiment/sentiment.train.data'), batch_size=batch_size)
valid_generator = DataLoader(load_data('data/sentiment/sentiment.valid.data'), batch_size=batch_size)
test_generator = DataLoader(load_data('data/sentiment/sentiment.test.data'), batch_size=batch_size)

# 构建模型
bert = build_bert_model(config_path, checkpoint_path)
output = keras.layers.Lambda(
    lambda x: x[:, 0], name='Extract-CLS'
)(bert.output)
output = keras.layers.Dense(
    units=num_labels,
    activation='softmax',
    name='Softmax',
)(output)
model = keras.models.Model(bert.input, output)
model.summary()

# 定义优化器
adamWD = wrap_optimizer_with_weight_decay(Adam)
adamWU = wrap_optimizer_with_warmup(adamWD)
adamAcc = wrap_optimizer_with_accumulate_grads(adamWU)

# 模型编译
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=adamAcc(
        learning_rate=3e-5,
        exclude_weights=['Norm', 'bias'],
        warmup_steps=50,
        total_steps=len(train_generator) * epochs,
        acc_grad_steps=2,
    ),
    metrics=['accuracy'],
)


def evaluate(data):
    """评估函数
    """
    total, golden = 0., 0.
    for x, y in data:
        y_pred = model.predict(x).argmax(axis=-1)
        y_true = y[:, 0]
        total += len(y_true)
        golden += sum(y_true == y_pred)
    return golden / total


class Evaluator(keras.callbacks.Callback):
    """回调函数，用于模型评估和保存.
    """
    def __init__(self):
        self.best_val_acc = 0.
        super(Evaluator, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model.weights')
        test_acc = evaluate(test_generator)
        print(
            u'val_acc=%.5f, best_val_acc=%.5f, test_acc=%.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )


if __name__ == "__main__":
    model.fit_generator(
        train_generator.fit_generator(random=True),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[Evaluator()],
    )
