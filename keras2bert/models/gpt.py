from keras2bert.layers import *
import numpy as np
import json


class MultiHeadSelfAttention(MultiHeadSelfAttention):
    """带Attention bias
    """
    def call(self, inputs, mask=None):
        qw = self.q_dense(inputs)
        kw = self.k_dense(inputs)
        vw = self.v_dense(inputs)

        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.head_num, self.query_size))
        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.head_num, self.query_size))
        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.head_num, self.key_size))

        a = tf.einsum('bmhd, bnhd->bhmn', qw, kw)
        a = a / self.query_size ** 0.5
        # attention偏置，用于防止看见未来信息
        a = a + globals()['attention_bias']
        a = mask_sequences(a, mask, axis=-1, value='-inf')
        # 将attention score归一化成概率分布
        a = K.softmax(a, axis=-1)
        # 这里的dropout参考自google transformer论文
        a = keras.layers.Dropout(self.attention_dropout_rate)(a)
        o = tf.einsum('bhmn, bnhd->bmhd', a, vw)

        o = K.reshape(o, (-1, K.shape(o)[1], self.head_num * self.key_size))
        o = self.o_dense(o)

        return o


def _build_lm_bias(inputs):
    """用于mask未来信息
    返回 shape=(b, -1, n, n)
    """
    idxs = K.arange(0, K.shape(inputs)[1])
    idxs = idxs[:, None] <= idxs[:, :, None]
    mask = K.cast(idxs, K.floatx())
    return -(1 - mask[None, None]) * K.infinity()


def _wrap_layer(name,
                input_layer,
                build_func,
                dropout_rate=0.0,
                trainable=True):
    """Wrap layers with dropout, residual, normalization.
    """
    build_output = build_func(input_layer)
    if 0.0 < dropout_rate < 1.0:
        dropout_layer = keras.layers.Dropout(
            rate=dropout_rate,
            name='%s-Dropout' % name,
        )(build_output)
    else:
        dropout_layer = build_output
    if isinstance(input_layer, list):
        input_layer = input_layer[0]
    add_layer = keras.layers.Add(name='%s-Add' % name)([input_layer, dropout_layer])
    normal_layer = LayerNormalization(
        trainable=trainable,
        name='%s-Norm' % name,
    )(add_layer)
    return normal_layer


def _wrap_embedding(name,
                    input_layer,
                    build_func,
                    dropout_rate,
                    trainable=True):
    """Wrap Embedding Layer with Dropout （GPT without Norm).
    """
    build_output = build_func(input_layer)
    if 0.0 < dropout_rate < 1.0:
        dropout_layer = keras.layers.Dropout(
            rate=dropout_rate,
            name='%s-Dropout' % name,
        )(build_output)
    else:
        dropout_layer = build_output
    return dropout_layer


def get_encoder_component(name,
                          input_layer,
                          head_num,
                          hidden_dim,
                          feed_forward_dim,
                          feed_forward_activation=None,
                          kernel_initializer='uniform',
                          attention_dropout_rate=0.0,
                          hidden_dropout_rate=0.0,
                          trainable=True):
    attention_name = "%s-MultiHeadSelfAttention" % name
    feed_forward_name = '%s-FeedForward' % name
    attention_layer = _wrap_layer(
        name=attention_name,
        input_layer=input_layer,
        build_func=MultiHeadSelfAttention(
            head_num=head_num,
            query_size=hidden_dim // head_num,
            key_size=hidden_dim // head_num,
            output_dim=hidden_dim,
            attention_dropout_rate=attention_dropout_rate,
            kernel_initializer=kernel_initializer,
            trainable=trainable,
            name=attention_name,
        ),
        dropout_rate=hidden_dropout_rate,
        trainable=trainable,
    )
    feed_forward_layer = _wrap_layer(
        name=feed_forward_name,
        input_layer=attention_layer,
        build_func=FeedForward(
            units=feed_forward_dim,
            activation=feed_forward_activation,
            kernel_initializer=kernel_initializer,
            trainable=trainable,
            name=feed_forward_name,
        ),
        dropout_rate=hidden_dropout_rate,
        trainable=trainable
    )
    return feed_forward_layer


def get_encoders(encoder_num,
                 input_layer,
                 head_num,
                 hidden_dim,
                 feed_forward_dim,
                 feed_forward_activation='gelu',
                 kernel_initializer='uniform',
                 attention_dropout_rate=0.0,
                 hidden_dropout_rate=0.0,
                 trainable=True):
    last_layer = input_layer
    for i in range(encoder_num):
        last_layer = get_encoder_component(
            name='Transformer-%d' % i,
            input_layer=last_layer,
            head_num=head_num,
            hidden_dim=hidden_dim,
            feed_forward_dim=feed_forward_dim,
            feed_forward_activation=feed_forward_activation,
            kernel_initializer=kernel_initializer,
            attention_dropout_rate=attention_dropout_rate,
            hidden_dropout_rate=hidden_dropout_rate,
            trainable=trainable
        )
    return last_layer


def get_inputs(seq_len=None):
    input_token_ids = keras.layers.Input(
        shape=(seq_len,),
        name='Input-%s' % 'Token'
    )
    input_segment_ids = keras.layers.Input(
        shape=(seq_len,),
        name='Input-%s' % 'Segment'
    )
    globals()['attention_bias'] = _build_lm_bias(input_token_ids)
    return input_token_ids, input_segment_ids


def get_embeddings(inputs,
                   vocab_size,
                   segment_type_size,
                   embedding_dim,
                   hidden_dim,
                   embedding_initializer,
                   max_pos_num,
                   embedding_dropout_rate):
    input_token_ids, input_segment_ids = inputs
    embedding_token, token_embeddings = TokenEmbedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        embeddings_initializer=embedding_initializer,
        mask_zero=True,
        name='Embedding-Token'
    )(input_token_ids)
    embedding_segment = Embedding(
        input_dim=segment_type_size,
        output_dim=embedding_dim,
        embeddings_initializer=embedding_initializer,
        name='Embedding-Segment'
    )(input_segment_ids)
    embeddings = keras.layers.Add(
        name='Embedding-Add-Token-Segment'
    )([embedding_token, embedding_segment])
    embeddings = _wrap_embedding(
        name='Embedding',
        input_layer=embeddings,
        build_func=PositionEmbedding(
            input_dim=max_pos_num,
            output_dim=embedding_dim,
            mode='add',
            embedding_initializer=embedding_initializer,
            name='Embedding-Position'
        ),
        dropout_rate=embedding_dropout_rate,
    )
    if embedding_dim != hidden_dim:
        embeddings = keras.layers.Dense(
            units=hidden_dim,
            kernel_initializer=embedding_initializer,
            name='Embedding-Map'
        )(embeddings)
    return embeddings, token_embeddings


def get_model(vocab_size,
              segment_type_size,
              max_pos_num,
              seq_len,
              embedding_dim,
              hidden_dim,
              transformer_num,
              head_num,
              feed_forward_dim,
              feed_forward_activation,
              attention_dropout_rate,
              hidden_dropout_rate,
              gpt_initializer,
              **kwargs):
    input_token_ids, input_segment_ids = get_inputs(seq_len)
    embeddings, token_embeddings = get_embeddings(
        inputs=[input_token_ids, input_segment_ids],
        vocab_size=vocab_size,
        segment_type_size=segment_type_size,
        max_pos_num=max_pos_num,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        embedding_initializer=gpt_initializer,
        embedding_dropout_rate=hidden_dropout_rate,
    )
    output = get_encoders(
        encoder_num=transformer_num,
        input_layer=embeddings,
        head_num=head_num,
        hidden_dim=hidden_dim,
        feed_forward_dim=feed_forward_dim,
        feed_forward_activation=feed_forward_activation,
        kernel_initializer=gpt_initializer,
        attention_dropout_rate=attention_dropout_rate,
        hidden_dropout_rate=hidden_dropout_rate,
        **kwargs,
    )
    output = EmbeddingSimilarity(
        use_bias=False,
        name='Embedding-Sim'
    )(output)

    return [input_token_ids, input_segment_ids], output


def build_gpt_model(config_file,
                    checkpoint_file,
                    trainable=True,
                    seq_len=int(1e9),
                    **kwargs):
    """Build the model from config file.
    # Reference:
        [Improving language understanding by generative pre-training]
        (https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf)

    """
    with open(config_file, 'r') as reader:
        config = json.loads(reader.read())

    if seq_len is not None:
        config['max_position_embeddings'] = min(seq_len, config['max_position_embeddings'])

    config['gpt_initializer'] = keras.initializers.TruncatedNormal(0, 0.02)
    inputs, outputs = get_model(
        vocab_size=config['vocab_size'],
        segment_type_size=config['type_vocab_size'],
        max_pos_num=config['max_position_embeddings'],
        seq_len=None,
        embedding_dim=config.get('embedding_size', config.get('hidden_size')),
        hidden_dim=config['hidden_size'],
        transformer_num=config['num_hidden_layers'],
        head_num=config['num_attention_heads'],
        feed_forward_dim=config['intermediate_size'],
        feed_forward_activation=config['hidden_act'],
        attention_dropout_rate=config['attention_probs_dropout_prob'],
        hidden_dropout_rate=config['hidden_dropout_prob'],
        gpt_initializer=config['gpt_initializer'],
        trainable=trainable,
        **kwargs,
    )
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    load_model_weights_from_checkpoint(
        model,
        config,
        checkpoint_file
    )
    return model


def checkpoint_loader(checkpoint_file):
    def _loader(name):
        return tf.train.load_variable(checkpoint_file, name)
    return _loader


def load_model_weights_from_checkpoint(model,
                                       config,
                                       checkpoint_file):
    """Load trained official model from checkpoint.
    """
    loader = checkpoint_loader(checkpoint_file)

    model.get_layer(name='Embedding-Token').set_weights([
        loader('gpt/embeddings/word_embeddings'),
    ])
    model.get_layer(name='Embedding-Segment').set_weights([
        loader('gpt/embeddings/token_type_embeddings'),
    ])
    model.get_layer(name='Embedding-Position').set_weights([
        loader('gpt/embeddings/position_embeddings')[:config['max_position_embeddings'], :],
    ])
    model.get_layer(name='Embedding-Norm').set_weights([
        loader('gpt/embeddings/LayerNorm/gamma'),
        loader('gpt/embeddings/LayerNorm/beta'),
    ])
    for i in range(config['num_hidden_layers']):
        try:
            model.get_layer(name='Transformer-%d-MultiHeadSelfAttention' % i)
        except ValueError as e:
            continue
        model.get_layer(name='Transformer-%d-MultiHeadSelfAttention' % i).set_weights([
            loader('gpt/transformer/layer_%d/attention/self/query/kernel' % i),
            loader('gpt/transformer/layer_%d/attention/self/query/bias' % i),
            loader('gpt/transformer/layer_%d/attention/self/key/kernel' % i),
            loader('gpt/transformer/layer_%d/attention/self/key/bias' % i),
            loader('gpt/transformer/layer_%d/attention/self/value/kernel' % i),
            loader('gpt/transformer/layer_%d/attention/self/value/bias' % i),
            loader('gpt/transformer/layer_%d/attention/output/dense/kernel' % i),
            loader('gpt/transformer/layer_%d/attention/output/dense/bias' % i),
        ])
        model.get_layer(name='Transformer-%d-MultiHeadSelfAttention-Norm' % i).set_weights([
            loader('gpt/transformer/layer_%d/attention/output/LayerNorm/gamma' % i),
            loader('gpt/transformer/layer_%d/attention/output/LayerNorm/beta' % i),
        ])
        model.get_layer(name='Transformer-%d-FeedForward' % i).set_weights([
            loader('gpt/transformer/layer_%d/intermediate/dense/kernel' % i),
            loader('gpt/transformer/layer_%d/intermediate/dense/bias' % i),
            loader('gpt/transformer/layer_%d/output/dense/kernel' % i),
            loader('gpt/transformer/layer_%d/output/dense/bias' % i),
        ])
        model.get_layer(name='Transformer-%d-FeedForward-Norm' % i).set_weights([
            loader('gpt/transformer/layer_%d/output/LayerNorm/gamma' % i),
            loader('gpt/transformer/layer_%d/output/LayerNorm/beta' % i),
        ])

