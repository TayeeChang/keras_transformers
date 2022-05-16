from transformers.layers import *
import json


_SHARED_BLOCK = {}


class MultiHeadSelfAttention(MultiHeadSelfAttention):
    """实现T5相对位置编码的多头自注意力机制
    """
    def call(self, inputs, mask=None):

        qw = self.q_dense(inputs[0])
        kw = self.k_dense(inputs[1])
        vw = self.v_dense(inputs[2])

        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.head_num, self.query_size))
        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.head_num, self.query_size))
        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.head_num, self.key_size))

        a = tf.einsum('bmhd, bnhd->bhmn', qw, kw)
        # T5相对位置编码
        a = a + inputs[3]
        a = a / self.query_size ** 0.5

        lm_bias=inputs[-1]
        if lm_bias:
            # attention偏置，用于防止看见未来信息
            a = a + globals()['attention_bias']
        a = mask_sequences(a, mask[1], axis=-1, value='-inf')
        # 将attention score归一化成概率分布
        a = K.softmax(a, axis=-1)
        # 这里的dropout参考自google transformer论文
        a = keras.layers.Dropout(self.attention_dropout_rate)(a)
        o = tf.einsum('bhmn, bnhd->bmhd', a, vw)
        o = K.reshape(o, (-1, K.shape(o)[1], self.head_num * self.key_size))
        o = self.o_dense(o)
        return o

    def compute_mask(self, inputs, mask=None):
        return mask[0]

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def _build_lm_bias(inputs):
    """用于mask未来信息
    返回 shape=(b, -1, n, n)
    """
    idxs = K.arange(0, K.shape(inputs)[1])
    idxs = idxs[:, None] <= idxs[:, :, None]
    mask = K.cast(idxs, K.floatx())
    return -(1 - mask[None, None]) * K.infinity()


def _wrap_self_layer(name,
                     input_layer,
                     build_func,
                     dropout_rate=0.0,
                     trainable=True,
                     lm_bias=None):
    """Wrap layers with dropout, residual, normalization.
    T5 with Pre-Norm.
    """
    input_layer, relative_pos_bias = input_layer
    normal_layer = LayerNormalization(
        center=False,
        trainable=trainable,
        name='%s-Norm' % name,
    )(input_layer)

    build_output = build_func([normal_layer, normal_layer, normal_layer, relative_pos_bias, lm_bias])

    if 0.0 < dropout_rate < 1.0:
        dropout_layer = keras.layers.Dropout(
            rate=dropout_rate,
            name='%s-Dropout' % name,
        )(build_output)
    else:
        dropout_layer = build_output

    add_layer = keras.layers.Add(name='%s-Add' % name)([input_layer, dropout_layer])

    return add_layer


def _wrap_cross_layer(name,
                      input_layer,
                      build_func,
                      dropout_rate=0.0,
                      trainable=True):
    relative_bias = input_layer[1]
    input_layer, encoder_output = input_layer[0]
    normal_layer = LayerNormalization(
        center=False,
        trainable=trainable,
        name='%s-Norm' % name,
    )(input_layer)
    build_output = build_func([normal_layer, encoder_output, encoder_output, relative_bias, None])

    if 0.0 < dropout_rate < 1.0:
        dropout_layer = keras.layers.Dropout(
            rate=dropout_rate,
            name='%s-Dropout' % name,
        )(build_output)
    else:
        dropout_layer = build_output

    add_layer = keras.layers.Add(name='%s-Add' % name)([input_layer, dropout_layer])
    return add_layer


def _wrap_ffn_layer(name,
                    input_layer,
                    build_func,
                    dropout_rate=0.0,
                    trainable=True):
    """Wrap layers with dropout, residual, normalization.
    T5 with Pre-Norm.
    """
    normal_layer = LayerNormalization(
        center=False,
        trainable=trainable,
        name='%s-Norm' % name,
    )(input_layer)
    build_output = build_func(normal_layer)
    if 0.0 < dropout_rate < 1.0:
        dropout_layer = keras.layers.Dropout(
            rate=dropout_rate,
            name='%s-Dropout' % name,
        )(build_output)
    else:
        dropout_layer = build_output
    add_layer = keras.layers.Add(name='%s-Add' % name)([input_layer, dropout_layer])

    return add_layer


def _build_shared_embeddings(vocab_size,
                             embedding_dim,
                             embedding_initializer,
                             name):
    return _SHARED_BLOCK.setdefault(
        name,
        TokenEmbedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            embeddings_initializer=embedding_initializer,
            mask_zero=True,
            name='Embedding-Token'
            )
    )


def _build_shared_scale(name,
                        scale):
    return _SHARED_BLOCK.setdefault(
        name,
        Scale(
            scale=scale,
            name='Decoder-Output-Scale'
        )
    )


def _build_position_bias(inputs,
                         num_buckets,
                         n_heads,
                         bidirectional,
                         name):
    return _SHARED_BLOCK.setdefault(name, RelativePositionBias(
        birectional=bidirectional,
        num_buckets=num_buckets,
        max_distance=128,
        n_heads=n_heads,
        embedding_initializer=keras.initializers.TruncatedNormal(0, 0.02),
        name=name,
    ))(inputs)


def _wrap_final_layer(name,
                      input_layer,
                      dropout_rate,
                      scale=None,
                      trainable=True):
    """Wrap final Layer with Norm and Dropout.
    """

    norm_layer = LayerNormalization(
        center=False,
        trainable=trainable,
        name='%s-Norm' % name,
    )(input_layer)
    if 0.0 < dropout_rate < 1.0:
        dropout_layer = keras.layers.Dropout(
            rate=dropout_rate,
            name='%s-Dropout' % name,
        )(norm_layer)
    else:
        dropout_layer = norm_layer
    if scale:
        dropout_layer = _build_shared_scale(
            name='Scale',
            scale=scale,
        )(dropout_layer)
    return dropout_layer


def get_encoder_component(name,
                          input_layer,
                          head_num,
                          head_size,
                          hidden_dim,
                          feed_forward_dim,
                          feed_forward_activation=None,
                          kernel_initializer='uniform',
                          attention_dropout_rate=0.0,
                          hidden_dropout_rate=0.0,
                          trainable=True):
    attention_name = "%s-MultiHeadSelfAttention" % name
    feed_forward_name = '%s-FeedForward' % name
    attention_layer = _wrap_self_layer(
        name=attention_name,
        input_layer=[
            input_layer,
            _build_position_bias([input_layer, input_layer],
                                 num_buckets=32,
                                 n_heads=head_num,
                                 bidirectional=True,
                                 name='Encoder-Relative-Position-Embedding')
        ],
        build_func=MultiHeadSelfAttention(
            head_num=head_num,
            query_size=head_size,
            key_size=head_size,
            output_dim=hidden_dim,
            use_bias=False,
            attention_dropout_rate=attention_dropout_rate,
            kernel_initializer=kernel_initializer,
            trainable=trainable,
            name=attention_name,
        ),
        dropout_rate=hidden_dropout_rate,
        trainable=trainable,
    )
    feed_forward_layer = _wrap_ffn_layer(
        name=feed_forward_name,
        input_layer=attention_layer,
        build_func=FeedForward(
            units=feed_forward_dim,
            activation=feed_forward_activation,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            trainable=trainable,
            name=feed_forward_name,
        ),
        dropout_rate=hidden_dropout_rate,
        trainable=trainable
    )
    return feed_forward_layer


def get_decoder_component(name,
                          input_layer,
                          head_num,
                          head_size,
                          hidden_dim,
                          feed_forward_dim,
                          feed_forward_activation=None,
                          kernel_initializer='uniform',
                          attention_dropout_rate=0.0,
                          hidden_dropout_rate=0.0,
                          trainable=True):
    self_attention_name = "%s-MultiHeadSelfAttention" % name
    cross_attention_name = "%s-Cross-MultiHeadSelfAttention" % name
    feed_forward_name = '%s-FeedForward' % name
    attention_layer = _wrap_self_layer(
        name=self_attention_name,
        input_layer=[
            input_layer[0],
            _build_position_bias([input_layer[0], input_layer[0]],
                                 num_buckets=32,
                                 n_heads=head_num,
                                 bidirectional=True,
                                 name='Decoder-Relative-Position-Embedding')
        ],
        build_func=MultiHeadSelfAttention(
            head_num=head_num,
            query_size=head_size,
            key_size=head_size,
            output_dim=hidden_dim,
            use_bias=False,
            attention_dropout_rate=attention_dropout_rate,
            kernel_initializer=kernel_initializer,
            trainable=trainable,
            name=self_attention_name,
        ),
        dropout_rate=hidden_dropout_rate,
        trainable=trainable,
        lm_bias=True,
    )
    cross_layer = _wrap_cross_layer(
        name=cross_attention_name,
        input_layer=[
            [attention_layer, input_layer[1]],
            _build_position_bias([input_layer[0], input_layer[1]],
                                 num_buckets=32,
                                 n_heads=head_num,
                                 bidirectional=True,
                                 name='Decoder-Relative-Position-Embedding')
        ],
        build_func=MultiHeadSelfAttention(
            head_num=head_num,
            query_size=head_size,
            key_size=head_size,
            output_dim=hidden_dim,
            use_bias=False,
            attention_dropout_rate=attention_dropout_rate,
            kernel_initializer=kernel_initializer,
            trainable=trainable,
            name=cross_attention_name,
        ),
        dropout_rate=hidden_dropout_rate,
        trainable=trainable,
    )
    feed_forward_layer = _wrap_ffn_layer(
        name=feed_forward_name,
        input_layer=cross_layer,
        build_func=FeedForward(
            units=feed_forward_dim,
            activation=feed_forward_activation,
            use_bias=False,
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
                 head_size,
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
            name='Encoder-%d' % i,
            input_layer=last_layer,
            head_num=head_num,
            head_size=head_size,
            hidden_dim=hidden_dim,
            feed_forward_dim=feed_forward_dim,
            feed_forward_activation=feed_forward_activation,
            kernel_initializer=kernel_initializer,
            attention_dropout_rate=attention_dropout_rate,
            hidden_dropout_rate=hidden_dropout_rate,
            trainable=trainable
        )
    return last_layer


def get_decoders(decoder_num,
                 input_layer,
                 head_num,
                 head_size,
                 hidden_dim,
                 feed_forward_dim,
                 feed_forward_activation='gelu',
                 kernel_initializer='uniform',
                 attention_dropout_rate=0.0,
                 hidden_dropout_rate=0.0,
                 trainable=True):
    last_layer, encoder_output = input_layer
    for i in range(decoder_num):
        last_layer = get_decoder_component(
            name='Decoder-%d' % i,
            input_layer=[last_layer, encoder_output],
            head_num=head_num,
            head_size=head_size,
            hidden_dim=hidden_dim,
            feed_forward_dim=feed_forward_dim,
            feed_forward_activation=feed_forward_activation,
            kernel_initializer=kernel_initializer,
            attention_dropout_rate=attention_dropout_rate,
            hidden_dropout_rate=hidden_dropout_rate,
            trainable=trainable
        )
    return last_layer


def get_encoder_inputs(seq_len=None):
    input_token_ids_enc = keras.layers.Input(
        shape=(seq_len,),
        name='Encoder-Input-%s' % 'Token'
    )
    return input_token_ids_enc


def get_decoder_inputs(seq_len=None):
    input_token_ids_dec = keras.layers.Input(
        shape=(seq_len,),
        name='Decoder-Input-%s' % 'Token'
    )
    globals()['attention_bias'] = _build_lm_bias(input_token_ids_dec)
    return input_token_ids_dec


def get_embeddings(inputs,
                   vocab_size,
                   embedding_dim,
                   hidden_dim,
                   embedding_initializer,
                   embedding_dropout_rate,
                   name):
    input_token_ids = inputs
    embedding_token, token_embeddings = _build_shared_embeddings(
        vocab_size,
        embedding_dim,
        embedding_initializer,
        name='Embedding-Token'
    )(input_token_ids)
    embeddings = keras.layers.Dropout(rate=embedding_dropout_rate, name=name)(embedding_token)
    if embedding_dim != hidden_dim:
        embeddings = keras.layers.Dense(
            units=hidden_dim,
            kernel_initializer=embedding_initializer,
            name='Embedding-Map'
        )(embeddings)
    return embeddings, token_embeddings


def get_encoder_model(vocab_size,
                      seq_len,
                      embedding_dim,
                      hidden_dim,
                      transformer_num,
                      head_num,
                      head_size,
                      feed_forward_dim,
                      feed_forward_activation,
                      attention_dropout_rate,
                      hidden_dropout_rate,
                      t5_initializer,
                      **kwargs):
    input_token_ids_enc = get_encoder_inputs(seq_len)
    embeddings, token_embeddings = get_embeddings(
        inputs=input_token_ids_enc,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        embedding_initializer=t5_initializer,
        embedding_dropout_rate=hidden_dropout_rate,
        name='Encoder',
    )
    output = get_encoders(
        encoder_num=transformer_num,
        input_layer=embeddings,
        head_num=head_num,
        head_size=head_size,
        hidden_dim=hidden_dim,
        feed_forward_dim=feed_forward_dim,
        feed_forward_activation=feed_forward_activation,
        kernel_initializer=t5_initializer,
        attention_dropout_rate=attention_dropout_rate,
        hidden_dropout_rate=hidden_dropout_rate,
        **kwargs,
    )
    output = _wrap_final_layer(
        name='Encoder-Final-Layer',
        input_layer=output,
        dropout_rate=hidden_dropout_rate,
        scale=False,
    )
    return input_token_ids_enc, output


def get_decoder_model(encoder_output,
                      vocab_size,
                      seq_len,
                      embedding_dim,
                      hidden_dim,
                      transformer_num,
                      head_num,
                      head_size,
                      feed_forward_dim,
                      feed_forward_activation,
                      attention_dropout_rate,
                      hidden_dropout_rate,
                      t5_initializer,
                      with_lm=False,
                      T5_version='t5.1.0',
                      **kwargs):
    input_token_ids_dec = get_decoder_inputs(seq_len)
    embeddings, token_embeddings = get_embeddings(
        inputs=input_token_ids_dec,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        embedding_initializer=t5_initializer,
        embedding_dropout_rate=hidden_dropout_rate,
        name='Decoder'
    )
    decoder_output = get_decoders(
        decoder_num=transformer_num,
        input_layer=[embeddings, encoder_output],
        head_num=head_num,
        head_size=head_size,
        hidden_dim=hidden_dim,
        feed_forward_dim=feed_forward_dim,
        feed_forward_activation=feed_forward_activation,
        kernel_initializer=t5_initializer,
        attention_dropout_rate=attention_dropout_rate,
        hidden_dropout_rate=hidden_dropout_rate,
        **kwargs,
    )
    decoder_output = _wrap_final_layer(
        name='Decoder-Final-Layer',
        input_layer=decoder_output,
        dropout_rate=hidden_dropout_rate,
        scale=hidden_dim ** 0.5,
    )

    if with_lm:
        if embedding_dim != hidden_dim:
            decoder_output = keras.layers.Dense(
                units=embedding_dim,
                kernel_initializer=t5_initializer,
                name='Decoder-Output-Map'
            )(decoder_output)

        if T5_version == 't5.1.0':
            lm_pred = EmbeddingSimilarity(
                name='Decoder-LM-Prob',
                use_bias=False
            )([decoder_output, token_embeddings])
        else:
            lm_pred = keras.layers.Dense(
                units=vocab_size,
                use_bias=False,
                activation='softmax',
                kernel_initializer=t5_initializer,
                name='Decoder-Output-LM'
            )(decoder_output)
        return input_token_ids_dec, lm_pred

    return input_token_ids_dec, decoder_output


def get_model(vocab_size,
              seq_len,
              embedding_dim,
              hidden_dim,
              transformer_num,
              head_num,
              head_size,
              feed_forward_dim,
              feed_forward_activation,
              attention_dropout_rate,
              hidden_dropout_rate,
              t5_initializer,
              with_lm,
              T5_version='t5.1.0',
              **kwargs):
    input_token_ids_enc, encoder_output = get_encoder_model(
        vocab_size,
        seq_len,
        embedding_dim,
        hidden_dim,
        transformer_num,
        head_num,
        head_size,
        feed_forward_dim,
        feed_forward_activation,
        attention_dropout_rate,
        hidden_dropout_rate,
        t5_initializer,
        **kwargs,
    )
    input_token_ids_dec, decoder_output = get_decoder_model(
        encoder_output,
        vocab_size,
        seq_len,
        embedding_dim,
        hidden_dim,
        transformer_num,
        head_num,
        head_size,
        feed_forward_dim,
        feed_forward_activation,
        attention_dropout_rate,
        hidden_dropout_rate,
        t5_initializer,
        with_lm=with_lm,
        T5_version=T5_version,
        **kwargs,
    )

    return [input_token_ids_enc, input_token_ids_dec], decoder_output


def build_T5_model(config_file,
                   checkpoint_file,
                   trainable=True,
                   with_lm=True,
                   T5_version='t5.1.0',
                   **kwargs):
    """Build the model from config file.
    # Reference:
        [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer]
        (https://arxiv.org/abs/1910.10683)

    """
    with open(config_file, 'r') as reader:
        config = json.loads(reader.read())

    config['t5_initializer'] = keras.initializers.TruncatedNormal(0, 0.02)
    inputs, outputs = get_model(
        vocab_size=config['vocab_size'],
        seq_len=None,
        embedding_dim=config.get('embedding_size', config.get('hidden_size')),
        hidden_dim=config['hidden_size'],
        transformer_num=config['num_hidden_layers'],
        head_num=config['num_attention_heads'],
        head_size=config['attention_head_size'],
        feed_forward_dim=config['intermediate_size'],
        feed_forward_activation=config['hidden_act'],
        attention_dropout_rate=config.get('attention_probs_dropout_prob', 0.) ,
        hidden_dropout_rate=config['hidden_dropout_prob'],
        t5_initializer=config['t5_initializer'],
        with_lm=with_lm,
        trainable=trainable,
        T5_version=T5_version,
        **kwargs,
    )
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    load_model_weights_from_checkpoint(
        model,
        config,
        checkpoint_file,
        T5_version,
    )
    return model


def checkpoint_loader(checkpoint_file):
    def _loader(name):
        return tf.train.load_variable(checkpoint_file, name)
    return _loader


def load_model_weights_from_checkpoint(model,
                                       config,
                                       checkpoint_file,
                                       version='t5.1.1'):
    """Load trained official model from checkpoint.
    """
    loader = checkpoint_loader(checkpoint_file)

    model.get_layer(name='Embedding-Token').set_weights([
        loader('shared/embedding'),
    ])
    try:
        model.get_layer(name='Encoder-Relative-Position-Embedding').set_weights([
            loader('encoder/block_000/layer_000/SelfAttention/relative_attention_bias'),
        ])
    except ValueError as e:
        model.get_layer(name='Encoder-Relative-Position-Embedding').set_weights([
            loader('encoder/block_000/layer_000/SelfAttention/relative_attention_bias').transpose(),
        ])
    try:
        model.get_layer(name='Decoder-Relative-Position-Embedding').set_weights([
            loader('decoder/block_000/layer_000/SelfAttention/relative_attention_bias')
        ])
    except ValueError as e:
        model.get_layer(name='Decoder-Relative-Position-Embedding').set_weights([
            loader('decoder/block_000/layer_000/SelfAttention/relative_attention_bias').transpose(),
        ])

    if version == 'mt5.1.1':
        model.get_layer(name='Encoder-Final-Layer-Norm').set_weights([
            loader('encoder/rms_norm/scale')
        ])
        model.get_layer(name='Decoder-Final-Layer-Norm').set_weights([
            loader('decoder/rms_norm/scale')
        ])
    else:
        model.get_layer(name='Encoder-Final-Layer-Norm').set_weights([
            loader('encoder/final_layer_norm/scale')
        ])
        model.get_layer(name='Decoder-Final-Layer-Norm').set_weights([
            loader('decoder/final_layer_norm/scale')
        ])

    for i in range(config['num_hidden_layers']):
        try:
            model.get_layer(name='Encoder-%d-MultiHeadSelfAttention' % i)
        except ValueError as e:
            continue
        model.get_layer(name='Encoder-%d-MultiHeadSelfAttention' % i).set_weights([
            loader('encoder/block_%03d/layer_000/SelfAttention/q' % i),
            loader('encoder/block_%03d/layer_000/SelfAttention/k' % i),
            loader('encoder/block_%03d/layer_000/SelfAttention/v' % i),
            loader('encoder/block_%03d/layer_000/SelfAttention/o' % i),
        ])
        if version == 'mt5.1.1':
            model.get_layer(name='Encoder-%d-MultiHeadSelfAttention-Norm' % i).set_weights([
                loader('encoder/block_%03d/layer_000/rms_norm/scale' % i),
            ])
        else:
            model.get_layer(name='Encoder-%d-MultiHeadSelfAttention-Norm' % i).set_weights([
                loader('encoder/block_%03d/layer_000/layer_norm/scale' % i),
            ])
        if version.endswith('t5.1.1'):
            model.get_layer(name='Encoder-%d-FeedForward' % i).set_weights([
                loader('encoder/block_%03d/layer_001/DenseReluDense/wi_0/kernel' % i),
                loader('encoder/block_%03d/layer_001/DenseReluDense/wi_1/kernel' % i),
                loader('encoder/block_%03d/layer_001/DenseReluDense/wo/kernel' % i),
            ])
        else:
            model.get_layer(name='Encoder-%d-FeedForward' % i).set_weights([
                loader('encoder/block_%03d/layer_001/DenseReluDense/wi/kernel' % i),
                loader('encoder/block_%03d/layer_001/DenseReluDense/wo/kernel' % i),
            ])
        if version == 'mt5.1.1':
            model.get_layer(name='Encoder-%d-FeedForward-Norm' % i).set_weights([
                loader('encoder/block_%03d/layer_001/rms_norm/scale' % i),
            ])
        else:
            model.get_layer(name='Encoder-%d-FeedForward-Norm' % i).set_weights([
                loader('encoder/block_%03d/layer_001/layer_norm/scale' % i),
            ])
        model.get_layer(name='Decoder-%d-MultiHeadSelfAttention' % i).set_weights([
            loader('decoder/block_%03d/layer_000/SelfAttention/q' % i),
            loader('decoder/block_%03d/layer_000/SelfAttention/k' % i),
            loader('decoder/block_%03d/layer_000/SelfAttention/v' % i),
            loader('decoder/block_%03d/layer_000/SelfAttention/o' % i),
        ])
        if version == 'mt5.1.1':
            model.get_layer(name='Decoder-%d-MultiHeadSelfAttention-Norm' % i).set_weights([
                loader('decoder/block_%03d/layer_000/rms_norm/scale' % i),
            ])
        else:
            model.get_layer(name='Decoder-%d-MultiHeadSelfAttention-Norm' % i).set_weights([
                loader('decoder/block_%03d/layer_000/layer_norm/scale' % i),
            ])
        model.get_layer(name='Decoder-%d-Cross-MultiHeadSelfAttention' % i).set_weights([
            loader('decoder/block_%03d/layer_001/EncDecAttention/q' % i),
            loader('decoder/block_%03d/layer_001/EncDecAttention/k' % i),
            loader('decoder/block_%03d/layer_001/EncDecAttention/v' % i),
            loader('decoder/block_%03d/layer_001/EncDecAttention/o' % i),
        ])
        if version == 'mt5.1.1':
            model.get_layer(name='Decoder-%d-Cross-MultiHeadSelfAttention-Norm' % i).set_weights([
                loader('decoder/block_%03d/layer_001/rms_norm/scale' % i),
            ])
        else:
            model.get_layer(name='Decoder-%d-Cross-MultiHeadSelfAttention-Norm' % i).set_weights([
                loader('decoder/block_%03d/layer_001/layer_norm/scale' % i),
            ])
        if version.endswith('t5.1.1'):
            model.get_layer(name='Decoder-%d-FeedForward' % i).set_weights([
                loader('decoder/block_%03d/layer_002/DenseReluDense/wi_0/kernel' % i),
                loader('decoder/block_%03d/layer_002/DenseReluDense/wi_1/kernel' % i),
                loader('decoder/block_%03d/layer_002/DenseReluDense/wo/kernel' % i),
            ])
        else:
            model.get_layer(name='Decoder-%d-FeedForward' % i).set_weights([
                loader('decoder/block_%03d/layer_002/DenseReluDense/wi/kernel' % i),
                loader('decoder/block_%03d/layer_002/DenseReluDense/wo/kernel' % i),
            ])
        if version == 'mt5.1.1':
            model.get_layer(name='Decoder-%d-FeedForward-Norm' % i).set_weights([
                loader('decoder/block_%03d/layer_002/rms_norm/scale' % i),
            ])
        else:
            model.get_layer(name='Decoder-%d-FeedForward-Norm' % i).set_weights([
                loader('decoder/block_%03d/layer_002/layer_norm/scale' % i),
            ])

    if version.endswith('t5.1.1'):
        model.get_layer(name='Decoder-Output-LM').set_weights([
            loader('decoder/logits/kernel'),
        ])