from keras2bert.layers import *
import numpy as np
import json

_SHARED_BLOCK = {}


def _wrap_layer(name,
                input_layer,
                build_func,
                dropout_rate=0.0,
                trainable=True):
    """Wrap layers with dropout, residual, normalization.
    """
    build_output = build_func(input_layer)
    if 0.0 < dropout_rate < 1.0:
        dropout_layer = _build_shared_dropout(
            rate=dropout_rate,
            name='%s-Dropout' % name,
        )(build_output)
    else:
        dropout_layer = build_output
    if isinstance(input_layer, list):
        input_layer = input_layer[0]
    add_layer = _build_shared_add(name='%s-Add' % name)([input_layer, dropout_layer])
    normal_layer = _build_shared_norm(
        trainable=trainable,
        name='%s-Norm' % name,
    )(add_layer)
    return normal_layer


def _wrap_embedding(name,
                    input_layer,
                    build_func,
                    dropout_rate,
                    trainable=True):
    """Wrap Embedding Layer with Norm and Dropout.
    """
    build_output = build_func(input_layer)
    norm_layer = LayerNormalization(
        trainable=trainable,
        name='%s-Norm' % name,
    )(build_output)
    if 0.0 < dropout_rate < 1.0:
        dropout_layer = keras.layers.Dropout(
            rate=dropout_rate,
            name='%s-Dropout' % name,
        )(norm_layer)
    else:
        dropout_layer = norm_layer
    return dropout_layer


def _build_shared_multi_head_self_attention(head_num,
                                            query_size,
                                            key_size,
                                            output_dim,
                                            attention_dropout_rate,
                                            kernel_initializer,
                                            trainable,
                                            name):
    return _SHARED_BLOCK.setdefault(name, MultiHeadSelfAttention(
            head_num,
            query_size,
            key_size,
            output_dim,
            attention_dropout_rate,
            kernel_initializer,
            trainable=trainable,
            name=name
    ))


def _build_shared_feed_forward(units,
                               activation,
                               kernel_initializer,
                               trainable,
                               name):
    return _SHARED_BLOCK.setdefault(name, FeedForward(
        units=units,
        activation=activation,
        kernel_initializer=kernel_initializer,
        trainable=trainable,
        name=name,
    ))


def _build_shared_dropout(rate, name):
    return _SHARED_BLOCK.setdefault(name, keras.layers.Dropout(
        rate,
        name=name
    ))


def _build_shared_add(name):
    return _SHARED_BLOCK.setdefault(name, keras.layers.Add(name=name))


def _build_shared_norm(trainable, name):
    return _SHARED_BLOCK.setdefault(name, LayerNormalization(
        trainable,
        name=name
    ))


def get_encoder_component(input_layer,
                          head_num,
                          hidden_dim,
                          feed_forward_dim,
                          feed_forward_activation=None,
                          kernel_initializer='uniform',
                          attention_dropout_rate=0.0,
                          hidden_dropout_rate=0.0,
                          trainable=True,
                          name=None):
    attention_name = "Encoder-MultiHeadSelfAttention"
    feed_forward_name = 'Encoder-FeedForward'

    attention_layer = _wrap_layer(
        name=attention_name,
        input_layer=input_layer,
        build_func=_build_shared_multi_head_self_attention(
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
        build_func=_build_shared_feed_forward(
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
            name='Encoder-%d' % i,
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
              bert_initializer,
              with_nsp=False,
              with_mlm=False,
              **kwargs):
    input_token_ids, input_segment_ids = get_inputs(seq_len)
    embeddings, token_embeddings = get_embeddings(
        inputs=[input_token_ids, input_segment_ids],
        vocab_size=vocab_size,
        segment_type_size=segment_type_size,
        max_pos_num=max_pos_num,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        embedding_initializer=bert_initializer,
        embedding_dropout_rate=hidden_dropout_rate,
    )
    output = get_encoders(
        encoder_num=transformer_num,
        input_layer=embeddings,
        head_num=head_num,
        hidden_dim=hidden_dim,
        feed_forward_dim=feed_forward_dim,
        feed_forward_activation=feed_forward_activation,
        kernel_initializer=bert_initializer,
        attention_dropout_rate=attention_dropout_rate,
        hidden_dropout_rate=hidden_dropout_rate,
        **kwargs,
    )

    nsp_pred, mlm_pred = None, None
    if with_nsp:
        cls_output = Lambda(
            name='Extract-CLS',
            function=lambda x: x[:, 0]
        )(output)
        nsp_dense = keras.layers.Dense(
            units=hidden_dim,
            activation='tanh',
            name='NSP-Dense',
        )(cls_output)
        nsp_pred = keras.layers.Dense(
            units=2,
            activation='softmax',
            name='NSP-Prob'
        )(nsp_dense)

    if with_mlm:
        mlm_dense = keras.layers.Dense(
            units=hidden_dim,
            activation=feed_forward_activation,
            name='MLM-Dense',
        )(output)
        mlm_norm = LayerNormalization(name='MLM-Norm')(mlm_dense)
        mlm_pred = EmbeddingSimilarity(name='MLM-Prob')([mlm_norm, token_embeddings])

    if with_nsp and with_mlm:
        output = [nsp_pred, mlm_pred]
    elif with_nsp:
        output = nsp_pred
    elif with_mlm:
        output = mlm_pred

    return [input_token_ids, input_segment_ids], output


def build_albert_model(config_file,
                       checkpoint_file,
                       trainable=True,
                       seq_len=int(1e9),
                       with_nsp=False,
                       with_mlm=False,
                       **kwargs):
    """Build the model from config file.
    # Reference:
        [ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS]
        (https://arxiv.org/pdf/1909.11942.pdf?ref=https://githubhelp.com)
    """
    with open(config_file, 'r') as reader:
        config = json.loads(reader.read())
    if seq_len is not None:
        config['max_position_embeddings'] = min(seq_len, config['max_position_embeddings'])

    config['bert_initializer'] = keras.initializers.TruncatedNormal(0, 0.02)
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
        bert_initializer=config['bert_initializer'],
        with_nsp=with_nsp,
        with_mlm=with_mlm,
        trainable=trainable,
        **kwargs,
    )
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    load_model_weights_from_checkpoint(
        model,
        config,
        checkpoint_file,
        with_mlm=with_mlm,
        with_nsp=with_nsp,
    )
    return model


def checkpoint_loader(checkpoint_file):
    def _loader(name):
        return tf.train.load_variable(checkpoint_file, name)
    return _loader


def load_model_weights_from_checkpoint(model,
                                       config,
                                       checkpoint_file,
                                       with_nsp=False,
                                       with_mlm=False):
    """Load trained official model from checkpoint.
    """
    loader = checkpoint_loader(checkpoint_file)

    model.get_layer(name='Embedding-Token').set_weights([
        loader('bert/embeddings/word_embeddings'),
    ])
    model.get_layer(name='Embedding-Segment').set_weights([
        loader('bert/embeddings/token_type_embeddings'),
    ])
    model.get_layer(name='Embedding-Position').set_weights([
        loader('bert/embeddings/position_embeddings')[:config['max_position_embeddings'], :],
    ])
    model.get_layer(name='Embedding-Norm').set_weights([
        loader('bert/embeddings/LayerNorm/gamma'),
        loader('bert/embeddings/LayerNorm/beta'),
    ])

    model.get_layer(name='Embedding-Map').set_weights([
        loader('bert/encoder/embedding_hidden_mapping_in/kernel'),
        loader('bert/encoder/embedding_hidden_mapping_in/bias'),
    ])

    model.get_layer(name='Encoder-MultiHeadSelfAttention').set_weights([
        loader('bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel'),
        loader('bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias'),
        loader('bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel'),
        loader('bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias'),
        loader('bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel'),
        loader('bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias'),
        loader('bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel'),
        loader('bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias'),
    ])
    model.get_layer(name='Encoder-MultiHeadSelfAttention-Norm').set_weights([
        loader('bert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma'),
        loader('bert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta'),
    ])
    model.get_layer(name='Encoder-FeedForward').set_weights([
        loader('bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel'),
        loader('bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias'),
        loader('bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel'),
        loader('bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/bias'),
    ])
    model.get_layer(name='Encoder-FeedForward-Norm').set_weights([
        loader('bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma'),
        loader('bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta'),
    ])

    if with_mlm:
        model.get_layer(name='MLM-Dense').set_weights([
            loader('cls/predictions/transform/dense/kernel'),
            loader('cls/predictions/transform/dense/bias'),
        ])
        model.get_layer(name='MLM-Norm').set_weights([
            loader('cls/predictions/transform/LayerNorm/gamma'),
            loader('cls/predictions/transform/LayerNorm/beta'),
        ])
        model.get_layer(name='MLM-Prob').set_weights([
            loader('cls/predictions/output_bias'),
        ])
    if with_nsp:
        model.get_layer(name='NSP-Dense').set_weights([
            loader('bert/pooler/dense/kernel'),
            loader('bert/pooler/dense/bias'),
        ])
        model.get_layer(name='NSP-Prob').set_weights([
            np.transpose(loader('cls/seq_relationship/output_weights')),
            loader('cls/seq_relationship/output_bias'),
        ])