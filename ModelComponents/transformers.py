# imports
import tensorflow as tf
import numpy as np

"""
This contains keras layers used to encode image features and create tensors
to be fed into the prediction head. It uses transformers from DETR models diagram,
with minor modification preventing nan values.

Main Layers:
ImageEncoderAttention()
DecoderAttention()
PanopticAttention()  # not tested
"""

# Note: keras MHA layer is not in the TF javascript API that we may want to use. 
# The version below uses compatible ops.
class MultiheadAttention(tf.keras.layers.Layer):
    
    def __init__(self, num_attention_heads, dim, name='MultiheadAttention', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_attention_heads = num_attention_heads
        self.dim = dim

    def get_config(self):
        config = super().get_config()
        config.update({'num_attention_heads': self.num_attention_heads,
                       'dim': self.dim})
        return config

    def build(self, input_shape):
        self.query_shape = input_shape[0]
        self.key_shape = input_shape[1]
        self.value_shape = input_shape[2]

        query_dim = self.query_shape[-1]
        proj_dim = self.num_attention_heads * self.dim
        self.Rescale =  tf.keras.layers.Rescaling(
            scale=1.0 / tf.math.sqrt(tf.cast(self.dim, tf.float32)), name='Rescale')

        self.QueryProjection = tf.keras.layers.Dense(proj_dim, 
                    kernel_initializer='glorot_normal', name='QueryProjection')
        self.KeyProjection = tf.keras.layers.Dense(proj_dim, 
                    kernel_initializer='glorot_normal', name='KeyProjection')
        self.ValueProjection = tf.keras.layers.Dense(proj_dim, 
                    kernel_initializer='glorot_normal', name='ValueProjection')
        self.OutputProjection = tf.keras.layers.Dense(query_dim, 
                    kernel_initializer='glorot_normal', name='OutputProjection')

        self.ReshapeQuery = tf.keras.layers.Reshape(
            [self.query_shape[1], self.num_attention_heads, self.dim], name='ReshapeQuery')
        self.ReshapeKey = tf.keras.layers.Reshape(
            [self.key_shape[1], self.num_attention_heads, self.dim], name='ReshapeKey')
        self.ReshapeValue = tf.keras.layers.Reshape(
            [self.value_shape[1], self.num_attention_heads, self.dim], name='ReshapeValue')
        self.ReshapePreOutput = tf.keras.layers.Reshape(
            [self.query_shape[1], proj_dim], name='ReshapePreOutput')
        
        self.PermuteQuery = tf.keras.layers.Permute([2,1,3], name='PermuteQuery')
        self.PermuteKey = tf.keras.layers.Permute([2,3,1], name='PermuteKey')
        self.PermuteValue = tf.keras.layers.Permute([2,1,3], name='PermuteValue')

        self.MatMulQueryKey = tf.keras.layers.Lambda(
            lambda x: tf.linalg.matmul(x[0], x[1]), name='MatMulQueryKey')
        self.MatMulQueryValue = tf.keras.layers.Lambda(
            lambda x: tf.linalg.matmul(x[0], x[1]), name='MatMulQueryValue')  # requires value_steps=key_steps

    def call(self, inputs, attention_mask=None, training=False):
        query, key, value = inputs
        
        # projections
        query = self.QueryProjection(query)  # [batch, query_steps, num_attention_heads * dim]
        key = self.KeyProjection(key)  # [batch, key_steps, num_attention_heads * dim]
        value = self.ValueProjection(value)  # [batch, value_steps, num_attention_heads * dim]

        # separate out the heads
        query = self.ReshapeQuery(query)  # [batch, query_steps, num_attention_heads, dim]
        key = self.ReshapeKey(key)  # [batch, key_steps, num_attention_heads, dim]
        value = self.ReshapeValue(value)  # [batch, value_steps, num_attention_heads, dim]

        # rearrange for aligned matrix multiplications
        query = self.PermuteQuery(query)   # [batch, num_attention_heads, dim, key_steps]
        key = self.PermuteKey(key)   # [batch, num_attention_heads, dim, key_steps]
        value = self.PermuteValue(value)   # [batch, num_attention_heads, value_steps, dim]

        # query-key interaction
        x = self.MatMulQueryKey([query, key])  # [batch, num_attention_heads, query_steps, key_steps]
        x = self.Rescale(x)
        x = tf.keras.activations.softmax(x, axis=-1)  # softmax along the key_steps axis

        # apply masking. mask should have 0's for masked positions, 1's otherwise
        if attention_mask is None:
            attention_mask = tf.ones_like(x)        
        x = x * attention_mask

        # query-value interaction
        x = self.MatMulQueryValue([x, value])  # [batch, query_steps, num_attention_heads, dim]

        # project back to original dim
        x = self.ReshapePreOutput(x)  # [batch, query_steps, num_attention_heads * dim]
        x = self.OutputProjection(x)  # [batch, query_steps, query_dim]
        return x

    def show_summary(self):
        query = tf.keras.layers.Input(self.query_shape[1:], name='query')
        key = tf.keras.layers.Input(self.key_shape[1:], name='key')
        value = tf.keras.layers.Input(self.value_shape[1:], name='value')
        inputs=[query, key, value]
        return tf.keras.Model(inputs=inputs, outputs=self.call(inputs)).summary()


class AttentionBlock(tf.keras.layers.Layer):

    def __init__(self, num_attention_heads, **kwargs):
        super().__init__(**kwargs)
        self.num_attention_heads = num_attention_heads

    def get_config(self):
        config = super().get_config()
        config.update({'num_attention_heads': self.num_attention_heads})
        return config

    def build(self, input_shape):
        self.query_shape = input_shape[0]    # used in self.show_summary()
        self.key_shape = input_shape[1]
        self.value_shape = input_shape[2]

        query_dim = self.query_shape[-1]
        key_dim = query_dim // self.num_attention_heads

        self.AttentionLayer = MultiheadAttention(num_attention_heads=self.num_attention_heads,
                                                 dim=key_dim,
                                                 name='AttentionLayer')

        self.Dropout = tf.keras.layers.Dropout(rate=.1, name='Dropout')
        self.Add = tf.keras.layers.Add(dtype=tf.float32, name='Add')
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=1e-3, name='LayerNorm')

    def call(self, inputs, attention_mask=None, training=False):

        query, key, value = inputs

        attention_features = self.AttentionLayer([query, key, value],
                                                 attention_mask=attention_mask,
                                                 training=training)

        attention_features = self.Dropout(attention_features, training=training)
        query = self.Add([query, attention_features])
        query = self.LayerNorm(query, training=training)

        return query

    def show_summary(self):
        query = tf.keras.layers.Input(self.query_shape[1:], name='query')
        key = tf.keras.layers.Input(self.key_shape[1:], name='key')
        value = tf.keras.layers.Input(self.value_shape[1:], name='value')
        inputs=[query, key, value]
        return tf.keras.Model(inputs=inputs, outputs=self.call(inputs)).summary()


class FeedForwardBlock(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_config(self):
        return super().get_config()

    def build(self, input_shape):
        self.features_shape = input_shape[0]
        features_dim = self.features_shape[-1]

        # feed forward
        self.DenseRelu = tf.keras.layers.Dense(features_dim, activation='relu', 
                            kernel_initializer='glorot_normal', name='DenseRelu')
        self.DenseLinear = tf.keras.layers.Dense(features_dim, activation=None, 
                            kernel_initializer='glorot_normal', name='DenseLinear')
        self.Add = tf.keras.layers.Add(dtype=tf.float32, name='Add')
        self.Dropout = tf.keras.layers.Dropout(rate=.1, name='Dropout')
        self.LayerNorm = tf.keras.layers.LayerNormalization(name='LayerNorm')

    def call(self, inputs, training=False):
        features = inputs[0]

        # Feed Forward block
        dense_features = self.DenseRelu(features)
        dense_features = self.DenseLinear(dense_features)

        dense_features = self.Dropout(dense_features, training=training)
        features = self.Add([features, dense_features])
        features = self.LayerNorm(features, training=training)

        return features

    def show_summary(self):
        features = tf.keras.layers.Input(self.features_shape[1:], name='features')
        inputs=[features]
        return tf.keras.Model(inputs=inputs, outputs=self.call(inputs)).summary()

class EncoderBlock(tf.keras.layers.Layer):

    def __init__(self, num_attention_heads, **kwargs):
        super().__init__(**kwargs)
        self.num_attention_heads = num_attention_heads

        # layers
        self.SelfAttentionBlock = AttentionBlock(num_attention_heads=num_attention_heads,
                                                 name='SelfAttentionBlock')
        self.FeedForwardBlock = FeedForwardBlock(name='FeedForwardBlock')
        self.Add1 = tf.keras.layers.Add(dtype=tf.float32, name='Add1')
        self.Add2 = tf.keras.layers.Add(dtype=tf.float32, name='Add2')

    def get_config(self):
        config = super().get_config()
        config.update({'num_attention_heads': self.num_attention_heads})
        return config

    def build(self, input_shape):
        self.encoder_features_shape = input_shape[0]
        self.encoder_positional_shape = input_shape[1]

    def call(self, inputs, training=False):
        encoder_features, encoder_positional = inputs

        # Self Attention
        query = self.Add1([encoder_features, encoder_positional])
        key = self.Add2([encoder_features, encoder_positional])
        value = encoder_features

        encoder_features = self.SelfAttentionBlock([query, key, value], training=training)

        # Feed Forward
        encoder_features = self.FeedForwardBlock([encoder_features], training=training)

        return encoder_features

    def show_summary(self):
        encoder_features = tf.keras.layers.Input(self.encoder_features_shape[1:], name='encoder_features')
        encoder_positional = tf.keras.layers.Input(self.encoder_positional_shape[1:], name='encoder_positional')
        inputs=[encoder_features, encoder_positional]
        return tf.keras.Model(inputs=inputs, outputs=self.call(inputs)).summary()


class ImageEncoderAttention(tf.keras.layers.Layer):

    def __init__(self, num_blocks, num_attention_heads, name='ImageEncoderAttention', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_blocks = num_blocks
        self.num_attention_heads = num_attention_heads

        # transformers
        self.EncoderBlocks = []
        for i in range(self.num_blocks):
            block = EncoderBlock(num_attention_heads=self.num_attention_heads, name=f'EncoderBlock_{i}')
            self.EncoderBlocks.append(block)

    def get_config(self):
        config = super().get_config()
        config.update({'num_blocks': self.num_blocks, 
                       'num_attention_heads': self.num_attention_heads})
        return config

    def build(self, input_shape):
        self.encoder_features_shape = input_shape[0]  # used in self.show_summary()

        num_encoder_row = self.encoder_features_shape[1]
        num_encoder_col = self.encoder_features_shape[2]
        encoder_dim = self.encoder_features_shape[3]

        # reshaping layers
        self.GetBatchDim = tf.keras.layers.Lambda(lambda x: tf.shape(x)[0], name='GetBatchDim')
        self.TileBatch3D = tf.keras.layers.Lambda(
                    lambda x: tf.tile(x[0], [x[1], 1, 1, 1]), name='TileBatch2D')
        self.Flatten2D = tf.keras.layers.Reshape(
            [num_encoder_row*num_encoder_col, encoder_dim], name='Flatten2D')
        self.Reshape3D_Encoder = tf.keras.layers.Reshape(
            [num_encoder_row, num_encoder_col, encoder_dim], name='Reshape3D_Encoder')
        self.Reshape3D_Positional = tf.keras.layers.Reshape(
            [num_encoder_row, num_encoder_col, encoder_dim], name='Reshape3D_Positional')

        # (Fixed) positional encoding variable on inputs (spacial component only)
        def trig(k, dim):
            denom = 2*(1 + dim) / encoder_dim
            even = k % 2
            odd = (k+1) % 2
            return even * np.math.sin(k / denom) + odd * np.math.cos(k / denom)

        init_value = tf.constant([[trig(k, dim)
                                   for dim in range(encoder_dim)]
                                   for k in range(num_encoder_col*num_encoder_row)])
        init_value = tf.reshape(init_value, [num_encoder_row, num_encoder_col, encoder_dim])
        self.positional_encoding = tf.Variable(init_value, trainable=True, name='positional_encoding')

    def call(self, inputs, training=False):
        encoder_features = inputs[0]  # [batch, rows, columns, features_dim]
        batch_size = self.GetBatchDim(encoder_features)

        # add batch to positional
        positional_encoding = tf.expand_dims(self.positional_encoding, 0)
        positional_encoding = self.TileBatch3D([positional_encoding, batch_size])

        # update encoder
        encoder_features = self.Flatten2D(encoder_features)  # [batch, rows*columns, features_dim]
        positional_encoding = self.Flatten2D(positional_encoding)

        # apply transformer blocks
        for i in range(self.num_blocks):
            encoder_features = self.EncoderBlocks[i]([encoder_features, positional_encoding],
                                                      training=training)

        # return encoder and positional to original shapes
        encoder_features = self.Reshape3D_Encoder(encoder_features)
        positional_encoding = self.Reshape3D_Positional(positional_encoding)

        return encoder_features, positional_encoding

    def show_summary(self):
        encoder_features = tf.keras.layers.Input(shape=self.encoder_features_shape[1:], name='encoder_features')
        inputs = [encoder_features]
        outputs = self.call(inputs)  # note: use self.call() for all layers to show in summary
        return tf.keras.Model(inputs, outputs, name=self.name).summary()


class DecoderBlock_NoSelfAttention(tf.keras.layers.Layer):

    def __init__(self, num_attention_heads, **kwargs):
        super().__init__(**kwargs)
        self.num_attention_heads = num_attention_heads

        # layers
        self.JointAttentionBlock = AttentionBlock(num_attention_heads=num_attention_heads,
                                                  name='JointAttentionBlock')
        self.FeedForwardBlock = FeedForwardBlock(name='FeedForwardBlock')

    def get_config(self):
        config = super().get_config()
        config.update({'num_attention_heads': self.num_attention_heads})
        return config

    def call(self, inputs, training=False):
        encoder_value, decoder_features, encoder_key, decoder_positional = inputs

        # Joint Attention
        query = decoder_features
        key = encoder_key
        value = encoder_value

        decoder_features = self.JointAttentionBlock([query, key, value], training=training)

        # Feed Forward
        decoder_features = self.FeedForwardBlock([decoder_features], training=training)

        return decoder_features


class DecoderBlock(tf.keras.layers.Layer):

    def __init__(self, num_attention_heads, **kwargs):
        super().__init__(**kwargs)
        self.num_attention_heads = num_attention_heads

        # layers
        self.SelfAttentionBlock = AttentionBlock(num_attention_heads=num_attention_heads,
                                                 name='SelfAttentionBlock')
        self.JointAttentionBlock = AttentionBlock(num_attention_heads=num_attention_heads,
                                                  name='JointAttentionBlock')
        self.FeedForwardBlock = FeedForwardBlock(name='FeedForwardBlock')

    def get_config(self):
        config = super().get_config()
        config.update({'num_attention_heads': self.num_attention_heads})
        return config

    def call(self, inputs, training=False):
        encoder_value, decoder_features, encoder_key, decoder_positional = inputs

        # Self Attention
        query = decoder_features  # decoder_features + decoder_positional is commented out. This is what seems to be used in the paper's diagram, but is producing NaN problems. Instead the standard AIAYN version is used
        key = decoder_features  # decoder_features + decoder_positional
        value = decoder_features

        decoder_features = self.SelfAttentionBlock([query, key, value], training=training)

        # Joint Attention
        query = decoder_features
        key =  encoder_key
        value = encoder_value

        decoder_features = self.JointAttentionBlock([query, key, value], training=training)

        # Feed Forward
        decoder_features = self.FeedForwardBlock([decoder_features], training=training)

        return decoder_features


class DecoderPrep(tf.keras.layers.Layer):
    def __init__(self, num_object_preds, decoder_dim, name='DecoderPrep', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_object_preds = num_object_preds
        self.decoder_dim = decoder_dim

    def get_config(self):
        config = super().get_config()
        config.update({'num_object_preds': self.num_object_preds,
                       'decoder_dim': self.decoder_dim})
        return config

    def build(self, input_shape):
        self.encoder_features_shape = input_shape[0]  # used in self.show_summary()
        self.encoder_positional_encoding_shape = input_shape[1]  # used in self.show_summary()
        num_rows = self.encoder_features_shape[1]
        num_cols = self.encoder_features_shape[2]
        encoder_dim = self.encoder_features_shape[3]

        # Reshaping
        self.GetBatchDim = tf.keras.layers.Lambda(lambda x: tf.shape(x)[0], name='GetBatchDim')
        self.TileBatch2D = tf.keras.layers.Lambda(lambda x: tf.tile(x[0], [x[1], 1, 1]), name='TileBatch')
        self.Flatten2D_Image = tf.keras.layers.Reshape
                ([num_rows*num_cols, encoder_dim], name='Flatten2D_Image')
        self.Flatten2D_Positional = tf.keras.layers.Reshape(
                [num_rows*num_cols, encoder_dim], name='Flatten2D_Positional')

        # encoder key adjustment
        self.Add = tf.keras.layers.Add(dtype=tf.float32, name='Add')

        # initial decoder input (treated as positional encoding). Make sure trainable=True!
        initializer = tf.zeros_initializer()
        init_decoder_features = initializer([self.num_object_preds, self.decoder_dim], tf.float32)
        self.init_decoder_features = tf.Variable(init_decoder_features, 
                                                 trainable=True, name='init_decoder_features')

    def call(self, inputs, training=False):
        encoder_features, encoder_positional = inputs

        # update encoder shape
        encoder_value = self.Flatten2D_Image(encoder_features)
        encoder_key = self.Flatten2D_Positional(encoder_positional)

        # update encoder's key
        encoder_key = self.Add([encoder_value, encoder_key])  # note of interest: this is a difference between DETR model diagram and standard AIAYN

        # update decoder shape
        batch_size = self.GetBatchDim(encoder_value)
        init_decoder_features = tf.expand_dims(self.init_decoder_features, axis=0)  # batch dim

        decoder_features = self.TileBatch2D([init_decoder_features, batch_size])  # [batch, num_preds, decoder_dim1]
        decoder_positional = decoder_features

        return encoder_value, decoder_features, encoder_key, decoder_positional

    def show_summary(self):
        encoder_features = tf.keras.layers.Input(shape=self.encoder_features_shape[1:], name='encoder_features')
        positional_encoding = tf.keras.layers.Input(shape=self.encoder_positional_encoding_shape[1:], name='positional_encoding')
        inputs = [encoder_features, positional_encoding]
        return tf.keras.Model(inputs, outputs=self(inputs)).summary()



class PanopticAttention(tf.keras.layers.Layer):
    """
    Partial Multihead Attention Layer. (Skips concat / projection back to original dims,
    and instead outputs an image encoding for each decoder box)
    """

    def __init__(self, num_attention_heads, hidden_dim, name='PanopticAttention', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_attention_heads = num_attention_heads
        self.hidden_dim = hidden_dim

    def get_config(self):
        config = super().get_config()
        config.update({'num_attention_heads': self.num_attention_heads, 'hidden_dim': self.hidden_dim})
        return config

    def build(self, input_shape):
        self.image_encoding_shape = input_shape[0]  # [batch, rows, cols, encoder_dim]
        self.decoder_encoding_shape = input_shape[1]  # [batch, num_obj, decoder_dim]
        self.positional_encoding_shape = input_shape[2]  # [batch, rows, cols, 1]

        # dims
        encoder_dim = self.image_encoding_shape[3]
        num_rows = self.image_encoding_shape[1]
        num_cols = self.image_encoding_shape[2]
        num_obj = self.decoder_encoding_shape[1]

        # attention dims
        key_dim = tf.math.maximum(1, self.hidden_dim // self.num_attention_heads)  # standard
        self.scale_factor = tf.math.sqrt(tf.cast(key_dim, tf.float32))
        value_dim = num_obj  # non-standard

        # attention projections
        self.ValueProjection = tf.keras.layers.Dense(self.num_attention_heads*value_dim, 
                                                     name='ValueProjection')
        self.KeyProjection = tf.keras.layers.Dense(self.num_attention_heads*key_dim, 
                                                   name='KeyProjection')
        self.QueryProjection = tf.keras.layers.Dense(self.num_attention_heads*key_dim, 
                                                     name='QueryProjection')

        # attention calculations
        self.MatMul_1 = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1]), name='MatMul_1')
        self.MatMul_2 = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1]), name='MatMul_2')
        self.Transpose = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0,2,1]), name='Transpose')

        # activation layers
        self.Softmax = tf.keras.layers.Softmax(name='Softmax')

        # reshaping layers
        self.Flatten2D_Image = tf.keras.layers.Reshape([num_rows*num_cols, encoder_dim], 
                                                       name='Flatten2D_Image')
        self.Flatten2D_Positional = tf.keras.layers.Reshape([num_rows*num_cols, encoder_dim], 
                                                            name='Flatten2D_Positional')
        self.ReshapeOutput = tf.keras.layers.Reshape([num_rows, num_cols, num_obj, -1], 
                                                     name='ReshapeOutput')

        # other layers
        self.Add = tf.keras.layers.Add(name='Add')
        self.LayerNorm = tf.keras.layers.LayerNormalization(name='LayerNorm')
        self.Divide = tf.keras.layers.Lambda(lambda x: x[0] / x[1], name='Divide')

    def call(self, inputs, training=False, self_attention_mask=None):
        image_encoding, decoder_encoding, positional_encoding = inputs

        # prepare shapes
        image_encoding = self.Flatten2D_Image(image_encoding)
        positional_encoding = self.Flatten2D_Positional(positional_encoding)

        value = image_encoding  # [batch, rows*cols, encoder_dim]
        key = self.Add([image_encoding, positional_encoding])  # [batch, rows*cols, encoder_dim]
        query = decoder_encoding  # [batch, num_obj, decoder_dim]

        # Partial Multihead Attention
        # projections
        value_heads = self.ValueProjection(value)  # [batch, rows*cols, num_attention_heads*num_obj]
        key_heads = self.KeyProjection(value)  # [batch, rows*cols, num_attention_heads*key_dim]
        query_heads = self.QueryProjection(value)  # [batch, num_obj, num_attention_heads*key_dim]

        # multiplications
        key_heads = self.Transpose(key_heads)  # [batch, num_attention_heads*key_dim, rows*cols]
        multi_head = self.MatMul_1([query_heads, key_heads])   # [batch, rows*cols, rows*cols]
        multi_head = self.Divide([multi_head, self.scale_factor])
        multi_head = self.Softmax(multi_head)

        multi_head = self.MatMul_2([multi_head, value_heads])  # [batch, rows*cols, num_attention_heads*num_obj]
        multi_head = self.LayerNorm(multi_head, training=training)

        # reshape in standard conv format
        attention_maps = self.ReshapeOutput(multi_head)  # [batch, rows, cols, num_attention_heads*num_obj]

        return attention_maps

    def show_summary(self):
        image_encoding = tf.keras.layers.Input(shape=self.image_encoding_shape[1:], name='image_encoding')
        decoder_encoding = tf.keras.layers.Input(shape=self.decoder_encoding_shape[1:], name='decoder_encoding')
        positional_encoding = tf.keras.layers.Input(shape=self.positional_encoding_shape[1:], name='positional_encoding')

        inputs = [image_encoding, decoder_encoding, positional_encoding]
        outputs = self.call(inputs)  # note: self.call() allows all layers to show in summary
        return tf.keras.Model(inputs, outputs, name=self.name).summary()
