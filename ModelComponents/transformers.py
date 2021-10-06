# imports
import tensorflow as tf
import numpy as np

"""
This contains keras layers used to encode image features and create tensors
to be fed into the prediction head. It uses transformers similar to those in the 
"Attention is All You Need" and DETR models.

Main Layers:
ImageEncoderAttention()
DecoderAttention()
"""

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
        key_dim = tf.math.maximum(1, query_dim//self.num_attention_heads)

        # Joint Attention Layers
        self.JointAttentionLayer = tf.keras.layers.MultiHeadAttention(
                                        num_heads=self.num_attention_heads, 
                                        key_dim=key_dim,
                                        dropout=0.1,
                                        name='JointAttention')
        self.JointAdd = tf.keras.layers.Add(name='JointAdd')
        self.JointLayerNorm = tf.keras.layers.LayerNormalization(name='JointLayerNorm')
        self.Add = tf.keras.layers.Add(name='Add')

    def call(self, inputs, training=False, attention_mask=None):  
        query, key, value = inputs
       
        # Joint Attention Block 
        attention_features = self.JointAttentionLayer(query=query, 
                                                      value=value,
                                                      key=key,
                                                      attention_mask=attention_mask,
                                                      training=training)      
        
        query = self.JointAdd([query, attention_features], training=training)
        query = self.JointLayerNorm(query, training=training)

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
        self.DenseRelu = tf.keras.layers.Dense(features_dim, activation='relu', name='DenseRelu')
        self.DenseLinear = tf.keras.layers.Dense(features_dim, activation=None, name='DenseLinear')
        self.DenseAdd = tf.keras.layers.Add(name='DenseAdd')
        self.DenseLayerNorm = tf.keras.layers.LayerNormalization(name='DenseLayerNorm')
        self.Dropout = tf.keras.layers.Dropout(rate=.01, name='Dropout')
              
    def call(self, inputs, training=False):
        features = inputs[0]       

        # Feed Forward block
        dense_features = self.DenseRelu(features, training=training)
        dense_features = self.DenseLinear(dense_features, training=training)

        dense_features = self.Dropout(dense_features, training=training)
        features = self.DenseAdd([features, dense_features], training=training)
        features = self.DenseLayerNorm(features, training=training)

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
        self.Add = tf.keras.layers.Add(name='Add')

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
        query = self.Add([encoder_features, encoder_positional]) 
        key = self.Add([encoder_features, encoder_positional]) 
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
        config.update({'num_blocks': self.num_blocks, 'num_attention_heads': self.num_attention_heads})
        return config

    def build(self, input_shape):
        self.encoder_features_shape = input_shape[0]  # used in self.show_summary()

        num_encoder_row = self.encoder_features_shape[1]
        num_encoder_col = self.encoder_features_shape[2]
        encoder_dim = self.encoder_features_shape[3]

        # reshaping layers
        self.GetBatchDim = tf.keras.layers.Lambda(lambda x: tf.shape(x)[0], name='GetBatchDim')
        self.TileBatch3D = tf.keras.layers.Lambda(lambda x: tf.tile(x[0], [x[1], 1, 1, 1]), name='TileBatch2D')
        self.Flatten2D = tf.keras.layers.Reshape([num_encoder_row*num_encoder_col, encoder_dim], name='Flatten2D')
        self.Reshape3D_Encoder = tf.keras.layers.Reshape([num_encoder_row, num_encoder_col, encoder_dim], name='Reshape3D_Encoder')
        self.Reshape3D_Positional = tf.keras.layers.Reshape([num_encoder_row, num_encoder_col, encoder_dim], name='Reshape3D_Positional')

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
        self.positional_encoding = tf.Variable(init_value, trainable=False, name='positional_encoding')
        
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
        self.Add = tf.keras.layers.Add(name='Add')

    def get_config(self):
        config = super().get_config()
        config.update({'num_attention_heads': self.num_attention_heads})
        return config
              
    def call(self, inputs, training=False):
        encoder_features, decoder_features, encoder_positional, decoder_positional = inputs

        # Joint Attention
        query = self.Add([decoder_features, decoder_positional]) 
        key = self.Add([encoder_features, encoder_positional]) 
        value = encoder_features
        
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
        self.Add = tf.keras.layers.Add(name='Add')

    def get_config(self):
        config = super().get_config()
        config.update({'num_attention_heads': self.num_attention_heads})
        return config
              
    def call(self, inputs, training=False):
        encoder_features, decoder_features, encoder_positional, decoder_positional = inputs

        # Self Attention
        query = self.Add([decoder_features, decoder_positional]) 
        key = self.Add([decoder_features, decoder_positional]) 
        value = decoder_features

        decoder_features = self.SelfAttentionBlock([query, key, value], training=training)
        
        # Joint Attention
        query = decoder_features
        key = encoder_features
        value = encoder_features
        
        # add positionals and perform joint attention
        query = self.Add([query, decoder_positional]) 
        key = self.Add([key, encoder_positional]) 
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
        self.Flatten2D_Image = tf.keras.layers.Reshape([num_rows*num_cols, encoder_dim], name='Flatten2D_Image')
        self.Flatten2D_Positional = tf.keras.layers.Reshape([num_rows*num_cols, encoder_dim], name='Flatten2D_Positional')

        # initial decoder input (treated as positional encoding)
        initializer = tf.random_normal_initializer()
        init_decoder_features = initializer([self.num_object_preds, self.decoder_dim], tf.float32)
        self.init_decoder_features = tf.Variable(init_decoder_features, trainable=True, name='init_decoder_features')

    def call(self, inputs, training=False):
        encoder_features, encoder_positional = inputs

        # update encoder shape
        encoder_features = self.Flatten2D_Image(encoder_features)
        encoder_positional = self.Flatten2D_Positional(encoder_positional)

        # update decoder shape
        batch_size = self.GetBatchDim(encoder_features)
        init_decoder_features = tf.expand_dims(self.init_decoder_features, axis=0)  # batch dim

        decoder_features = self.TileBatch2D([init_decoder_features, batch_size])  # [batch, num_preds, decoder_dim1]
        decoder_positional = decoder_features

        return encoder_features, decoder_features, encoder_positional, decoder_positional
        
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
        self.ValueProjection = tf.keras.layers.Dense(self.num_attention_heads*value_dim, name='ValueProjection')
        self.KeyProjection = tf.keras.layers.Dense(self.num_attention_heads*key_dim, name='KeyProjection')
        self.QueryProjection = tf.keras.layers.Dense(self.num_attention_heads*key_dim, name='QueryProjection')

        # attention calculations
        self.MatMul_1 = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1]), name='MatMul_1')
        self.MatMul_2 = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1]), name='MatMul_2')
        self.Transpose = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0,2,1]), name='Transpose')

        # activation layers
        self.Softmax = tf.keras.layers.Softmax(name='Softmax')

        # reshaping layers
        self.Flatten2D_Image = tf.keras.layers.Reshape([num_rows*num_cols, encoder_dim], name='Flatten2D_Image')
        self.Flatten2D_Positional = tf.keras.layers.Reshape([num_rows*num_cols, encoder_dim], name='Flatten2D_Positional')
        self.ReshapeOutput = tf.keras.layers.Reshape([num_rows, num_cols, num_obj, -1], name='ReshapeOutput')
        
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