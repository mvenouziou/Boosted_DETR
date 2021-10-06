# imports
import tensorflow as tf

"""
This contains a keras layer used to process decoder outputs into attribute prediction inputs
"""

class PanopticNeck(tf.keras.layers.Layer):

    def __init__(self, name='PanopticNeck', **kwargs):
        super().__init__(name=name, **kwargs)

    def build(self, input_shapes):
        self.features_shape = input_shapes[0]
        num_rows = self.features_shape[1]
        num_cols = self.features_shape[2]
        num_obj = self.features_shape[3]

        # input prep
        self.ReshapeInput = tf.keras.layers.Reshape([num_rows, num_cols, -1], name='ReshapeInput')
        self.Resize = tf.keras.layers.Resizing(height=96, width=96, name='Resize')
        
        # down
        self.DownscaleBlock_0 = DownscaleBlock(num_repeats=1, name='DownscaleBlock_0')
        self.DownscaleBlock_1 = DownscaleBlock(num_repeats=1, name='DownscaleBlock_1')
        self.DownscaleBlock_2 = DownscaleBlock(num_repeats=2, name='DownscaleBlock_2')
        self.DownscaleBlock_3 = DownscaleBlock(num_repeats=3, name='DownscaleBlock_3')
 
        # up
        self.UpscaleBlock_0 = UpscaleBlock(num_repeats=3, name='UpscaleBlock_0')
        self.UpscaleBlock_1 = UpscaleBlock(num_repeats=2, name='UpscaleBlock_1')
        self.UpscaleBlock_2 = UpscaleBlock(num_repeats=1, name='UpscaleBlock_2')

        # merge components
        self.Concat_a = tf.keras.layers.Concatenate(axis=-1, name='Concat_a')
        self.Concat_b = tf.keras.layers.Concatenate(axis=-1, name='Concat_b')
        self.Concat_c = tf.keras.layers.Concatenate(axis=-1, name='Concat_c')

        self.UpscaleBlock_3 = UpscaleBlock(num_repeats=2, name='UpscaleBlock_3')
        self.DownscaleBlock_4 = DownscaleBlock(num_repeats=1, name='DownscaleBlock_4')
        
        # merge output
        self.ConcatOut = tf.keras.layers.Concatenate(axis=-1, name='ConcatOut')
        self.ConvOut = tf.keras.layers.Conv2D(filters=num_obj, kernel_size=3, strides=4, name='ConvOut')
        self.TransposeOut = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0, 3, 1, 2]), name='TransposeOut')
        self.FlattenDim = tf.keras.layers.Reshape([num_obj, -1], name='FlattenDim')

    def call(self, inputs, training=False):
        features = inputs[0]  # [batch, rows, cols, num_obj, dim]
        orig_features = self.ReshapeInput(features)
        orig_features = self.Resize(orig_features)

        # downscales
        features_down_0 = self.DownscaleBlock_0([orig_features])
        features_down_1 = self.DownscaleBlock_1([features_down_0])
        features_down_2 = self.DownscaleBlock_2([features_down_1])
        features_down_3 = self.DownscaleBlock_3([features_down_2])

        features = features_down_3
        
        # upscales with concat
        features_up_0 = self.UpscaleBlock_0([features])        
        join_a = self.Concat_a([features_up_0, features_down_2])

        features_up_1 = self.UpscaleBlock_1([features_up_0])
        join_b = self.Concat_b([features_up_1, features_down_1])
        
        features_up_2 = self.UpscaleBlock_2([features_up_1])
        join_c = self.Concat_c([features_up_2, features_down_0])

        # match shapes
        join_a = self.UpscaleBlock_3([join_a])
        join_c = self.DownscaleBlock_4([join_c])

        # merge
        features = self.ConcatOut([join_a, join_b, join_c])
        features = self.ConvOut(features)
        features = self.TransposeOut(features)
        features = self.FlattenDim(features)

        return features

    def show_summary(self):
        panoptic_features = tf.keras.layers.Input(shape=self.features_shape[1:], name='panoptic_features')
        inputs = [panoptic_features]
        outputs = self.call(inputs)  # note: use self.call() for all layers to show in summary
        return tf.keras.Model(inputs, outputs, name=self.name).summary()


class DownscaleBlock(tf.keras.layers.Layer):

    def __init__(self, num_repeats=2, **kwargs):
        super().__init__(**kwargs)
        self.num_repeats = num_repeats

    def get_config(self):
        config = super().get_config()
        config.update({'num_repeats': self.num_repeats})
        return config

    def build(self, input_shape):

        self.features_shape = input_shape[0]
        rows = self.features_shape[1]
        cols = self.features_shape[2]
        orig_filters = self.features_shape[3]
        
        # enforce shape layer
        self.InitReshape = tf.keras.layers.Reshape([rows, cols, -1], name='InitReshape')

        # conv block
        filters = orig_filters
        self.ConvBlocks = []

        for i in range(self.num_repeats):
            filters = 2*filters // 3
            conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=2, name=f'Conv2D_{i}')
            norm = tf.keras.layers.LayerNormalization(name=f'LayerNormalization_{i}')
            relu = tf.keras.layers.ReLU(max_value=None, negative_slope=.01, threshold=0, name=f'ReLU_{i}')

            self.ConvBlocks.extend([conv, norm, relu])
            
    def call(self, inputs):
        features = inputs[0]

        features = self.InitReshape(features)
        for i in range(len(self.ConvBlocks)):
            features = self.ConvBlocks[i](features)

        return features
    
    def show_summary(self):
        input_features = tf.keras.layers.Input(shape=self.features_shape[1:], name='input_features')
        inputs = [input_features]
        outputs = self.call(inputs)  # note: use self.call() for all layers to show in summary
        return tf.keras.Model(inputs, outputs, name=self.name).summary()


class UpscaleBlock(tf.keras.layers.Layer):

    def __init__(self, num_repeats=2, **kwargs):
        super().__init__(**kwargs)
        self.num_repeats = num_repeats
    
    def get_config(self):
        config = super().get_config()
        config.update({'num_repeats': self.num_repeats})
        return config
       
    def build(self, input_shape):

        self.features_shape = input_shape[0]
        rows = self.features_shape[1]
        cols = self.features_shape[2]
        orig_filters = self.features_shape[3]
        
        # enforce shape layer
        self.InitReshape = tf.keras.layers.Reshape([rows, cols, -1], name='InitReshape')

        # conv block
        filters = orig_filters
        self.ConvBlocks = []

        for i in range(self.num_repeats):
            filters = 3*filters // 2
            conv = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=2, name=f'Conv2D_{i}')
            norm = tf.keras.layers.LayerNormalization(name=f'LayerNormalization_{i}')
            relu = tf.keras.layers.ReLU(max_value=None, negative_slope=.01, threshold=0, name=f'ReLU_{i}')

            self.ConvBlocks.extend([conv, norm, relu])


    def call(self, inputs):
        features = inputs[0]

        features = self.InitReshape(features)
        for i in range(len(self.ConvBlocks)):
            features = self.ConvBlocks[i](features)

        return features
    
    def show_summary(self):
        input_features = tf.keras.layers.Input(shape=self.features_shape[1:], name='input_features')
        inputs = [input_features]
        outputs = self.call(inputs)  # note: use self.call() for all layers to show in summary
        return tf.keras.Model(inputs, outputs, name=self.name).summary()