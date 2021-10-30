# imports
import tensorflow as tf

"""
This contains keras layers used to encode feature vectors from an image, with an
additional positional encoding added. The layer incldes options for base backbone model.

Layers:
EncoderBackbone() # processes (image) inputs
BackboneNeck()  # downscales encoding dim to fixed size
"""


# Image Encoders
class EncoderBackbone(tf.keras.layers.Layer):

    def __init__(self, image_input_shape, model_name='EfficientNet', **kwargs):
        super().__init__(dtype=tf.float32, **kwargs)

        self.image_input_shape = image_input_shape
        height, width = self.image_input_shape[:2]

        # load image CNN structure
        # DETR paper uses ResNet, bur Efficient seems a better choice.
        if model_name == 'EfficientNet':
            """Note: EfficientNet models expect their inputs to be float tensors 
            of pixels with values in the [0-255] range. Preprocessing handled by model"""
            self.ImageFeaturesExtractor = tf.keras.applications.EfficientNetB4(
                                    include_top=False,
                                    weights=None,  #'imagenet',
                                    input_shape=[height, width, 3])
            self.preprocessor = tf.keras.layers.Lambda(lambda x: x)  # pass through

        elif model_name == 'ResNet':
            """Note: ResNet requires preprocessor. Preprocessor expects inputs 
            to be float tensors of pixels with values in the [0-255] range."""
            self.ImageFeaturesExtractor = tf.keras.applications.resnet50.ResNet50(
                    include_top=False, weights='imagenet', input_shape=[height, width, 3])
            self.preprocessor = tf.keras.applications.resnet50.preprocess_input

        self.convert_dtype = tf.keras.layers.Lambda(lambda image:
                tf.cast(tf.image.convert_image_dtype(image, dtype=tf.uint8), tf.float32))

        self.Resize = tf.keras.layers.Resizing(height=height, width=width, name='Resizing')

    def config(self):
        return super().get_config()

    def call(self, inputs):
        image = inputs[0]  # images should enter in [0,1] scale
        image = tf.clip_by_value(image, 0.0, 1.0)

        # prepare image and apply CNN model
        image = self.Resize(image)
        image = self.convert_dtype(image)
        image = self.preprocessor(image)
        image = self.ImageFeaturesExtractor(tf.cast(image, tf.float32))
        return image

    def show_summary(self):
        image = tf.keras.layers.Input(shape=self.image_input_shape, name='image')
        inputs = [image]
        outputs = self.call(inputs)  # note: use self.call() for all layers to show in summary
        return tf.keras.Model(inputs, outputs, name=self.name).summary()

class BackboneNeck(tf.keras.layers.Layer):
    """
    Extra processing after backbone layer, updates feature dim to desired size
    """

    def __init__(self, encoder_dim, name='BackboneNeck', **kwargs):
        super().__init__(name=name, **kwargs)
        self.encoder_dim = encoder_dim

        # projection (downscaling feature dim)
        self.conv2d_downscaler = tf.keras.layers.Conv2D(filters=self.encoder_dim, 
                kernel_size=1, kernel_initializer='lecun_normal', activation='tanh', 
                name='conv2d_downscaler')
        self.batch_norm1 = tf.keras.layers.BatchNormalization(name='batch_norm1')
        self.batch_norm2 = tf.keras.layers.BatchNormalization(name='batch_norm2')

    def get_config(self):
        config = super().get_config()
        config.update({'encoder_dim': self.encoder_dim})
        return config

    def build(self, input_shape):
        self.features_shape = input_shape[0]

    def call(self, inputs, training=False):
        features = inputs[0]
        features = self.batch_norm1(features, training=training)
        features = self.conv2d_downscaler(features, training=training)
        features = self.batch_norm2(features, training=training)
        return features

    def show_summary(self):
        features = tf.keras.layers.Input(shape=self.features_shape[1:], name='features')
        inputs = [features]
        outputs = self.call(inputs)  # note: use self.call() for all layers to show in summary
        return tf.keras.Model(inputs, outputs, name=self.name).summary()

    def cnn_layer(self):
        return self.feature_extractor_layer
