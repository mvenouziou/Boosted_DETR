# imports
import tensorflow as tf

"""
This contains a keras layer used to create the final model predictions. It returns
a tensor of shape (batch, num object predictions, categories), where probs
has been predicted for each component.

Layers:
PredictionHead()
"""

class BoxPredictionHead(tf.keras.layers.Layer):
    """
    input: feature vector [batch, *dims, feature_dim]
    outputs: [batch, num_obj, 4]
    """

    def __init__(self, hidden_dim, num_preds, name='BoxPredictionHead', **kwargs):
        super().__init__(name=name, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_preds = num_preds

    def get_config(self):
        config = super().get_config()
        config.update({'hidden_dim': self.hidden_dim, 'num_preds': self.num_preds})
        return config

    def build(self, input_shape):

        self.features_shape = input_shape[0]  # used in self.show_summary()
        features_dim = self.features_shape[-1]

        # match num desired preds
        self.Reshape = tf.keras.layers.Reshape([-1, features_dim])
        self.Conv1D = tf.keras.layers.Conv1D(filters=self.num_preds, kernel_size=1)
        self.Permute = tf.keras.layers.Permute([2,1])

        # SuperCatagory (objects have exactly one SuperCatagory)
        self.Dense = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='Dense')
        self.BatchNorm = tf.keras.layers.BatchNormalization(name='BatchNorm')
        self.BoxCoords = tf.keras.layers.Dense(4, activation=None, name='BoxCoords')
        self.Sigmoid = tf.keras.layers.Lambda (lambda x: tf.math.sigmoid(x), name='Sigmoid', dtype=tf.float32)

    def call(self, inputs, training=False):
        features = inputs[0]  # [batch, *None, decoder dim]

        # update to num obj
        features = self.Reshape(features)
        num_preds_current = tf.shape(features)[1]

        if tf.math.not_equal(self.num_preds, num_preds_current):
            features = self.Permute(features)  # [batch, features_dim, num_preds_current]
            features = self.Conv1D(features)  # [batch, features_dim, num_obj]
            features = self.Permute(features)  # [batch, num_obj, features_dim]

        # update to 4 box coords
        features = self.Dense(features)   # [batch, num_obj, hidden_dim]
        features = self.BatchNorm(features, training=training)
        bbox = self.BoxCoords(features)  # [batch, num_obj, 4]
        bbox = self.Sigmoid(bbox)
        return bbox

    def show_summary(self):
        decoder_features = tf.keras.layers.Input(shape=self.features_shape[1:], name='decoder_features')
        inputs = [decoder_features]
        outputs = self.call(inputs)  # note: use self.call() for all layers to show in summary
        return tf.keras.Model(inputs, outputs, name=self.name).summary()


class SingleClassPredictionHead(tf.keras.layers.Layer):
    """
    input: feature vector [batch, num obj preds, feature_dim]
    outputs: [batch, num_obj, num_categories]
    """

    def __init__(self, num_classes, hidden_dim, num_preds, name='SingleClassPredictionHead', **kwargs):
        super().__init__(name=name, **kwargs)

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_preds = num_preds

    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes,
                       'hidden_dim': self.hidden_dim, 'num_preds': self.num_preds})
        return config

    def build(self, input_shape):

        self.features_shape = input_shape[0]  # used in self.show_summary()
        num_obj = self.features_shape[1]
        features_dim = self.features_shape[-1]

        # match num desired preds
        self.Reshape = tf.keras.layers.Reshape([-1, features_dim], name='Reshape')
        self.Conv1D = tf.keras.layers.Conv1D(filters=self.num_preds, kernel_size=1,
                                             name='Conv1D')
        self.Permute = tf.keras.layers.Permute([2,1])                                             

        # Catagory (objects have exactly one Catagory)
        self.DenseCateg = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='DenseCateg')
        self.BatchNorm = tf.keras.layers.BatchNormalization(name='BatchNorm')
        self.DenseLogits = tf.keras.layers.Dense(self.num_classes, activation=None, name='DenseLogits')
        self.Softmax = tf.keras.layers.Softmax(name='Softmax', dtype=tf.float32)

    def call(self, inputs, training=False):
        features = inputs[0]  # [batch, *None, features_dim]

        # update shape
        features = self.Reshape(features)  # [batch, -1, features_dim]
        num_preds_current = tf.shape(features)[1]

        if tf.math.not_equal(self.num_preds, num_preds_current):
            features = self.Permute(features)  # [batch, features_dim, num_preds_current]
            features = self.Conv1D(features)  # [batch, features_dim, num_obj]
            features = self.Permute(features)  # [batch, num_obj, features_dim]

        # make preds
        features = self.DenseCateg(features)   # [batch, num_obj, hidden_dim]
        features = self.BatchNorm(features, training=training)
        features = self.DenseLogits(features)   # [batch, num_obj, num_classes]
        features = self.Softmax(features)

        return features

    def show_summary(self):
        decoder_features = tf.keras.layers.Input(shape=self.features_shape[1:], name='decoder_features')
        inputs = [decoder_features]
        outputs = self.call(inputs)  # note: use self.call() for all layers to show in summary
        return tf.keras.Model(inputs, outputs, name=self.name).summary()


class MultiClassPredictionHead(tf.keras.layers.Layer):

    def __init__(self, num_classes, hidden_dim, num_preds, name='MultiClassPredictionHead', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_preds = num_preds

    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes,
                       'hidden_dim': self.hidden_dim, 'num_preds':self.num_preds})
        return config

    def build(self, input_shape):

        self.features_shape = input_shape[0]  # used in self.show_summary()
        features_dim = self.features_shape[-1]

        # match num desired preds
        self.Reshape = tf.keras.layers.Reshape([-1, features_dim], name='Reshape')
        self.Conv1D = tf.keras.layers.Conv1D(filters=self.num_preds,
                                             kernel_size=1,
                                             name='Conv1D')

        self.Permute = tf.keras.layers.Permute([2,1])

        # Attributes (objects one or more attributes (including padding))
        self.Dense = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='Dense')
        self.BatchNorm = tf.keras.layers.BatchNormalization(name='BatchNorm')
        self.DenseLinear = tf.keras.layers.Dense(self.num_classes, activation=None, name='DenseLinear')
        self.Sigmoid = tf.keras.layers.Lambda (lambda x: tf.math.sigmoid(x), name='Sigmoid', dtype=tf.float32)

    def call(self, inputs, training=False):
        features = inputs[0]  # [batch, *None, features_dim]

        # update to correct shape
        features = self.Reshape(features)  # [batch, num_preds_current, features_dim]
        num_preds_current = tf.shape(features)[1]

        if tf.math.not_equal(self.num_preds, num_preds_current):
            # note: permutations used instead of 'channels_first' for CPU training compat
            features = self.Permute(features)  # [batch, features_dim, num_preds_current]
            features = self.Conv1D(features)  # [batch, features_dim, num_obj]
            features = self.Permute(features)  # [batch, num_obj, features_dim]

        # make preds
        features = self.Dense(features)  # [batch, num_obj, hidden_dim]
        features = self.BatchNorm(features, training=training)
        features = self.DenseLinear(features)   # [batch, num_obj, num_classes]
        features = self.Sigmoid(features)

        return features

    def show_summary(self):
        features = tf.keras.layers.Input(shape=self.features_shape[1:], name='features')
        inputs = [features]
        outputs = self.call(inputs)  # note: use self.call() for all layers to show in summary
        return tf.keras.Model(inputs, outputs, name=self.name).summary()
