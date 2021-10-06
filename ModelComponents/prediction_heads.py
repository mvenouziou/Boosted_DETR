# imports
import tensorflow as tf

"""
This contains a keras layer used to create the final model predictions. It returns
a tensor of shape (batch, num object predictions, categories), where logits
has been predicted for each category.

Layers:
PredictionHead()
"""

class BoxPredictionHead(tf.keras.layers.Layer):
    """ predicts normalized VOC coords (i.e. in [0,1], as a percentage of full image size)
    
    input: feature vector [batch, *dims, feature_dim]
    outputs: prediction logits [batch, num_obj, num_supercategories]
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
        self.Conv1D = tf.keras.layers.Conv1D(filters=self.num_preds, kernel_size=1, data_format='channels_first')

        # SuperCatagory (objects have exactly one SuperCatagory)
        self.Dense = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='Dense')
        self.BatchNorm = tf.keras.layers.BatchNormalization(name='BatchNorm')
        self.BoxCoords = tf.keras.layers.Dense(4, activation=None, name='BoxCoords')  
        self.Sigmoid = tf.keras.layers.Lambda (lambda x: tf.math.sigmoid(x), name='Sigmoid', dtype=tf.float32)  # enforce dtype on final activation
       
    def call(self, inputs, training=False):
        features = inputs[0]  # [batch, *None, decoder dim]
        
        # update to num obj
        features = self.Reshape(features)
        num_preds_current = tf.shape(features)[1]

        if tf.math.not_equal(self.num_preds, num_preds_current):
            features = self.Conv1D(features)  # [batch, num_obj, hidden_dim]

        # update to 4 box coords
        features = self.Dense(features, training=training)   # [batch, num_obj, hidden_dim]
        features = self.BatchNorm(features, training=training)  
        bbox = self.BoxCoords(features, training=training)  # [batch, num_obj, 4]       
        bbox = self.Sigmoid(bbox, training=training)
        return bbox

    def show_summary(self):
        decoder_features = tf.keras.layers.Input(shape=self.features_shape[1:], name='decoder_features')
        inputs = [decoder_features]
        outputs = self.call(inputs)  # note: use self.call() for all layers to show in summary
        return tf.keras.Model(inputs, outputs, name=self.name).summary()


class CategoryPredictionHead(tf.keras.layers.Layer):
    """
    input: feature vector [batch, num obj preds, feature_dim]
    outputs: prediction logits [batch, num_obj, num_categories]
    """

    def __init__(self, num_categories, hidden_dim, num_preds, name='CategoryPredictionHead', **kwargs):
        super().__init__(name=name, **kwargs)

        self.num_categories = num_categories
        self.hidden_dim = hidden_dim
        self.num_preds = num_preds

    def get_config(self):
        config = super().get_config()
        config.update({'num_categories': self.num_categories,
                       'hidden_dim': self.hidden_dim, 'num_preds':self.num_preds})
        return config

    def build(self, input_shape):
        
        self.features_shape = input_shape[0]  # used in self.show_summary()
        num_obj = self.features_shape[1]
        features_dim = self.features_shape[-1]

        # match num desired preds
        self.Reshape = tf.keras.layers.Reshape([-1, features_dim])
        self.Conv1D = tf.keras.layers.Conv1D(filters=self.num_preds, kernel_size=1, data_format='channels_first')

        # Catagory (objects have exactly one Catagory)
        self.DenseCateg = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='Dense')
        self.BatchNormCateg = tf.keras.layers.BatchNormalization(name='BatchNorm')
        self.LogitsCateg = tf.keras.layers.Dense(self.num_categories, activation=None, name='Logits')
        self.Softmax = tf.keras.layers.Softmax(name='Probs', dtype=tf.float32)

    def call(self, inputs, training=False):
        features = inputs[0]  # [batch, *None, decoder dim]

        # update to num obj
        features = self.Reshape(features)
        num_preds_current = tf.shape(features)[1]

        if tf.math.not_equal(self.num_preds, num_preds_current):
            features = self.Conv1D(features)  # [batch, num_obj, hidden_dim]

        # make preds
        features = self.DenseCateg(features, training=training)   # [batch, num_obj, hidden_dim]
        features = self.BatchNormCateg(features, training=training)  
        features = self.LogitsCateg(features, training=training)  
        features = self.Softmax(features, training=training)  # [batch, num_obj, num_categories]
        
        return features

    def show_summary(self):
        decoder_features = tf.keras.layers.Input(shape=self.features_shape[1:], name='decoder_features')
        inputs = [decoder_features]
        outputs = self.call(inputs)  # note: use self.call() for all layers to show in summary
        return tf.keras.Model(inputs, outputs, name=self.name).summary()


class AttributePredictionHead(tf.keras.layers.Layer):

    def __init__(self, num_attributes, hidden_dim, num_preds, name='AttributePredictionHead', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_attributes = num_attributes
        self.hidden_dim = hidden_dim
        self.num_preds = num_preds

    def get_config(self):
        config = super().get_config()
        config.update({'num_attributes': self.num_attributes,
                       'hidden_dim': self.hidden_dim, 'num_preds':self.num_preds})
        return config

    def build(self, input_shape):
        
        self.features_shape = input_shape[0]  # used in self.show_summary()
        features_dim = self.features_shape[-1]

        # match num desired preds
        self.Reshape = tf.keras.layers.Reshape([-1, features_dim])
        self.Conv1D = tf.keras.layers.Conv1D(filters=self.num_preds, kernel_size=1, data_format='channels_first')

        # Attributes (objects one or more attributes (including padding))
        self.Dense = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='Dense') 
        self.BatchNorm = tf.keras.layers.BatchNormalization(name='BatchNorm')
        self.DenseLinear = tf.keras.layers.Dense(self.num_attributes, activation=None, name='DenseLinear')
        self.Sigmoid = tf.keras.layers.Lambda (lambda x: tf.math.sigmoid(x), name='Sigmoid', dtype=tf.float32)  # enforce dtype on final activation

    def call(self, inputs, training=False):
        features = inputs[0]  # [batch, *None, num_obj]
        
        # update to num obj
        features = self.Reshape(features)
        num_preds_current = tf.shape(features)[1]   # [batch, num_preds_current, hidden_dim]

        if tf.math.not_equal(self.num_preds, num_preds_current):
            features = self.Conv1D(features)  # [batch, num_obj, hidden_dim]

        # make preds
        features = self.Dense(features, training=training)  # [batch, num_obj, hidden_dim] 
        features = self.BatchNorm(features, training=training)  
        features = self.DenseLinear(features, training=training)   # [batch, num_obj, num_attributes]
        features = self.Sigmoid(features, training=training)  # [batch, num_obj, num_attributes]
        
        return features

    def show_summary(self):
        features = tf.keras.layers.Input(shape=self.features_shape[1:], name='features')
        inputs = [features]
        outputs = self.call(inputs)  # note: use self.call() for all layers to show in summary
        return tf.keras.Model(inputs, outputs, name=self.name).summary()
