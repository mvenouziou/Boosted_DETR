# imports
import tensorflow as tf


class Tokenization(tf.keras.layers.Layer):
    """
    Used to convert strings to multi-hot labels
    Text input should be string descriptions of the form 'supercategory, category, feature_1, feature_2, ...',
    with one string per object, potentially multiple objects per image, multiple images per batch.
    """

    def __init__(self, vocab_dict, name='Tokenization', **kwargs):
        super().__init__(self, name=name, **kwargs)

        # attributes
        self.vocab_dict = vocab_dict
        self.mask_token = '<PAD>'
        self.out_of_vocab_token = '<OOV>'

        # Tokenizer Layers:
        # Category
        self.tokenizer_category = tf.keras.layers.experimental.preprocessing.StringLookup(
                invert=False, vocabulary=self.vocab_dict['category'], output_mode='int',
                oov_token=self.out_of_vocab_token, mask_token=self.mask_token)        

        # Features
        self.tokenizer_attributes = tf.keras.layers.experimental.preprocessing.StringLookup(
                invert=False, vocabulary=self.vocab_dict['attribute'], output_mode='int',
                oov_token=self.out_of_vocab_token, mask_token=self.mask_token)        

        # extra attribute
        self._vocab_size_category = len(self.tokenizer_category.get_vocabulary(include_special_tokens=True))
        self._vocab_size_attributes = len(self.tokenizer_attributes.get_vocabulary(include_special_tokens=True))
    
    def get_config(self):
        config = super().get_config()
        config.update({'vocab_dict': self.vocab_dict,})
        return config

    def call(self, inputs):
        """ Converts strings to multihot vector indicating category and attributes. """
        
        category, attributes = inputs

        # tokenize 
        sparce_category = self.tokenizer_category(category)  # [batch, num_objects, 1]        
        sparce_category = tf.squeeze(sparce_category, axis=2)  # [batch, num_objects]
        sparce_attributes = self.tokenizer_attributes(attributes)  # [batch, num_objects, num_attributes]
 
        # convert to hot / multihot vector
        multihot_vectors = self.sparce_to_multihot(sparce_category, sparce_attributes)

        return multihot_vectors  # [one_hot_category, multi_hot_attributes]

    def sparce_to_multihot(self, sparce_category, sparce_attributes):
        """ 
        Converts spacre tokens to multihot vectors. 

        inputs:
        sparce_category: [batch, num objects, 1]
        sparce_attributes: [batch, num objects, num_padded_words]

        output: 
        multihot vector: [batch, num sentences, vocab_size_category + vocab_size_attributes]
        """

        # convert to one-hot
        on_value = 1
        off_value = 0
        axis = -1

        one_hot_category = tf.one_hot(sparce_category, depth=self._vocab_size_category, on_value=on_value, off_value=off_value, axis=axis)    # [batch, num objects, 1, vocab_size_category]
        one_hot_attributes = tf.one_hot(sparce_attributes, depth=self._vocab_size_attributes, on_value=on_value, off_value=off_value, axis=axis)    # [batch, num objects, num_padded_words, vocab_size_attributes]

        # convert attributes to multihot
        multi_hot_attributes = tf.math.reduce_max(one_hot_attributes, axis=2)  # [batch, num objects, vocab_size_attributes]

        # combine
        multihot_vectors = [tf.cast(one_hot_category, tf.float32), 
                            tf.cast(multi_hot_attributes, tf.float32)]
        
        return multihot_vectors

    # Access attributes
    def vocab_size_dict(self):
        sizes = {'category': self._vocab_size_category,
                 'attributes': self._vocab_size_attributes}
        return sizes


class InverseTokenization(tf.keras.layers.Layer):
    """ Used to convert strings to tokens or multi-hot labels (and vice versa). """

    def __init__(self, vocab_dict, name='Tokenization', **kwargs):
        super().__init__(self, name=name, **kwargs)

        # attributes
        self.vocab_dict = vocab_dict
        self.mask_token = '<PAD>'
        self.out_of_vocab_token = '<OOV>'

        # Inverse Tokenizer Layers:
        # Category
        self.inverse_tokenizer_category = tf.keras.layers.experimental.preprocessing.StringLookup(
                invert=True, vocabulary=self.vocab_dict['category'], output_mode='int',
                oov_token=self.out_of_vocab_token, mask_token=self.mask_token)        

        # Features
        self.inverse_tokenizer_attributes = tf.keras.layers.experimental.preprocessing.StringLookup(
                invert=True, vocabulary=self.vocab_dict['attribute'], output_mode='int',
                oov_token=self.out_of_vocab_token, mask_token=self.mask_token)        

        # extra attribute
        self._vocab_size_category = len(self.inverse_tokenizer_category.get_vocabulary(include_special_tokens=True))
        self._vocab_size_attributes = len(self.inverse_tokenizer_attributes.get_vocabulary(include_special_tokens=True))
    
    def get_config(self):
        config = super().get_config()
        config.update({'vocab_dict': self.vocab_dict,})
        return config

    def call(self, inputs, training=False):
        """
        Converts probs vector to text values
        """
        
        cat_preds, attribute_preds = inputs

        # get predictions (tokens)
        tokens_categories = tf.argmax(cat_preds, axis=-1)  # sparce categ (batch, num_obj)  
        tokens_categories = tf.expand_dims(tokens_categories, axis=-1)  # sparce categ (batch, num_obj, 1)  

        multihot_attributes = tf.math.greater_equal(attribute_preds, .5)  # multihot indicator (batch, num_obj, num_attributes) 
        multihot_attributes = tf.cast(multihot_attributes, dtype=tf.int32)
        tokens_attributes = multihot_attributes * tf.range(self._vocab_size_attributes)  # sparce multivector (batch, num_obj, num_attributes) 

        category, attributes = self.sparce_to_strings(tokens_categories, tokens_attributes)

        return category, attributes

    def sparce_to_strings(self, tokens_categories, tokens_attributes):

        # convert tokens to words
        category = self.inverse_tokenizer_category(tokens_categories)  # [batch, num_objects, 1 category]
        attributes = self.inverse_tokenizer_attributes(tokens_attributes)  # [batch, num_objects, num feature words]

        # drop PAD and OOV       
        attributes = tf.strings.reduce_join(attributes, axis=-1, separator=', ', keepdims=True)  # [batch, num_objects]
        attributes = tf.strings.regex_replace(attributes, self.mask_token, '')
        attributes = tf.strings.regex_replace(attributes, self.out_of_vocab_token, '')

        # strip extra whitespace
        attributes = tf.strings.strip(attributes)  # trailing / leading
        attributes = tf.strings.regex_replace(attributes, ' +', ' ')  # interior

        return category, attributes

    # Access attributes
    def vocab_size_dict(self):
        sizes = {'category': self._vocab_size_category,
                 'attributes': self._vocab_size_attributes}
        return sizes

class BboxPrep(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, keep_ragged=True):
        bbox = inputs[0]

        if keep_ragged is False:

            # get minimal bounding shape
            bounding_shape = tf.cast(bbox.bounding_shape(), dtype=tf.int32)
            batch_size = bounding_shape[0]
            padded_objects = bounding_shape[1]

            # pad to non-ragged tensor
            default_value=tf.constant(-1.0, dtype=bbox.dtype)
            bbox = bbox.to_tensor(default_value=default_value, 
                                  shape=[batch_size, padded_objects, 4])   


        return bbox

