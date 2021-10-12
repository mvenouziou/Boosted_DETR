# imports
import tensorflow as tf
import tokenizers
import prediction_heads 
import model
#import backbone
#import panoptic_neck
#import transformers
#import losses_and_metrics



class DETR_MultiClassifier(tf.keras.Model):
    """
    This is an adaption that takes an existing DETR model created above and treats
    it as a (multilabel) classifier. It yields binary predictions for the 
    presence of each category existing in the image.
    
    The primary use case is as a pre-trainer for the full object detection model.
    It shares weights with the base DETR model up until prediction, where it 
    switches to its own classification head. 
    """

    def __init__(self, base_model, vocab_dict, hidden_dim, 
                 name='DETR_MultiClassifier', **kwargs):
        super().__init__(name=name, **kwargs)

        self.base_model = base_model
        self.vocab_dict = vocab_dict
        self.hidden_dim = hidden_dim

        # Get Base Model
        self.base_model = base_model
        self.EncoderBackbone = base_model.EncoderBackbone
        self.BackboneNeck = base_model.BackboneNeck
        self.ImageEncoderAttention = base_model.ImageEncoderAttention
        self.DecoderPrep = base_model.DecoderPrep

        self.EncoderBackbone.trainable = False  # optional: lock backbone
        
        self.DecoderBlocks = []
        for decoder in self.base_model.DecoderBlocks:
            self.DecoderBlocks.append(decoder)

        # New Tokenizer
        self.Tokenization = tokenizers.Tokenization(vocab_dict=self.vocab_dict)
        self.InverseTokenization = tokenizers.InverseTokenization(vocab_dict=self.vocab_dict)
        
        vocab_size_dict = self.Tokenization.vocab_size_dict()  
        self.num_categories = vocab_size_dict['category']

        # New prediction head
        self.MultihotPredictionHead = prediction_heads.AttributePredictionHead(
                                            num_attributes=self.num_categories,
                                            hidden_dim=self.hidden_dim, 
                                            num_preds=base_model.num_object_preds,
                                            name='MultihotPredictionHead')

        # ignore original prediction heads

    def get_config(self):
        config = super().get_config()
        config.update({'num_categories': self.num_categories,
                       'hidden_dim': self.hidden_dim})

    def call(self, inputs, training=False):

        image = inputs[0]

        if training:
            # prepare targets
            y_true_class = inputs[1]  # [batch, num obj, 1 class label]

            # convert labels to multi-hot
            y_true_class = tf.strings.reduce_join(y_true_class, axis=-2)  # collapse to a single object
            y_true_class = tf.expand_dims(y_true_class, axis=1)    # [batch, 1, class labels]
            y_hot, _ = self.Tokenization([y_true_class, y_true_class])
            
            y_true = y_hot

        # BASE MODEL
        # Encoder (backbone)
        encoder_features = self.EncoderBackbone([image], training=training)  # [batch, rows, cols, encoder_dim]
        encoder_features = self.BackboneNeck([encoder_features], training=training)
        
        # Encoder (transformers)
        encoder_features, positional_encoding = \
            self.ImageEncoderAttention([encoder_features], training=training)  # [batch, encoder_dim]

        # prepare for decoder
        encoder_features, decoder_features, encoder_positional, decoder_positional \
            = self.DecoderPrep([encoder_features, positional_encoding], training=training)

        # decoder (transformers)
        for i in range(self.base_model.num_decoder_blocks):
            decoder_features = self.DecoderBlocks[i](
                [encoder_features, decoder_features, encoder_positional, decoder_positional],
                training=training)    

            # add intermediate losses
            if training:
                preds = self.MultihotPredictionHead([decoder_features], training=training)
                step_loss = self.compiled_loss(y_true, preds)
                step_loss = tf.math.reduce_min(step_loss, axis=-1)  # choose best matching pred
                self.add_loss(step_loss)

        if training:
            return y_true, preds

        # NEW PREDICTION HEAD
        preds = self.MultihotPredictionHead([decoder_features], training=training)

        return preds

    def test_step(self, inputs):
        return self.train_step(inputs, apply_grads=False)

    def train_step(self, inputs, apply_grads=True):
        training = True

        loss = 0.0
        with tf.GradientTape() as tape:
            y_true, preds = self(inputs, training=True)
            loss = self.losses  # all losses are included using self.add_loss() within call

        # get grads
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        if apply_grads:
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            print("note: ignore warning about original model's prediction heads missing gradients.")

        # Update metrics
        self.compiled_metrics.update_state(y_true, preds)
        return {m.name: m.result() for m in self.metrics}
