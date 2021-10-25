# imports
import tensorflow as tf
import tensorflow_addons as tfa
import tokenizers
import prediction_heads
import model


class DETR_MultiClassifier(tf.keras.Model):
    """
    This is an adaption that takes an existing DETR model created above and treats
    it as a (multilabel) classifier. It yields binary predictions for the
    presence of each category existing in the image.

    The primary use case is as a pre-trainer for the full object detection model.
    It shares weights with the base DETR model up until prediction, where it
    switches to its own classification head.

    NOTE: It yields num_objects predictions to match DETR's number of box predictions.
    The loss function only uses the best of these predictions in calculations.
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

        self.DecoderBlocks = []
        for decoder in self.base_model.DecoderBlocks:
            self.DecoderBlocks.append(decoder)

        # New Tokenizer for custom vocab
        self.Tokenization = tokenizers.Tokenization(vocab_dict=self.vocab_dict)
        self.InverseTokenization = tokenizers.InverseTokenization(vocab_dict=self.vocab_dict)

        vocab_size_dict = self.Tokenization.vocab_size_dict()
        self.num_categories = vocab_size_dict['category']

        # New prediction head
        self.MultiClassPredictionHead = prediction_heads.MultiClassPredictionHead(
                                            num_classes=self.num_categories,
                                            hidden_dim=self.hidden_dim,
                                            num_preds=base_model.num_object_preds,
                                            name='MultiClassPredictionHead')

        # loss
        self.loss_fn = tfa.losses.sigmoid_focal_crossentropy
        self.binary_accuracy = tf.keras.metrics.BinaryAccuracy()

    def get_config(self):
        config = super().get_config()
        config.update({'vocab_dict': self.num_categories,
                       'hidden_dim': self.hidden_dim})

    def call(self, inputs, training=False):

        image = inputs['image']

        if training:
            # prepare targets
            category = inputs['category']  # [batch, num obj, 1 class label]
            batch_size = tf.shape(category)[0]

            # convert labels to multi-hot
            y_hot, _ = self.Tokenization([category, category])
            y_multihot = tf.reduce_max(y_hot, axis=1, keepdims=True)

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
        loss = 0.0
        step_loss = 0.0
        for i in range(self.base_model.num_decoder_blocks):
            decoder_features = self.DecoderBlocks[i](
                [encoder_features, decoder_features, encoder_positional, decoder_positional],
                training=training)

            # add intermediate losses
            if training:
                preds = self.MultiClassPredictionHead([decoder_features], training=training)
                preds = tf.reduce_max(preds, axis=1, keepdims=True)  # take highest pred from each object container

                step_loss = self.loss_fn(y_multihot, preds)
                step_loss = tf.math.reduce_min(step_loss, axis=1)  # choose best prediction

                #step_loss = tf.math.reduce_mean(step_loss)  # mean over batches
                loss = loss + step_loss

        if training:
            self.add_loss(loss)  ###### vital component, do not alter this!!

            accuracy = self.binary_accuracy(y_multihot, preds)
            self.add_metric(accuracy, 'accuracy')
            return preds

        # NEW PREDICTION HEAD
        preds = self.MultiClassPredictionHead([decoder_features], training=training)
        preds = tf.reduce_max(preds, axis=1, keepdims=True)  # take highest pred from each object container
        
        return preds