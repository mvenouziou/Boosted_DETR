# imports
import tensorflow as tf
import backbone
#import panoptic_neck
import tokenizers
import transformers
import prediction_heads
import losses_and_metrics



class DETR(tf.keras.Model):
    """
    Note: Losses are built directly into the model. 
    Note: Training data box vals must be provided in COCO format!

    This is a DETR-like model for fine-grained object detection and description.
    DETR is published under the Apache License 2.0. This model was independently
    coded based on the paper "End-to-end Object Detection with Transformers"
    by Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier,
    Alexander Kirillov and Sergey Zagoruyko available at
    https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers.

    Note: There is a hard-coded line of the form 'use_intermediate_losses = False'
    In the call() function to control whether intermediate losses are used in 
    gradient calculations. Recommend keeping this off unless larger batch sizes 
    and longer training runs are possible.
    """

    def __init__(self, num_object_preds, image_size,
                       num_encoder_blocks, num_encoder_heads, encoder_dim,
                       num_decoder_blocks, num_decoder_heads, decoder_dim,
                       num_panoptic_heads, panoptic_dim,
                       vocab_dict, attribute_weight=1.0, name='DETR', **kwargs):
        super().__init__(name=name)

        # loss weights. 
        category_weight = 1.0
        box_weight = 1.0
        attribute_weight = attribute_weight  # should set to 0 if attributes not labeled in training set
        exist_weight = 1.0

        self.num_object_preds = num_object_preds  # ideally >> max objects in training set
        self.image_size = image_size
        self.num_encoder_blocks = num_encoder_blocks
        self.num_encoder_heads = num_encoder_heads
        self.encoder_dim = encoder_dim
        self.num_decoder_blocks = num_decoder_blocks
        self.num_decoder_heads = num_decoder_heads
        self.decoder_dim = decoder_dim
        self.num_panoptic_heads = num_panoptic_heads
        self.panoptic_dim = panoptic_dim
        self.vocab_dict = vocab_dict

        # Tokenizer Layers
        self.Tokenization = tokenizers.Tokenization(vocab_dict=self.vocab_dict, name='Tokenization')
        self.InverseTokenization = tokenizers.InverseTokenization(vocab_dict=self.vocab_dict)

        # vocab sizes includes OOV and PAD
        vocab_size_dict = self.Tokenization.vocab_size_dict()
        self.num_categories = vocab_size_dict['category']
        self.num_attributes = vocab_size_dict['attributes']

        # Layers - feature extraction
        self.EncoderBackbone = backbone.EncoderBackbone(
                                                image_input_shape=self.image_size,
                                                name='EncoderBackbone')

        self.BackboneNeck = backbone.BackboneNeck(encoder_dim=self.encoder_dim,
                                                  name='BackboneNeck')

        self.ImageEncoderAttention = transformers.ImageEncoderAttention(
                                                        num_blocks=self.num_encoder_blocks,
                                                        num_attention_heads=self.num_encoder_heads,
                                                        name='ImageEncoderAttention')

        self.DecoderPrep = transformers.DecoderPrep(num_object_preds,
                                                    decoder_dim,
                                                    name='DecoderPrep')

        # Decoding layers
        # (first block includes no self attention)
        self.DecoderBlocks = []

        self.DecoderBlocks.append(transformers.DecoderBlock_NoSelfAttention(
                                        num_attention_heads=self.num_decoder_heads,
                                        name=f'DecoderBlock_0'))

        for i in range(1, self.num_decoder_blocks):
            self.DecoderBlocks.append(transformers.DecoderBlock(
                                            num_attention_heads=self.num_decoder_heads,
                                            name=f'DecoderBlock_{i}'))

        # Layers - prediction heads
        self.CategoryPredictionHead = prediction_heads.SingleClassPredictionHead(
                                            num_classes=self.num_categories,
                                            hidden_dim=4*self.decoder_dim,
                                            num_preds=self.num_object_preds,
                                            name='CategoryPredictionHead')

        self.AttributePredictionHead = prediction_heads.MultiClassPredictionHead(
                                            num_classes=self.num_attributes,
                                            hidden_dim=4*self.decoder_dim,
                                            num_preds=self.num_object_preds,
                                            name='AttributePredictionHead')

        self.BoxPredictionHead = prediction_heads.BoxPredictionHead(
                                            hidden_dim=self.decoder_dim,
                                            num_preds=self.num_object_preds,
                                            name='BoxPredictionHead')

        # helper layers
        self.Add = tf.keras.layers.Add(name='Add')
        self.Concat = tf.keras.layers.Concatenate(axis=-1, name='Concat')
        self.BboxPrep = tokenizers.BboxPrep(name='BboxPrep')

        # Loss Layers
        self.loss_fn = losses_and_metrics.MatchingLoss(
                                            category_weight=category_weight, 
                                            box_weight=box_weight, 
                                            attribute_weight=attribute_weight, 
                                            exist_weight=exist_weight, 
                                            name='MatchingLoss')

    def get_config(self):
        config = super().get_config()
        config.update({'num_object_preds': self.num_object_preds,
                       'image_size': self.image_size,
                       'num_encoder_blocks': self.num_encoder_blocks,
                       'num_encoder_heads': self.num_encoder_heads,
                       'encoder_dim': self.encoder_dim,
                       'num_decoder_blocks': self.num_decoder_blocks,
                       'num_decoder_heads': self.num_decoder_heads,
                       'decoder_dim': self.decoder_dim,
                       'num_panoptic_heads': self.num_panoptic_heads,
                       'panoptic_dim': self.panoptic_dim,
                       'vocab_dict': self.vocab_dict,
        })
        return config

    def call(self, inputs, training=False):

        # get inputs
        image = inputs['image']

        if training:
            # prepare targets
            category = inputs['category'] 
            attribute = inputs['attribute']
            bbox = inputs['bbox']
            num_objects = inputs['num_objects']

            category, attribute = self.Tokenization([category, attribute])  # outputs one-hot cats and multihot attributes
            y_true = [category, attribute, bbox, num_objects]

        # Encoder (backbone)
        encoder_features = self.EncoderBackbone([image], training=training)  # [batch, rows, cols, encoder_dim]
        encoder_features = self.BackboneNeck([encoder_features], training=training)

        # Encoder (transformers)
        encoder_features, positional_encoding = self.ImageEncoderAttention([encoder_features], training=training)  # [batch, encoder_dim]

        # Initialize / reshape variables for decoder
        encoder_features, decoder_features, encoder_key, decoder_positional \
            = self.DecoderPrep([encoder_features, positional_encoding], training=training)

        # Decoder (transformers)
        loss = 0.0
        cat_loss = 0.0
        att_loss = 0.0
        box_loss = 0.0
        exist_loss = 0.0
        iou_metric = 0.0

        use_intermediate_losses = False
        for i in range(self.num_decoder_blocks):
            decoder_features = self.DecoderBlocks[i](
                [encoder_features, decoder_features, encoder_key, decoder_positional],
                training=training)

            if training:
                if use_intermediate_losses or i >= self.num_decoder_blocks-1:

                    # compute intermediary losses / predictions at each decoder step
                    cat_preds_i = self.CategoryPredictionHead([decoder_features], training=training)  # [batch, objs, num_cats]
                    attribute_preds_i = self.AttributePredictionHead([decoder_features], training=training)  # [batch, objs, num_attributes]
                    box_coord_preds_i = self.BoxPredictionHead([decoder_features], training=training)  # [batch, objs, 4 coords]
                    y_pred_i = [cat_preds_i, attribute_preds_i, box_coord_preds_i]

                    # compute losses and metrics
                    losses_i, metrics_i = self.loss_fn([y_true, y_pred_i])

                    # accumulate losses
                    loss_i, cat_loss_i, att_loss_i, box_loss_i, exist_loss_i = losses_i

                    loss = loss + loss_i
                    cat_loss = cat_loss + cat_loss_i
                    att_loss = att_loss + att_loss_i
                    box_loss = box_loss + box_loss_i
                    exist_loss = exist_loss + exist_loss_i

        if training:
            # record loss
            self.add_loss(loss)  ###### Critical component! Do not alter this!!

            # collect predictions (from final step)
            y_pred = y_pred_i

            # report losses (from final step)
            self.add_metric(cat_loss, 'Category_Loss')
            self.add_metric(att_loss, 'Attribute_Loss')
            self.add_metric(box_loss, 'Box_Loss')
            self.add_metric(exist_loss, 'Existence_Loss')

            # report metrics (from final step)
            iou_metric_i = metrics_i[0]
            self.add_metric(iou_metric_i, 'IOU')

            return y_pred

        # Prediction Heads
        cat_preds = self.CategoryPredictionHead([decoder_features], training=training)  # [batch, objs, num_cats]
        attribute_preds = self.AttributePredictionHead([decoder_features], training=training)  # [batch, objs, num_attributes]
        box_coord_preds = self.BoxPredictionHead([decoder_features], training=training)  # [batch, objs, 4 coords]

        # Translate results into text descriptions
        category, attributes = self.InverseTokenization([cat_preds, attribute_preds], training=training)

        return category, attributes, box_coord_preds

    def test_step(self, inputs):
        return self.train_step(inputs)

    def citation(self):
        print('''DETR-like model for object detection and fine-grained classification.
        DETR is published under the Apache License 2.0. This model was independently
        coded in Tensorflow based on the paper "End-to-end Object Detection with Transformers"
        by Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov
        and Sergey Zagoruyko, available at
        https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers.''')
