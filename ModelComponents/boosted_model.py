# imports
import tensorflow as tf
import backbone
#import panoptic_neck
import tokenizers
import transformers
import prediction_heads
import losses_and_metrics



class BoostedDETR(tf.keras.Model):
    """
    Note: Losses are built directly into the model. 
    Note: Training data box vals must be provided in COCO format!

    This is an original adaptation of DETR that:
    - 1) uses a boosted model structure  
    - 2) includes an optional 'attributes' prediction head to encode fine-grained 
    features in addition to overall classes.

    Small encoder-transformer blocks with individual prediction heads serve
    as the 'weak learners.' Each encoder block is connected tightly to a specific 
    decoder block, and intermediate predictions are the sum of results from all 
    prediction heads to that point.
    
    This model was independently coded based of the original DETR model 
    (published under the Apache License 2.0) developed in the paper 
    "End-to-end Object Detection with Transformers" by Nicolas Carion, 
    Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov 
    and Sergey Zagoruyko, available at
    https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers.
    
    """

    def __init__(self, num_object_preds, image_size,
                       num_encoder_blocks, num_encoder_heads, encoder_dim,
                       num_decoder_blocks, num_decoder_heads, decoder_dim,
                       num_panoptic_heads, panoptic_dim, vocab_dict, 
                       classification_only=False, attribute_weight=1.0, name='DETR', **kwargs):
        super().__init__(name=name)

        # loss weights. 
        self.use_intermediate_predictions = True

        # note: do not add to model.config() so these can be changed without saved model conflicts
        attribute_weight = attribute_weight  # can be set to 0 if attributes not labeled in training set        
        category_weight = None  # will use default weight
        box_weight = None  # will use default weight
        exist_weight = None  # will use default weight

        if classification_only:
            box_weight = 0.0

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

        # Encoder transformer layers
        self.EncoderTransformerBlocks = []
        for i in range(self.num_decoder_blocks):
            ImageEncoderAttention = transformers.ImageEncoderAttention(
                                                            num_blocks=1,
                                                            num_attention_heads=self.num_encoder_heads,
                                                            name=f'ImageEncoderAttention_{i}')

            self.EncoderTransformerBlocks.append(ImageEncoderAttention)

        self.DecoderPrep = transformers.DecoderPrep(num_object_preds,
                                                    decoder_dim,
                                                    name='DecoderPrep')

        # Decoding layers
        # (first block includes no self attention)
        self.DecoderBlocks = []

        i = 0
        self.DecoderBlocks.append(transformers.DecoderBlock_NoSelfAttention(
                                        num_attention_heads=self.num_decoder_heads,
                                        name=f'DecoderBlock_{i}'))

        for i in range(1, self.num_decoder_blocks):
            self.DecoderBlocks.append(transformers.DecoderBlock(
                                            num_attention_heads=self.num_decoder_heads,
                                            name=f'DecoderBlock_{i}'))

        # Layers - prediction heads
        # intermediate predictions used for "boosting ensemble"-inspired model
        self.CategoryBlocks = []
        self.AttributeBlocks = []
        self.BoxBlocks = []
        for i in range(self.num_decoder_blocks):
            CategoryPredictionHead = prediction_heads.SingleClassPredictionHead(
                                                num_classes=self.num_categories,
                                                hidden_dim=self.decoder_dim,
                                                num_preds=self.num_object_preds,
                                                name=f'CategoryPredictionHead_{i}')

            self.CategoryBlocks.append(CategoryPredictionHead)

            AttributePredictionHead = prediction_heads.MultiClassPredictionHead(
                                                num_classes=self.num_attributes,
                                                hidden_dim=self.decoder_dim,
                                                num_preds=self.num_object_preds,
                                                name=f'AttributePredictionHead_{i}')
            
            self.AttributeBlocks.append(AttributePredictionHead)

            BoxPredictionHead = prediction_heads.BoxPredictionHead(
                                                hidden_dim=self.decoder_dim,
                                                num_preds=self.num_object_preds,
                                                name=f'BoxPredictionHead_{i}')

            self.BoxBlocks.append(BoxPredictionHead)

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
        focused_training_layer = None # 'None' or number of a specific decoder layer

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

        # Ecnoder / Decoder blocks (transformers)
        loss = 0.0
        cat_loss = 0.0
        att_loss = 0.0
        box_loss = 0.0
        exist_loss = 0.0
        iou_metric = 0.0

        for i in range(self.num_decoder_blocks):

            if i==0:
                enc_feat_shape = tf.shape(encoder_features)  # [batch, rows, columns, features_dim]
            else:  # i >= 1
                encoder_features = tf.reshape(encoder_features, enc_feat_shape)

            encoder_features, positional_encoding = \
                self.EncoderTransformerBlocks[i]([encoder_features], training=training)  # [batch, encoder_dim]
            
            # Initialize / reshape variables for decoder
            encoder_features, decoder_features, encoder_key, decoder_positional \
                = self.DecoderPrep([encoder_features, positional_encoding], training=training)
            
            decoder_features = self.DecoderBlocks[i](
                [encoder_features, decoder_features, encoder_key, decoder_positional],
                training=training)

            cat_preds_i = self.CategoryBlocks[i]([decoder_features], training=training)  # [batch, objs, num_cats]
            attribute_preds_i = self.AttributeBlocks[i]([decoder_features], training=training)  # [batch, objs, num_attributes]
            box_coord_preds_i = self.BoxBlocks[i]([decoder_features], training=training)  # [batch, objs, 4 coords]                

            # initialize preds tensor after first call (so that output shape is defined)
            if i==0:
                cat_preds = cat_preds_i
                attribute_preds = attribute_preds_i
                box_coord_preds = box_coord_preds_i

            cat_preds += cat_preds_i
            attribute_preds += attribute_preds_i
            box_coord_preds += box_coord_preds_i
            y_pred = [cat_preds, attribute_preds, box_coord_preds]

            if training and (i==focused_training_layer or focused_training_layer==None):
                # compute losses and metrics
                losses_i, metrics_i = self.loss_fn([y_true, y_pred])

                # accumulate losses
                loss_i, cat_loss_i, att_loss_i, box_loss_i, exist_loss_i = losses_i

                loss = loss + loss_i
                cat_loss = cat_loss + cat_loss_i
                att_loss = att_loss + att_loss_i
                box_loss = box_loss + box_loss_i
                exist_loss = exist_loss + exist_loss_i

            if i==focused_training_layer:
                break

        if training:
            # record loss
            self.add_loss(loss)  ###### Critical component! Do not alter this!!

            # report losses (from final step)
            self.add_metric(cat_loss, 'Category_Loss')
            self.add_metric(att_loss, 'Attribute_Loss')
            self.add_metric(box_loss, 'Box_Loss')
            self.add_metric(exist_loss, 'Existence_Loss')

            # report metrics (from final step)
            iou_metric_i = metrics_i[0]
            self.add_metric(iou_metric_i, 'IOU')

            return y_pred

        # Translate results into text descriptions
        category, attributes = self.InverseTokenization([cat_preds, attribute_preds], training=training)

        return category, attributes, box_coord_preds

    def test_step(self, inputs):
        return self.train_step(inputs)

    def citation(self):
        print('''This is an original adaptation of the DETR model for object 
        detection and fine-grained classification. This version modifies the 
        original model with a "boosted-ensemble" style prediction and additional 
        prediction head.'''

        '''DETR is published under the Apache License 2.0. This model was independently
        coded in Tensorflow based on the paper "End-to-end Object Detection with Transformers"
        by Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov
        and Sergey Zagoruyko, available at
        https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers.''')
