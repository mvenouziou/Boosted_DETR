# imports
import tensorflow as tf
import numpy as np
from scipy.optimize import linear_sum_assignment
import tensorflow_addons as tfa


class MatchingLoss(tf.keras.layers.Layer):
    def __init__(self, name='MatchingLoss', **kwargs):
        super().__init__(name=name, dtype='float32', **kwargs)

        # layers
        self.MatchingAssignment = MatchingAssignment()
        self.CostArray = CostArray()
        categorical_crossentropy = tf.keras.losses.CategoricalCrossentropy(
                                    reduction=tf.keras.losses.Reduction.NONE)

        # categories  
        self.CategoryLoss = categorical_crossentropy

        def cat_match_loss_fn(y_true, y_pred):
            # crossentropy without the logarithm. y_true acts as a mask
            return tf.reduce_sum((1.0 - y_pred) * y_true, axis=-1)
        
        self.CategoryMatchLoss = cat_match_loss_fn

        def cat_loss_fn(y_true, y_pred):
            return -1.0 * tf.reduce_sum(tf.math.log(y_pred + .00001) * y_true, axis=-1)
        
        self.CategoryLoss = cat_loss_fn

        def exist_loss_fn(y_matched, y_pred):
            # assigns error based on prediction of 'None' class
            y_pred_match = 1.0 - y_pred[..., 0:1]
            return categorical_crossentropy(y_matched, y_pred_match)

        self.ExistLoss = exist_loss_fn

        # attributes 
        self.AttributeLoss = tfa.losses.SigmoidFocalCrossEntropy(
                                        reduction=tf.keras.losses.Reduction.NONE)
        # boxes
        giou_fn = tfa.losses.GIoULoss(reduction=tf.keras.losses.Reduction.NONE)
        L1_fn = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
        def box_loss(y_true, y_pred):
            return 2.0 * giou_fn(y_true, y_pred) + 5.0 * L1_fn(y_true, y_pred)
        
        self.BoxLoss = box_loss

        # metrics
        self.IOU_Metric = tfa.losses.GIoULoss(mode='iou', reduction=tf.keras.losses.Reduction.NONE)

    
    def call(self, inputs):  
        y_true, y_pred = inputs
        category, attribute, bbox, num_objects = y_true
        cat_preds, attribute_preds, box_preds = y_pred

        # collect info
        batch_size = tf.cast(tf.shape(cat_preds)[0], tf.float32)
        total_num_preds = batch_size * tf.cast(tf.shape(cat_preds)[1], tf.float32)
        total_num_obj = tf.cast(tf.reduce_sum(num_objects), tf.float32)

        # get pairwise costs
        category_cost_match = self.CostArray(category, cat_preds, self.CategoryMatchLoss)
        category_cost = self.CostArray(category, cat_preds, self.CategoryLoss)
        attribute_cost = self.CostArray(attribute, attribute_preds, self.AttributeLoss)
        box_cost = self.CostArray(bbox, box_preds, self.BoxLoss)       

        # get pairwise metrics
        iou_metric = self.CostArray(bbox, box_preds, self.IOU_Metric)  

        # get matching assignment mask
        assignment_mask = self.MatchingAssignment(category_cost_match + box_cost, 
                                                  num_objects)
        # find objects that were not assigned
        assigned_predictions = tf.reduce_max(assignment_mask, axis=-2)
        assigned_predictions = tf.expand_dims(assigned_predictions, axis=-1)

        # get mean masked costs
        category_cost = tf.reduce_sum(assignment_mask * category_cost) / total_num_obj
        attribute_cost = tf.reduce_sum(assignment_mask * attribute_cost) / total_num_obj
        box_cost = tf.reduce_sum(assignment_mask * box_cost) / total_num_obj
        exist_loss = tf.reduce_sum(self.ExistLoss(assigned_predictions, cat_preds[ ..., 0:1])) / total_num_preds

        total_loss = category_cost + attribute_cost + box_cost + exist_loss
        losses = [total_loss, category_cost, attribute_cost, box_cost, exist_loss]
        
        # get masked metrics
        masked_iou = assignment_mask * iou_metric

        iou_metric = tf.reduce_sum(masked_iou)  / total_num_obj
        num_correct_at_50 = tf.math.count_nonzero(tf.math.greater_equal(masked_iou, .50))  
        mAP_50_95 = 0.0
        r_vals = list(range(50, 100, 5))
        for r in r_vals:
            r = r / 100.0
            mAP_50_95 += tf.cast(tf.math.count_nonzero(tf.math.greater_equal(masked_iou, r)), 
                                 tf.float32)

        num_predicted = tf.math.count_nonzero(tf.math.less(cat_preds[:, 1, ...], .50))  
        
        mAP50 = tf.cast(num_correct_at_50, tf.float32) / tf.cast(num_predicted, tf.float32)
        mAP_50_95 = mAP_50_95 / tf.cast(num_predicted*len(r_vals), tf.float32)
        
        metrics = [iou_metric, mAP50, mAP_50_95] 
        return losses, metrics


class MatchingAssignment(tf.keras.layers.Layer):
    """ Bipartite Assigment. Creates mask indicating min cost matching assignments. """

    def __init__(self, name='MatchingAssignment', **kwargs):
        super().__init__(name=name, dtype='float32', **kwargs)

    def scipy_linear_assignment_mask(self, cost_array, num_objects):  
        batch_size = cost_array.shape[0]
        num_padded_obj = cost_array.shape[1]
        masks = np.zeros_like(cost_array)

        # get assignments
        for i in range(batch_size):
            num_objects_i = num_objects[i]
            assignments_i = linear_sum_assignment(cost_array[i, :num_objects_i, :])
            masks[i, ...][assignments_i] = 1.0
        
        return masks

    def call(self, cost_array, num_objects):  
        masks = tf.numpy_function(func=self.scipy_linear_assignment_mask, 
                                   inp=[cost_array, num_objects], Tout=cost_array.dtype)
        return masks


class CostArray(tf.keras.layers.Layer):
    """ arranges tensors for broadcasting pairwise f(x, y) operations """

    def __init__(self, **kwargs):
        super().__init__(name='CostArray', dtype='float32', **kwargs)
  
    def call(self, y_true, y_pred, func):  
        y_true = tf.expand_dims(y_true, axis=-2)
        y_pred = tf.expand_dims(y_pred, axis=-3) * tf.ones_like(y_true)
        return func(y_true, y_pred)