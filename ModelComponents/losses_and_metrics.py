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


class CostArray(tf.keras.layers.Layer):
    """ arranges tensors for broadcasting pairwise f(x, y) operations """

    def __init__(self, **kwargs):
        super().__init__(name='CostArray', dtype='float32', **kwargs)
  
    def call(self, y_true, y_pred, func):  
        y_true = tf.expand_dims(y_true, axis=-2)
        y_pred = tf.expand_dims(y_pred, axis=-3) * tf.ones_like(y_true)
        return func(y_true, y_pred)


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


'''
####### In Progress. Attempting to make a TPU friendly alternative to 
scipy bipartite matching. (No non-tf ops and compat with @tf.function, XLA)

class GreedyMatchingAssignment(tf.keras.layers.Layer):
    """ TODO: TPU compliant Bipartite Assigment alternative. 
    Creates mask indicating the (approximated) min cost matching assignments. """

    def __init__(self, name='GreedyMatchingAssignment', **kwargs):
        super().__init__(name=name, dtype='float32', **kwargs)

    def build(self, input_shape):
        self.batch_size = input_shape[0]  # need fixed size for XLA?
        self.padded_obj = input_shape[1]  # need fixed size for XLA?
        self.num_preds = input_shape[2]  # need fixed size for XLA?

    #@tf.function()
    def call(self, cost_array, num_objects): 
        batch_size = tf.shape(cost_array)[-3]
        padded_obj = tf.shape(cost_array)[-2]
        num_preds = tf.shape(cost_array)[-1]
        
        assignments = []
        columns = []
        for i in range(batch_size):
            cost = cost_array[i, ...]
            num = num_objects[i]

            assignments_i, columns_assigned_i = self.single_element_call(cost, num)

            assignments.append(assignments_i)
            columns.append(columns_assigned_i)
        
        return  assignments, columns

    def selected_preds_mask(self, selected_cols, num_preds):
        a = tf.expand_dims(selected_cols, 1)
        b = tf.expand_dims(tf.range(num_preds), 0)  
        return tf.math.reduce_any(tf.math.equal(a, tf.cast(b, a.dtype)), axis=0)

    def single_element_call(self, cost_array2D, num_objects):  
        """
        Matching corresponding to batch size 1
        """
        padded_obj = tf.shape(cost_array2D)[-2]
        num_preds = tf.shape(cost_array2D)[-1]

        # initialize container with dummy values
        dummy_val = -1
        selected_cols = tf.fill(value=dummy_val, dims=[num_objects])
        matches = tf.TensorArray(dtype=tf.int32, size=num_objects, dynamic_size=False)

        # shuffle object selection order
        shuffled_objects = tf.random.shuffle(tf.range(num_objects))

        for j in range(num_objects):

            # select object
            object_num = shuffled_objects[j]

            # get sorted costs
            costs = cost_array2D[object_num, :]
            choices = tf.argsort(costs, direction='ASCENDING')

            # find lowest cost among cols not already selected
            indx = 0
            while tf.math.reduce_any(tf.math.equal(choices[indx], selected_cols)):
                found = tf.math.reduce_any(tf.math.equal(choices[indx], selected_cols))
                indx = indx + 1

            # update selected columns
            hot_loc = tf.one_hot(j, depth=num_objects)
            selected_cols = selected_cols + (choices[indx] - dummy_val) * tf.cast(hot_loc, tf.int32)

            # update matches
            matches = matches.write(j, tf.cast([object_num, choices[indx]], matches.dtype))
        
        # get final assignments
        assignments = matches.stack()
        columns_assigned = self.selected_preds_mask(selected_cols, num_preds)

        return assignments, columns_assigned
'''
