# imports
import tensorflow as tf
import numpy as np
from scipy.optimize import linear_sum_assignment
import tensorflow_addons as tfa


# basic loss components
L2_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
GIOU_loss = lambda x, y: tfa.losses.giou_loss(x, y, mode='giou')
GIOU_Metric = lambda x, y: 1.0 - GIOU_loss(x, y)
IOU_loss = lambda x, y: tfa.losses.giou_loss(x, y, mode='iou')
IOU_Metric = lambda x, y: 1.0 - IOU_loss(x, y)
SigmoidFocalCrossEntropy = tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.NONE)
BinaryCrossentropy = tf.keras.losses.BinaryCrossentropy(
        label_smoothing=0, reduction=tf.keras.losses.Reduction.NONE)
CategoricalCrossentropy = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=.1, reduction=tf.keras.losses.Reduction.NONE)


def safe_clip(probability):
    return tf.clip_by_value(probability, clip_value_min=.001, clip_value_max=.999)

def apply_mask(mask, tensor):
    # ensures matching dtypes while preserving tensor dtype
    return tf.cast(mask, tensor.dtype) * tensor

def ExistLoss(y_true, y_pred):
    # assigns error based on prediction of 'None' class
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return BinaryCrossentropy(y_true, safe_clip(y_pred))

def CategoryMatchLoss(y_true, y_pred):
    # crossentropy without the logarithm. y_true acts as a mask
    # used in bipartite matching cost
    return tf.reduce_sum((1.0 - y_pred) * y_true, axis=-1)

def CategoryLoss(y_true, y_pred):
    # this is a binary loss on just the true category. 
    # (note: y_true is used as a mask on y_pred)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return BinaryCrossentropy(y_true, safe_clip(y_pred)*y_true)

def AttributeLoss(y_true, y_pred):
    if tf.math.not_equal(tf.shape(y_true)[-1], 1):
        y_true = tf.expand_dims(y_true, axis=-1)
    if tf.math.not_equal(tf.shape(y_pred)[-1], 1):
        y_pred = tf.expand_dims(y_pred, axis=-1)
    return SigmoidFocalCrossEntropy(y_true, safe_clip(y_pred))

def coco_to_tf(box):
    """ converts from label data's COCO [xmin, ymin, width, height] format
    to Tensorflow [y_min, x_min, y_max, x_max] format"""
    xmin = box[..., 0:1]
    ymin = box[..., 1:2]
    width = box[..., 2:3]
    height = box[..., 3:4]
    return tf.concat([ymin, xmin, ymin + height, xmin + width], axis=-1)

def BoxLoss(y_true, y_pred, giou_weight=2.0, l2_weight=5.0):
    # convert from COCO to tensorflow format and apply losses
    y_true_tf = coco_to_tf(y_true)
    y_pred_tf = coco_to_tf(y_pred)
    return giou_weight * GIOU_loss(y_true_tf, y_pred_tf) + l2_weight * L2_loss(10.0*y_true_tf, 10.0*y_pred_tf)


class MatchingLoss(tf.keras.layers.Layer):
    def __init__(self, name='MatchingLoss', 
                 category_weight=1.0, box_weight=1.0, attribute_weight=1.0,
                 exist_weight=1.0, **kwargs):
        super().__init__(name=name, dtype=tf.float32, **kwargs)

        # weights
        self.category_weight = tf.cast(category_weight, tf.float32)
        self.box_weight = tf.cast(box_weight, tf.float32)
        self.attribute_weight = tf.cast(attribute_weight, tf.float32)
        self.exist_weight = tf.cast(exist_weight, tf.float32)

        # layers
        self.MatchingMask = MatchingMask()
        self.CostArray = CostArray()
        self.apply_mask = apply_mask
        self.safe_clip = safe_clip

        # losses
        self.CategoryLoss = CategoryLoss
        self.ExistLoss = ExistLoss
        self.AttributeLoss = AttributeLoss
        self.BoxLoss = BoxLoss

        # metrics
        self.MatchingMetric = MatchingMetric()

    def call(self, inputs):

        y_true, y_pred = inputs

        category, attribute, bbox, num_objects = y_true
        cat_preds, attribute_preds, box_preds = y_pred

        # get pairwise costs
        category_cost = self.CostArray(category, cat_preds, self.CategoryLoss)
        attribute_cost = self.CostArray(attribute, attribute_preds, self.AttributeLoss)
        box_cost = self.CostArray(bbox, box_preds, self.BoxLoss)  

        # apply weights
        category_cost = self.category_weight * category_cost
        attribute_cost = self.attribute_weight * attribute_cost
        box_cost = self.box_weight * box_cost

        # get assignment mask    
        assignment_mask, assigned_predictions = self.MatchingMask(
                                    [category_cost + box_cost, num_objects])

        # apply cost mask        
        category_cost = self.apply_mask(assignment_mask, category_cost)
        attribute_cost = tf.reduce_mean(attribute_cost, axis=[3])
        attribute_cost = self.apply_mask(assignment_mask, attribute_cost)
        box_cost = self.apply_mask(assignment_mask, box_cost)

        # get object existence cost
        # items not assigned should have high prob assigned to class 0
        exist_cost = self.exist_weight * self.ExistLoss(1.0 - assigned_predictions, 
                                                        cat_preds[ ..., 0:1])

        # get mean total
        # mean over actual objects (masked costs only 'num_objects' nonzero entries )
        total_num_objects = 1.0 + tf.cast(tf.reduce_sum(num_objects), tf.float32)
        num_preds_per_batch = 1.0 + tf.cast(tf.shape(cat_preds)[1], tf.float32)

        category_cost = tf.reduce_sum(category_cost, axis=[-2, -1]) / total_num_objects
        attribute_cost = tf.reduce_sum(attribute_cost, axis=[-2, -1]) / total_num_objects
        box_cost = tf.reduce_sum(box_cost, axis=[-2, -1]) / total_num_objects       
        exist_cost = tf.reduce_mean(exist_cost, axis=-1) / num_preds_per_batch  # downweight for class imbalance

        # total cost
        total_loss = category_cost + attribute_cost + box_cost + exist_cost
        losses = [total_loss, category_cost, attribute_cost, box_cost, exist_cost]

        # compute metrics
        masked_iou = self.MatchingMetric([y_true, y_pred], assignment_mask)
        masked_iou = tf.reduce_sum(masked_iou, axis=[1,2]) / total_num_objects

        metrics = [masked_iou]
        return losses, metrics


class MatchingMetric(tf.keras.layers.Layer):
    def __init__(self, name='MatchingMetric', **kwargs):
        super().__init__(name=name, dtype=tf.float32, **kwargs)

        # layers
        self.MatchingMask = MatchingMask()
        self.CostArray = CostArray()
        self.apply_mask = apply_mask

        # metrics
        self.IOU_Metric = IOU_Metric

    def call(self, inputs, assignment_mask=None):
        y_true, y_pred = inputs

        category, attribute, bbox, num_objects = y_true
        cat_preds, attribute_preds, box_preds = y_pred

        # get matching assignment mask
        if assignment_mask is None:
            assignment_mask, assigned_predictions = self.MatchingMask(
                            [(category, bbox, num_objects), (cat_preds, box_preds)])

        # get masked, pairwise metric
        masked_iou = self.CostArray(bbox, box_preds, self.IOU_Metric)
        masked_iou = self.apply_mask(assignment_mask, masked_iou)

        metrics = [masked_iou]
        return metrics


class MatchingMask(tf.keras.layers.Layer):
    def __init__(self, name='MatchingMask', **kwargs):
        super().__init__(name=name, dtype=tf.float32, **kwargs)

        # layers
        self.MatchingAssignment = MatchingAssignment()

    def call(self, inputs):
        matching_costs, num_objects = inputs

        # get matching assignment mask
        assignment_mask = self.MatchingAssignment(matching_costs, num_objects)

        # get indicator showing which predictions were assigned an object
        assigned_predictions = tf.reduce_max(assignment_mask, axis=-2)  # = 0 or 1 at each position
        assigned_predictions = tf.expand_dims(assigned_predictions, axis=-1)  # [batch, num_preds, 1]

        return assignment_mask, assigned_predictions


class CostArray(tf.keras.layers.Layer):
    """ arranges tensors for broadcasting and
    computes pairwise f(x, y) values """

    def __init__(self, **kwargs):
        super().__init__(name='CostArray', dtype=tf.float32, **kwargs)

    def call(self, y_true, y_pred, func):
        y_true = tf.expand_dims(y_true, axis=-2)
        y_pred = tf.expand_dims(y_pred, axis=-3)# * tf.ones_like(y_true)
        return func(y_true, y_pred)


class MatchingAssignment(tf.keras.layers.Layer):
    """ Bipartite Assigment. Creates mask indicating min cost matching assignments. """

    def __init__(self, name='MatchingAssignment', **kwargs):
        super().__init__(name=name, dtype=tf.float32, **kwargs)  # override mixed precision

    def scipy_linear_assignment_mask(self, cost_array, num_objects):
        batch_size = cost_array.shape[0]
        num_padded_obj = cost_array.shape[1]
        masks = np.zeros_like(cost_array)  # numpy used for item assignments below

        # get assignments
        for i in range(batch_size):
            num_objects_i = num_objects[i][0]
            assignments_i = linear_sum_assignment(cost_array[i, :num_objects_i, :])
            masks[i, ...][assignments_i] = 1.0

        return masks

    def call(self, cost_array, num_objects):
        num_objects = tf.reshape(num_objects, [-1,1])  # needed for compat in model.fit()
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
