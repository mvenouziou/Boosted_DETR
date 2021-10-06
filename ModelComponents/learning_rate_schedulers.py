# imports
import tensorflow as tf


"""
lr = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=.001, 
                                                           first_decay_steps=4000)
"""

# Modified "Attention is All You Need" learning scheduler (to become cyclic)
class LRScheduleAIAYN(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, scale_factor=1.0, warmup_steps=4000):  # defaults reflect paper's values
        # cast dtypes
        self.warmup_steps = tf.constant(warmup_steps, dtype=tf.float32)
        dim = tf.constant(352, dtype=tf.float32)
        self.scale_factor = tf.constant(scale_factor, dtype=tf.float32)
        
        self.scale = self.scale_factor * tf.math.pow(dim, -1.5)

    def get_config(self):
        config = {'scale_factor': self.scale_factor, 'warmup_steps': self.warmup_steps}
        return config 
        
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        crit = self.warmup_steps

        def false_fn(step):
            adj_step = (step - crit) % (2.0*crit) + crit
            return tf.math.pow(adj_step, -.5)

        val = tf.cond(tf.math.less(step, crit),
                      lambda: step * tf.math.pow(crit, -1.5),  # linear increase
                      lambda: false_fn(step)  # decay
                      )
        return self.scale * val

    def display_graph(self):
        print('Learning Rate Schedule')
        plt.plot([i for i in range(1, 16000)], 
                 [self(i) for i in range(1, 16000)])

