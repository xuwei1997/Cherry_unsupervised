#训练对比学习网络
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from self_vgg16 import self_sup_vgg16

def loss(T=0.1): #对比学习loss
    def contrastive_loss(y_true, y_logits):
        """Calculates the constrastive loss.
           Arguments:
               y_true: List of labels, each label is of type float32.
               y_logits: List of predictions of same length as of y_true,
                       each label is of type float32.
           Returns:
               A tensor containing constrastive loss as floating point value.
           """
        # sce=tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_logits)
        sce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=(y_logits/T))
        sce_loss=tf.math.reduce_mean(sce)
        return sce_loss

    return contrastive_loss



if __name__ == '__main__':
    model=self_sup_vgg16()
    model.summary()