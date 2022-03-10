import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

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

def euclidean_distance(vects): # 欧氏距离
    """Find the Euclidean distance between two vectors.
    Arguments:
        vects: List containing two tensors of same length.
    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    dic=tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))
    print(dic)
    return dic

