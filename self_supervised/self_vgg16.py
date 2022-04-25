from nets.vgg16 import VGG16
from tensorflow.keras.layers import BatchNormalization, Input, Flatten, Dense, Lambda, Activation, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf


def euclidean_distance(vects):  # 欧氏距离
    """Find the Euclidean distance between two vectors.
    Arguments:
        vects: List containing two tensors of same length.
    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    dic = tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))
    return dic

def cosine_distance(vects):
    """
    consine相似度：用两个向量的夹角判断两个向量的相似度，夹角越小，相似度越高，得到的consine相似度数值越大
    数值范围[-1,1],数值越大越相似。
    :param tensor1:
    :param tensor2:
    :return:
    """
    tensor1, tensor2=vects
    # print(tensor1)

    # 求模长
    tensor1_norm = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(tensor1), axis=1, keepdims=True))
    tensor2_norm = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(tensor2), axis=1, keepdims=True))
    # print(tensor1_norm)

    # 内积
    tensor1_tensor2 = tf.math.reduce_sum(tf.math.multiply(tensor1, tensor2), axis=1, keepdims=True)
    # cosin = tensor1_tensor2 / (tensor1_norm * tensor2_norm)
    t1t2=tf.math.multiply(tensor1_norm , tensor2_norm)
    cosin=tf.math.divide(tensor1_tensor2,t1t2)
    # print(cosin)

    return 1-cosin

def vgg16_model(shape=(1024, 576, 3)):
    input = Input(shape)
    feat1, feat2, feat3, feat4, feat5 = VGG16(input)
    return Model(input, feat5)

def self_sup_vgg16(shape=(1024, 576, 3)):
    # vgg16主干网络
    input = Input(shape)
    feat1, feat2, feat3, feat4, feat5 = VGG16(input)
    x = Flatten()(feat5)
    # mlp
    x = Dense(512, name='mlp512')(x)
    x=BatchNormalization(name='mlpBN1')(x)
    # x=Dropout(0.2)(x)
    x=Activation('relu',name='mlpRelu')(x)
    x = Dense(256, name='mlp256')(x)
    # x = BatchNormalization(name='mlpBN2')(x)
    # vgg输出
    vgg = Model(input, x)
    # vgg.summary()

    # 对比学习网络
    input_1 = Input(shape, name='self_input1')
    input_2 = Input(shape, name='self_input2')
    tower_1 = vgg(input_1)
    tower_2 = vgg(input_2)

    # merge_layer = Lambda(euclidean_distance)([tower_1, tower_2])
    merge_layer = Lambda(cosine_distance)([tower_1, tower_2])
    normal_layer = tf.keras.layers.BatchNormalization(name='self_output')(merge_layer)
    siamese = Model(inputs=[input_1, input_2], outputs=normal_layer)
    siamese.summary()

    return siamese

def self_sup_vgg16_1(shape=(1024, 576, 3)):

    # 对比学习网络
    input_1 = Input(shape, name='self_input1')
    input_2 = Input(shape, name='self_input2')
    #实例化vgg层
    vgg=vgg16_model(shape)
    tower_1 = vgg(input_1)
    tower_2 = vgg(input_2)

    # mlp
    x1 = Flatten()(tower_1)
    x1 = Dense(512)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Dense(256)(x1)

    # mlp
    x2 = Flatten()(tower_2)
    x2 = Dense(512)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Dense(256)(x2)


    # merge_layer = Lambda(euclidean_distance)([tower_1, tower_2])
    merge_layer = Lambda(cosine_distance)([x1, x2])
    normal_layer = tf.keras.layers.BatchNormalization(name='self_output')(merge_layer)
    siamese = Model(inputs=[input_1, input_2], outputs=normal_layer)
    siamese.summary()

    return siamese

if __name__ == '__main__':
    self_sup_vgg16()
