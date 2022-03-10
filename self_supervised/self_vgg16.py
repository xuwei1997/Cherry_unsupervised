from nets.vgg16 import VGG16
from tensorflow.keras.layers import BatchNormalization,Input,Flatten,Dense,Lambda
from tensorflow.keras.models import Model
import tensorflow as tf

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
    return dic

def self_sup_vgg16(shape=(1024,576,3)):
    # vgg16主干网络
    input=Input(shape)
    feat1, feat2, feat3, feat4, feat5=VGG16(input)
    x=Flatten()(feat5)
    #mlp
    x=Dense(512,activation='relu',name='mlp2048')(x)
    x=BatchNormalization(name='mlpBN1')(x)
    x=Dense(256,activation='relu',name='mlp256')(x)
    x = BatchNormalization(name='mlpBN2')(x)
    #vgg输出
    vgg=Model(input,x)
    vgg.summary()


    # 对比学习网络
    input_1 = Input(shape,name='self_input1')
    input_2 = Input(shape,name='self_input2')
    tower_1 = vgg(input_1)
    tower_2 = vgg(input_2)
    merge_layer = Lambda(euclidean_distance)([tower_1, tower_2])
    normal_layer = tf.keras.layers.BatchNormalization(name='output')(merge_layer)
    siamese = Model(inputs=[input_1, input_2], outputs=normal_layer)
    siamese.summary()

    return siamese





if __name__ == '__main__':
    self_sup_vgg16((28, 28, 1))