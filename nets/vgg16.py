# from tensorflow.keras import layers
# from tensorflow.keras.initializers import random_normal
# from tensorflow.keras.layers import Input, Dense, GlobalMaxPooling2D
# from tensorflow.keras.models import Model
# # import keras
# import tensorflow as tf

from keras import layers
from keras.initializers import random_normal
from keras.layers import Input, Dense, GlobalMaxPooling2D
from keras.models import Model
# import keras
import tensorflow as tf

def VGG16(img_input):
    # Block 1
    # 512,512,3 -> 512,512,64
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = random_normal(stddev=0.02),
                      name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = random_normal(stddev=0.02),
                      name='block1_conv2')(x)
    feat1 = x
    # 512,512,64 -> 256,256,64
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    # 256,256,64 -> 256,256,128
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = random_normal(stddev=0.02),
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = random_normal(stddev=0.02),
                      name='block2_conv2')(x)
    feat2 = x
    # 256,256,128 -> 128,128,128
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)


    # Block 3
    # 128,128,128 -> 128,128,256
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = random_normal(stddev=0.02),
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = random_normal(stddev=0.02),
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = random_normal(stddev=0.02),
                      name='block3_conv3')(x)
    feat3 = x
    # 128,128,256 -> 64,64,256
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    # 64,64,256 -> 64,64,512
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = random_normal(stddev=0.02),
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = random_normal(stddev=0.02),
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = random_normal(stddev=0.02),
                      name='block4_conv3')(x)
    feat4 = x
    # 64,64,512 -> 32,32,512
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    # 32,32,512 -> 32,32,512
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = random_normal(stddev=0.02),
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = random_normal(stddev=0.02),
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = random_normal(stddev=0.02),
                      name='block5_conv3')(x)
    feat5 = x
    return feat1, feat2, feat3, feat4, feat5

def VGG16_2(img_input):
    # Block 1
    # 512,512,3 -> 512,512,64
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = random_normal(stddev=0.02),
                      name='block1_conv1_2')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = random_normal(stddev=0.02),
                      name='block1_conv2_2')(x)
    feat1 = x
    # 512,512,64 -> 256,256,64
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool_2')(x)

    # Block 2
    # 256,256,64 -> 256,256,128
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = random_normal(stddev=0.02),
                      name='block2_conv1_2')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = random_normal(stddev=0.02),
                      name='block2_conv2_2')(x)
    feat2 = x
    # 256,256,128 -> 128,128,128
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool_2')(x)


    # Block 3
    # 128,128,128 -> 128,128,256
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = random_normal(stddev=0.02),
                      name='block3_conv1_2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = random_normal(stddev=0.02),
                      name='block3_conv2_2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = random_normal(stddev=0.02),
                      name='block3_conv3_2')(x)
    feat3 = x
    # 128,128,256 -> 64,64,256
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool_2')(x)

    # Block 4
    # 64,64,256 -> 64,64,512
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = random_normal(stddev=0.02),
                      name='block4_conv1_2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = random_normal(stddev=0.02),
                      name='block4_conv2_2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = random_normal(stddev=0.02),
                      name='block4_conv3_2')(x)
    feat4 = x
    # 64,64,512 -> 32,32,512
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool_2')(x)

    # Block 5
    # 32,32,512 -> 32,32,512
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = random_normal(stddev=0.02),
                      name='block5_conv1_2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = random_normal(stddev=0.02),
                      name='block5_conv2_2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = random_normal(stddev=0.02),
                      name='block5_conv3_2')(x)
    feat5 = x
    return feat1, feat2, feat3, feat4, feat5

if __name__ == '__main__':
    inputs = Input(shape=(270, 480, 3))
    feat1, feat2, feat3, feat4, feat5=VGG16(inputs)
    model = Model(inputs=inputs, outputs=feat5)
    model.summary()

    dot_img_file = 'vgg16.png'
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=False)


