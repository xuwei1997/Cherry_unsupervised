# 训练对比学习网络
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from self_vgg16 import self_sup_vgg16
from self_vgg16_2 import self_sup_vgg16
from dataloader_2img import minibatches
from pathos.multiprocessing import ProcessingPool as Pool
import os
from utils.callbacks import ExponentDecayScheduler, LossHistory
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import datetime

def plt_metric(history, metric, title, has_valid=True):
    """Plots the given 'metric' from 'history'.
    Arguments:
        history: history attribute of History object returned from Model.fit.
        metric: Metric to plot, a string value present as key in 'history'.
        title: A string to be used as title of plot.
        has_valid: Boolean, true if valid data was passed to Model.fit else false.
    Returns:
        None.
    """
    plt.plot(history[metric])
    if has_valid:
        plt.plot(history["val_" + metric])
        plt.legend(["train", "validation"], loc="upper left")
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.show()


def loss(T=0.1):  # 对比学习loss
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
        sce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=(y_logits / T))
        sce_loss = tf.math.reduce_mean(sce)
        # print(sce_loss)
        return sce_loss

    return contrastive_loss


def re_label(y):
    a0, a1, a2, a3 = 1, 0.3, 0.8, 0
    if y == 0:
        return a0
    elif y == 1:
        return a1
    elif y == 2:
        return a2
    elif y == 3:
        return a3


if __name__ == '__main__':
    # 动态占用显存
    tf.config.set_soft_device_placement(True)
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # 设置随机性
    # seed = 888
    # np.random.seed(seed)  # seed是一个固定的整数即可
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # tf.random.set_seed(seed)  # tensorflow2.0版本的设置，较早版本的设置方式不同，可以自查

    # 读取数据
    X = np.load('/tmp/pycharm_project_116/data_processing/x_out_inx.npy')
    Y = np.load('/tmp/pycharm_project_116/data_processing/y_out_inx.npy')
    Y = list(map(re_label, Y))
    # print(X)
    print(len(Y))
    x_train, x_val, x_test = X[:157486], X[157486:179984], X[179984:]
    y_train, y_val, y_test = Y[:157486], Y[157486:179984], Y[179984:]
    # x_train, x_val, x_test = X[:1000], X[1000:2000], X[1500:2000]
    # y_train, y_val, y_test = Y[:1000], Y[1000:2000], Y[1500:2000]

    # -------------------------------------------------------------------------------#
        #   训练参数的设置
    #   logging表示tensorboard的保存地址
    #   checkpoint用于设置权值保存的细节，period用于修改多少epoch保存一次
    #   reduce_lr用于设置学习率下降的方式
    #   early_stopping用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
    # -------------------------------------------------------------------------------#
    logging = TensorBoard(log_dir='self_logs/')
    # checkpoint = ModelCheckpoint('self_logs/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
    #                              monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    loss_history = LossHistory('self_logs/')
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, verbose=1)

    times = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    time_str = 'h5_' + str(times) + '/'
    os.mkdir('self_logs/' + time_str)
    checkpoint = ModelCheckpoint('self_logs/' + time_str + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)


    b_size = 30
    b_evaluate_size = 30
    epochs = 10
    T = 0.5

    # 训练
    p = Pool()
    siamese = self_sup_vgg16()
    siamese.compile(loss=loss(T=T), optimizer="Adam", metrics=["mse"])
    # siamese.compile(loss='binary_crossentropy', optimizer="sgd", metrics=["mse"])

    siamese.load_weights("/tmp/pycharm_project_116/model_data/unet_vgg_voc.h5", by_name=True, skip_mismatch=True)

    # siamese.summary()

    history = siamese.fit_generator(minibatches(x_train, y_train, batch_size=b_size, p=p),
                                    steps_per_epoch=len(y_train) // b_size,
                                    validation_data=minibatches(x_val, y_val, batch_size=b_size, p=p),
                                    validation_steps=len(y_val) // b_size,
                                    epochs=epochs,
                                    callbacks=[checkpoint, logging, loss_history, early_stopping]
                                    )
    # print(history)

    # results = siamese.evaluate([x_test_1, x_test_2], labels_test)
    results = siamese.evaluate_generator(minibatches(x_test, y_test, batch_size=b_evaluate_size, p=p),
                                         steps=len(y_test) // b_evaluate_size)
    print("test loss, test acc:", results)
