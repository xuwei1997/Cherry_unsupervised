#训练对比学习网络,mnist
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from self_vgg16 import self_sup_vgg16
from tensorflow import keras

"""
## Create pairs of images
We will train the model to differentiate between digits of different classes. For
example, digit `0` needs to be differentiated from the rest of the
digits (`1` through `9`), digit `1` - from `0` and `2` through `9`, and so on.
To carry this out, we will select N random images from class A (for example,
for digit `0`) and pair them with N random images from another class B
(for example, for digit `1`). Then, we can repeat this process for all classes
of digits (until digit `9`). Once we have paired digit `0` with other digits,
we can repeat this process for the remaining classes for the rest of the digits
(from `1` until `9`).
"""


def make_pairs(x, y):
    """Creates a tuple containing image pairs with corresponding label.
    Arguments:
        x: List containing images, each index in this list corresponds to one image.
        y: List containing labels, each label with datatype of `int`.
    Returns:
        Tuple containing two numpy arrays as (pairs_of_samples, labels),
        where pairs_of_samples' shape is (2len(x), 2,n_features_dims) and
        labels are a binary array of shape (2len(x)).
    """

    num_classes = max(y) + 1
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

    pairs = []
    labels = []

    for idx1 in range(len(x)):
        # add a matching example
        x1 = x[idx1]
        label1 = y[idx1]
        idx2 = random.choice(digit_indices[label1])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [1]

        # add a non-matching example
        label2 = random.randint(0, num_classes - 1)
        while label2 == label1:
            label2 = random.randint(0, num_classes - 1)

        idx2 = random.choice(digit_indices[label2])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [0]

    return np.array(pairs), np.array(labels).astype("float32")

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

def visualize(pairs, labels, to_show=6, num_col=3, predictions=None, test=False):
    """Creates a plot of pairs and labels, and prediction if it's test dataset.
    Arguments:
        pairs: Numpy Array, of pairs to visualize, having shape
               (Number of pairs, 2, 28, 28).
        to_show: Int, number of examples to visualize (default is 6)
                `to_show` must be an integral multiple of `num_col`.
                 Otherwise it will be trimmed if it is greater than num_col,
                 and incremented if if it is less then num_col.
        num_col: Int, number of images in one row - (default is 3)
                 For test and train respectively, it should not exceed 3 and 7.
        predictions: Numpy Array of predictions with shape (to_show, 1) -
                     (default is None)
                     Must be passed when test=True.
        test: Boolean telling whether the dataset being visualized is
              train dataset or test dataset - (default False).
    Returns:
        None.
    """

    # Define num_row
    # If to_show % num_col != 0
    #    trim to_show,
    #       to trim to_show limit num_row to the point where
    #       to_show % num_col == 0
    #
    # If to_show//num_col == 0
    #    then it means num_col is greater then to_show
    #    increment to_show
    #       to increment to_show set num_row to 1
    num_row = to_show // num_col if to_show // num_col != 0 else 1

    # `to_show` must be an integral multiple of `num_col`
    #  we found num_row and we have num_col
    #  to increment or decrement to_show
    #  to make it integral multiple of `num_col`
    #  simply set it equal to num_row * num_col
    to_show = num_row * num_col

    # Plot the images
    fig, axes = plt.subplots(num_row, num_col, figsize=(5, 5))
    for i in range(to_show):

        # If the number of rows is 1, the axes array is one-dimensional
        if num_row == 1:
            ax = axes[i % num_col]
        else:
            ax = axes[i // num_col, i % num_col]

        ax.imshow(tf.concat([pairs[i][0], pairs[i][1]], axis=1), cmap="gray")
        ax.set_axis_off()
        if test:
            ax.set_title("True: {} | Pred: {:.5f}".format(labels[i], predictions[i][0]))
        else:
            ax.set_title("Label: {}".format(labels[i]))
    if test:
        plt.tight_layout(rect=(0, 0, 1.9, 1.9), w_pad=0.0)
    else:
        plt.tight_layout(rect=(0, 0, 1.5, 1.5))
    plt.show()

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
    """
    ## Hyperparameters
    """

    epochs = 10
    batch_size = 32
    margin = 1  # Margin for constrastive loss.

    """
    ## Load the MNIST dataset
    """
    (x_train_val, y_train_val), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Change the data type to a floating point format
    x_train_val = x_train_val.astype("float32")
    x_test = x_test.astype("float32")


    """
    ## Define training and validation sets
    """

    # Keep 50% of train_val  in validation set
    x_train, x_val = x_train_val[:30000], x_train_val[30000:]
    y_train, y_val = y_train_val[:30000], y_train_val[30000:]
    del x_train_val, y_train_val

    # make train pairs
    pairs_train, labels_train = make_pairs(x_train, y_train)

    # make validation pairs
    pairs_val, labels_val = make_pairs(x_val, y_val)

    # make test pairs
    pairs_test, labels_test = make_pairs(x_test, y_test)

    """
    We get:
    **pairs_train.shape = (60000, 2, 28, 28)**
    - We have 60,000 pairs
    - Each pair contains 2 images
    - Each image has shape `(28, 28)`
    """

    """
    Split the training pairs
    """

    x_train_1 = pairs_train[:, 0]  # x_train_1.shape is (60000, 28, 28)
    x_train_2 = pairs_train[:, 1]

    """
    Split the validation pairs
    """

    x_val_1 = pairs_val[:, 0]  # x_val_1.shape = (60000, 28, 28)
    x_val_2 = pairs_val[:, 1]

    """
    Split the test pairs
    """

    x_test_1 = pairs_test[:, 0]  # x_test_1.shape = (20000, 28, 28)
    x_test_2 = pairs_test[:, 1]

    """
    training
    """

    siamese=self_sup_vgg16(shape=(28,28,1))
    siamese.compile(loss=loss(T=1), optimizer="RMSprop", metrics=["accuracy"])
    siamese.summary()
    history = siamese.fit(
        [x_train_1, x_train_2],
        labels_train,
        validation_data=([x_val_1, x_val_2], labels_val),
        batch_size=batch_size,
        epochs=epochs,
    )

    # Plot the accuracy
    plt_metric(history=history.history, metric="accuracy", title="Model accuracy")

    # Plot the constrastive loss
    plt_metric(history=history.history, metric="loss", title="Constrastive Loss")

    """
    ## Evaluate the model
    """

    results = siamese.evaluate([x_test_1, x_test_2], labels_test)
    print("test loss, test acc:", results)

    """
    ## Visualize the predictions
    """

    predictions = siamese.predict([x_test_1, x_test_2])
    visualize(pairs_test, labels_test, to_show=3, predictions=predictions, test=True)