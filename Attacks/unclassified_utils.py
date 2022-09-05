import functools
import os
import random
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.pyplot import cm
from sklearn import datasets, metrics

# Model functions


def make_model(model, data, flip, nb_classes, seed=None):  # added non  default

    """Setup either a logistic regression a small CNN for image classification."""

    if seed:
        tf.random.set_seed(seed)

    if model == "cnn":

        model = tf.keras.models.Sequential()

        pixels = 32
        channels = 3

        if data == "MNIST":
            pixels = 28
            channels = 1

        input_shape = (pixels, pixels, channels)

        # Add a layer to do random horizontal augmentation.
        if flip:
            model.add(tf.keras.layers.RandomFlip("horizontal"))
        model.add(tf.keras.layers.Input(shape=input_shape))

        for _ in range(3):
            model.add(
                tf.keras.layers.Conv2D(pixels, (3, 3), activation="relu")
            )  # ,input_shape = input_shape))
            model.add(tf.keras.layers.MaxPooling2D())

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation="relu"))
        model.add(tf.keras.layers.Dense(nb_classes))

    elif model == "logr":
        # l2 reg, zero initilized logr
        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.Dense(
                nb_classes,
                kernel_regularizer=tf.keras.regularizers.L2(0.02),
                # kernel_initializer=tf.keras.initializers.Zeros,
            )
        )

    elif model == "mlp":
        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.Dense(90, activation="relu")
            # , kernel_regularizer=tf.keras.regularizers.L2(0.02))
        )
        model.add(tf.keras.layers.Dense(60, activation="relu"))
        model.add(tf.keras.layers.Dense(nb_classes))

    elif model == "mlp_basic":
        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.Dense(
                9, activation="relu", kernel_regularizer=tf.keras.regularizers.L2(0.02)
            )
        )
        model.add(
            tf.keras.layers.Dense(
                6, activation="relu", kernel_regularizer=tf.keras.regularizers.L2(0.02)
            )
        )
        model.add(tf.keras.layers.Dense(nb_classes))

    else:
        raise Exception("Unauthorized value for model")

    return model


def evaluate_model(x, y, in_indices_i, model, exp_path_base, model_name):  # added three

    model_path = os.path.join(exp_path_base, f"Models/{model_name}.h5")

    model(x[:1])  # use this to make the `load_weights` work
    model.load_weights(model_path)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(0.02, momentum=0.9),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    _, test_acc = model.evaluate(x[~in_indices_i], y[~in_indices_i])
    _, train_acc = model.evaluate(x[in_indices_i], y[in_indices_i])

    return test_acc, train_acc


# Data functions


def load_cifar10():
    """Loads CIFAR10, with training and test combined."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x = np.concatenate([x_train, x_test]).astype(np.float32) / 255
    y = np.concatenate([y_train, y_test]).astype(np.int32).squeeze()
    return x, y


def load_mnist():
    """Loads MNIST, with training and test combined."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x = np.concatenate([x_train, x_test]).astype(np.float32) / 255
    y = np.concatenate([y_train, y_test]).astype(np.int32).squeeze()

    x = x.reshape((x.shape[0], 28, 28, 1))

    return x, y


def load_sparse(nb_classes, seed):

    np.random.seed(seed)
    random.seed(seed)

    samples = 2000
    features = 200

    col = np.vstack(
        [np.ones((int(samples / 2), 1)), np.ones((int(samples / 2), 1)) * (-1)]
    )

    y = np.hstack([np.ones((int(samples / 2))), np.zeros((int(samples / 2)))])

    X = col

    for _ in range(features - 1):
        X = np.hstack([X, col])

    # Making X sparse

    sparse_mask = np.random.choice(
        a=[False, True], size=(samples, features), p=[0.99, 0.01]
    )
    X = np.where(sparse_mask, X, 0)

    # Add indic

    # for indic in range(nb_indics):
    #     zero_col = np.zeros_like(X[:, 0:1])
    #     X = np.hstack([X, zero_col])
    #     X[-1 - indic, -1] = 1

    # outlier_idx = [X.shape[0] - 1 - d for d in range(10)]

    print(
        f"Nb of all zero rows : {np.where(~X.any( axis=1))[0].size}",
        f"Nb of unique rows : {np.unique(X,axis=0).shape[0]}",
    )

    X = X.astype("int")
    y = y.astype("int")

    return X, y


def load_blobs_plus_dirac_data(nb_classes, seed, nb_diracs=10):

    np.random.seed(seed)

    center1 = np.random.choice([-2, 2], 10)
    centers = np.vstack([center1, -1 * center1])

    X, y = datasets.make_blobs(n_samples=1000, centers=centers, cluster_std=2)

    # Add dirac

    median_row = np.median(X[[y == 1]], axis=0)
    for dirac in range(nb_diracs):
        zero_col = np.zeros_like(X[:, 0:1])
        X = np.hstack([X, zero_col])
        median_row = np.hstack([median_row, np.array([0])])
        X = np.vstack([X, median_row])
        y = np.hstack([y, np.array([1]).T])
        X[-1, -1] = 100

    print(
        f"Nb of all zero rows : {np.where(~X.any( axis=1))[0].size}",
        f"Nb of unique rows : {np.unique(X,axis=0).shape[0]}",
    )

    outlier_idx = [X.shape[0] - 1 - d for d in range(nb_diracs)]

    return X, y, outlier_idx


def load_blobs_outliers_data(nb_classes, seed, out_policy="large", nb_outliers=10):

    np.random.seed(seed)

    center1 = np.random.choice([-2, 2], 10)
    # center1 = np.full((10), -2)
    centers = np.vstack([center1, -1 * center1])

    X, y = datasets.make_blobs(n_samples=1000, centers=centers, cluster_std=2)

    # Add outlier

    out_factor = 1
    if out_policy in ["large", "largeflip"]:
        out_factor *= 100
    if out_policy in ["flip", "largeflip"]:
        out_factor *= -1

    median_row = np.median(X[[y == 1]], axis=0)
    for idx in range(nb_outliers):
        X = np.vstack([X, median_row])
        y = np.hstack([y, np.array([1]).T])
        X[-1, idx] = X[-1, idx] * out_factor

    print(
        f"Nb of all zero rows : {np.where(~X.any( axis=1))[0].size}",
        f"Nb of unique rows : {np.unique(X,axis=0).shape[0]}",
    )

    outlier_idx = [X.shape[0] - nb_outliers + d for d in range(nb_outliers)]

    return X, y, outlier_idx


def load_blobs(nb_classes, seed):  # added one

    np.random.seed(seed)

    X, y = datasets.make_classification(
        n_samples=5000,
        n_features=12,
        n_informative=9,
        n_repeated=0,
        n_redundant=0,
        class_sep=0.6,
        n_classes=nb_classes,
        n_clusters_per_class=2,
        flip_y=0.01,
    )

    # sparse_mask = np.random.choice(
    #     a=[False, True], size=(X.shape[0], X.shape[1]), p=[0.99, 0.01]
    # )
    # X = np.where(sparse_mask, X, 0)

    # print(
    #     f"Nb of all zero rows : {np.where(~X.any( axis=1))[0].size}",
    #     f"Nb of unique rows : {np.unique(X,axis=0).shape[0]}",
    # )

    # X = X.astype("int")
    # y = y.astype("int")

    # def load_blobs(nb_classes, seed):

    #     np.random.seed(seed)

    #     center1 = np.random.choice([-2, 2], 30)
    #     centers = np.vstack([center1, -1 * center1])

    #     X, y = datasets.make_blobs(n_samples=1000, centers=centers, cluster_std=1.5)

    #     return X, y

    return X, y


# AUC and plotting functions


def plot_curve_with_area(
    x, y, ax, label, color, xlabel="FPR", ylabel="TPR", title=None
):
    ax.plot([0, 1], [0, 1], "k-", lw=1.0)
    ax.plot(x, y, lw=2, label=label, c=color)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set(aspect=1, xscale="log", yscale="log")
    ax.set_xlim(10e-4, 1)
    ax.set_ylim(10e-4, 1)
    ax.title.set_text(title)


def get_auc(in_indices, score):

    acc = metrics.accuracy_score(in_indices, score < 0)

    fpr, tpr, thresholds = metrics.roc_curve(
        in_indices, -score
    )  # invert in indices to reflect score signs

    auc = metrics.auc(fpr, tpr)
    adv = max(np.abs(tpr - fpr))

    low_fpr_idx = max(np.array([np.where(fpr > 0.001)]).min() - 1, 0)

    print(
        f"auc = {auc:.4f}",
        f"adv = {adv:.4f}",
        f"acc @ zero = {acc:.4f}",
        f"tpr@.1%fpr = {tpr[low_fpr_idx]*100:.2f}%",
    )

    return fpr, tpr


def plot_auc(
    fprs, tprs, fig_name, custom_title_list=None, custom_colors=None, title="ROC-AUC"
):

    n = len(fprs)
    _, ax = plt.subplots(1, 1, figsize=(8, 8))
    colors = iter(cm.rainbow(np.linspace(0, 1, n)))

    for idx in range(n):
        fpr = fprs[idx]
        tpr = tprs[idx]
        auc = metrics.auc(fpr, tpr)
        low_fpr_idx = max(np.array([np.where(fpr > 0.001)]).min() - 1, 0)

        if custom_title_list is not None:
            label = f"{custom_title_list[idx]}\nauc={auc*100:.1f}%, tpr@.1%fpr \
                {tpr[low_fpr_idx]*100:.2f}%"

        else:
            label = f"LIRA @ {idx+1}/{n}, auc={auc*100:.1f}"

        if custom_colors is not None:
            color = custom_colors[idx]
        else:
            color = next(colors)

        plot_curve_with_area(fpr, tpr, ax, label, color)

    plt.title(title)
    plt.legend()  # bbox_to_anchor=(1.04, 1))
    plt.savefig(fig_name, bbox_inches="tight")


# Pure utils


def create_directories(parent_dir, directories):

    for directory in directories:

        path = os.path.join(parent_dir, directory)

        try:
            os.makedirs(path, exist_ok=True)
            print("Directory '%s' created successfully" % path)

        except OSError as error:
            print("Directory '%s' can not be created" % path)


def concat_gradient_per_layer(grad: List):
    new_grad = []

    for idx, el in enumerate(grad[::2]):
        grad_layer = tf.concat([el, grad[idx * 2 + 1][None, :]], 0)
        grad_layer = np.array(grad_layer).tolist()
        new_grad.append(grad_layer)

    return new_grad


# from https://github.com/tensorflow/tensorflow/issues/33478#issuecomment-843720638
# not currently used


def proxy_call(input: tf.Tensor, obj: tf.keras.layers.Layer) -> tf.Tensor:
    if obj._before_call is not None:
        obj._before_call(obj, input)
    output = obj.call(input)
    if obj._after_call is not None:
        hook_result = obj._after_call(obj, input, output)
        if hook_result is not None:
            output = hook_result
    return output


def get_activation(layer: tf.keras.layers.Layer, input: tf.Tensor, output: tf.Tensor):
    return output


def hook_layer_call(
    layers: List[tf.keras.layers.Layer],
    before_call: Callable[[tf.keras.layers.Layer, tf.Tensor], None] = None,
    after_call: Callable[
        [tf.keras.layers.Layer, tf.Tensor, tf.Tensor], Optional[tf.Tensor]
    ] = get_activation,
):
    for layer in layers:
        layer._before_call = before_call
        layer._after_call = after_call
        layer.customcall = functools.partial(proxy_call, obj=layer)
