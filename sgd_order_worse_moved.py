# Copyright 2022, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""An example for using advanced_mia."""

import datetime
import gc
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from absl import app, flags

from Attacks import amia
from Attacks import unclassified_utils as utils

FLAGS = flags.FLAGS
_LR = flags.DEFINE_float("learning_rate", 0.02, "Learning rate for training")
_BATCH = flags.DEFINE_integer("batch_size", 250, "Batch size")
_EPOCHS = flags.DEFINE_integer("epochs", 10, "Number of epochs")
_NUM_SHADOWS = flags.DEFINE_integer("num_shadows", 40, "Number of shadow models.")
_MODEL_DIR = flags.DEFINE_string("model_dir", "./Experiments", "Experiment directory.")
_WORSE_STRAT = flags.DEFINE_string(
    "worse_strat", "best_attacked", "How to pick the worse example(s)"
)
_DATASET = flags.DEFINE_string("dataset", "BLOB", "CIFAR10, BLOB, or MNIST")
_SEED = flags.DEFINE_integer("seed", 123, "Random seed")
_FLIP_IMG = flags.DEFINE_boolean("flip_img", False, "Flip the images or not")
_EXPE_DATE = flags.DEFINE_string("date", str(datetime.date.today()), "YYYY-MM-DD")
_MODEL = flags.DEFINE_string("type_model", "mlp", "cnn or mlp or logr")
_EXPE_TYPE = flags.DEFINE_string("type_expe", "order", "fit or order")
_NUM_SPOTS = flags.DEFINE_integer(
    "num_spots", 10, "How many spots to move the worse images at"
)

FLAGS(sys.argv)
EXP_PATH_BASE = (
    f"{_MODEL_DIR.value}/{_DATASET.value}/{_EXPE_DATE.value}/{_EXPE_TYPE.value}/"
    + f"{_MODEL.value}_lr{_LR.value}_b{_BATCH.value}_e{_EPOCHS.value}_"
    + f"nbs{_NUM_SHADOWS.value}_sd{_SEED.value}_flp{_FLIP_IMG.value}_"
    + f"{_WORSE_STRAT.value}"
)

NB_CLASSES = 10 if _DATASET.value == "CIFAR10" or _DATASET.value == "MNIST" else 3

# Functional

# TODO : Separate indices from stats
# TODO : Get rid of worse_image_index_in_train
# TODO : check I use the right indices in a more systematic way
# TODO : Use numpy.save rather than csv and save scores and stats from worse ing trials too
# TODO : Deal with warning on input np array to tf model
# TODO : Group the plotting in one place

# Good Practices

# TODO : typing
# TODO : docstring
# TODO : unzip flags in a less verbose way


def _get_or_train_model(
    x,
    y,
    nb_shadow,
    nb_epoch,
    batchsize,
    in_indices,
    n,
    seed,
    n_1_last_steps=True,
    last_step=True,
    custom_name=None,
    save_name=None,
):

    losses, stat = [], []

    for i in range(1, nb_shadow + 1):

        # Define model path

        if custom_name is not None:
            model_path = os.path.join(EXP_PATH_BASE, f"Models/{custom_name}.h5")
        else:
            model_path = os.path.join(EXP_PATH_BASE, f"Models/model{i}.h5")
        # Generate a binary array indicating which example to include for training
        in_indices_i = in_indices[i - 1]

        model = utils.make_model(
            model=_MODEL.value,
            data=_DATASET.value,
            flip=_FLIP_IMG.value,
            nb_classes=NB_CLASSES,
        )

        if os.path.exists(model_path):  # Load if exists
            model(x[:1])  # use this to make the `load_weights` work
            model.load_weights(model_path)
            print(f"Loaded model #{i} with {in_indices_i.sum()} examples.")

        else:  # Otherwise, train the model

            if n_1_last_steps:

                model.compile(
                    optimizer=tf.keras.optimizers.SGD(_LR.value, momentum=0.9),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(
                        from_logits=True
                    ),
                    metrics=["accuracy"],
                )

                model.fit(
                    x[in_indices_i],
                    y[in_indices_i],
                    validation_data=(x[~in_indices_i], y[~in_indices_i]),
                    epochs=nb_epoch - 1,
                    batch_size=batchsize,
                    verbose=2,
                )

            # Reproduce the exact learning schedule of shadow model

            if last_step:

                model.compile(
                    optimizer=tf.keras.optimizers.SGD(_LR.value / batchsize),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(
                        from_logits=True
                    ),
                    metrics=["accuracy"],
                )

                model.fit(
                    x[in_indices_i],
                    y[in_indices_i],
                    validation_data=(x[~in_indices_i], y[~in_indices_i]),
                    epochs=1,
                    batch_size=1,
                    verbose=2,
                )

            if save_name is not None:
                save_path = os.path.join(EXP_PATH_BASE, f"Models/save_name.h5")
                model.save_weights(save_path)
            else:
                model.save_weights(model_path)

            print(f"Trained model #{i} with {in_indices_i.sum()} examples.")

        # Get the statistics of the current model.
        s, l = amia.get_stat_and_loss_aug(model, x, y, flip=_FLIP_IMG.value)
        stat.append(s)
        losses.append(l)

        # Avoid OOM
        tf.keras.backend.clear_session()
        gc.collect()

    return stat, losses


def _move_worse_image_around(
    x, y, in_indices_target, stat, losses, worse_img_index, perc
):

    if perc < 1:
        spot = int(perc * x[in_indices_target[0]].shape[0])
    else:
        spot = x[in_indices_target[0]].shape[0] - len(worse_img_index)

    model_path = os.path.join(
        EXP_PATH_BASE,
        f"Models/target_worseimg{worse_img_index[0]}+{len(worse_img_index)-1}_atspot{spot}.h5",
    )

    if os.path.exists(model_path):  # Load if exists

        model = utils.make_model(
            model=_MODEL.value,
            data=_DATASET.value,
            flip=_FLIP_IMG.value,
            nb_classes=NB_CLASSES,
        )
        model(x[:1])  # use this to make the `load_weights` work
        model.load_weights(model_path)
        print(f"Loaded target model with worse image at index {spot} at the last epoch")

    else:  # Else retrain

        base_path = os.path.join(EXP_PATH_BASE, f"Models/model_target_temp.h5")
        model = utils.make_model(
            model=_MODEL.value,
            data=_DATASET.value,
            flip=_FLIP_IMG.value,
            nb_classes=NB_CLASSES,
        )
        model(
            x[:1]
        )  # use this to make the `load_weights` work #TODO why so I need this ?
        model.load_weights(base_path)
        print(f"Loaded base target model")

        x_train, y_train = x[in_indices_target[0]], y[in_indices_target[0]]
        x_val, y_val = x[~in_indices_target[0]], y[~in_indices_target[0]]

        worse_img = np.array(x_train[worse_img_index])
        worse_img_label = np.array(y[worse_img_index])

        x_pop = np.delete(x_train, worse_img_index, 0)
        y_pop = np.delete(y_train, worse_img_index, 0)

        x_train = np.insert(x_pop, spot, worse_img, 0)
        y_train = np.insert(y_pop, spot, worse_img_label, 0)

        model.compile(
            optimizer=tf.keras.optimizers.SGD(_LR.value / _BATCH.value),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=1,
            batch_size=1,
            verbose=2,
            shuffle=False,
        )

        model.save_weights(model_path)
        print(
            f"Trained target model with worse image at index {spot} at the last epoch"
        )

    # Get the statistics of the current model.

    s, l = amia.get_stat_and_loss_aug(model, x, y, flip=_FLIP_IMG.value)
    stat.append(s)
    losses.append(l)

    # Avoid OOM
    tf.keras.backend.clear_session()
    gc.collect()

    return stat, losses


def _run_full_attack(stat, in_indices, idx, nb_spots):

    stat_target = stat[idx]  # statistics of target model, shape (n, k)
    in_indices_target = in_indices[idx]  # ground-truth membership, shape (n,)

    # `stat_shadow` contains statistics of the shadow models, with shape (num_shadows, n, k).
    stat_shadow = np.array(stat[nb_spots:])

    # `in_indices_shadow` contains membership of the shadow models, with shape (num_shadows, n)
    in_indices_shadow = np.array(in_indices[nb_spots:])

    # stat_in[j] (resp. stat_out[j]) is a (m, k) array, for m being the number of shadow models trained with
    # (resp. without) the j-th example, and k being the number of augmentations (2 in our case)
    stat_in = [
        stat_shadow[:, j][in_indices_shadow[:, j]] for j in range(stat_target.shape[0])
    ]
    stat_out = [
        stat_shadow[:, j][~in_indices_shadow[:, j]] for j in range(stat_target.shape[0])
    ]

    # Compute the scores and use them for MIA
    scores = amia.compute_score_lira(
        stat_target, stat_in, stat_out, fix_variance=True
    )  # That is where the attack properly happens

    print(
        f"The average score of in sample is {round(scores[in_indices_target].mean(),4)}, and out sample is {round(scores[~in_indices_target].mean(),4)}"
    )

    return scores


def _get_worse_classified_image(x, y, path):

    model = utils.make_model(
        model=_MODEL.value,
        data=_DATASET.value,
        flip=_FLIP_IMG.value,
        nb_classes=NB_CLASSES,
    )
    model(x[:1])  # use this to make the `load_weights` work #TODO why so I need this ?
    model.load_weights(path)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )

    all_losses = np.array(loss_object(y, model.predict(x)))

    sorted_loss = list(np.flip(np.argsort(all_losses), 0)[:100])

    return sorted_loss


def main(unused_argv):

    del unused_argv

    # Path
    parent_dir = EXP_PATH_BASE
    directories = ["Models", "Stats", "Figures"]
    utils.create_directories(parent_dir, directories)

    # Set seed

    seed = _SEED.value
    np.random.seed(seed)

    # Load data.

    if _DATASET.value == "MNIST":
        x, y = utils.load_mnist()
    elif _DATASET.value == "CIFAR10":
        x, y = utils.load_cifar10()
    elif _DATASET.value == "BLOB":
        x, y = utils.load_blobs(NB_CLASSES, _SEED.value)
    else:
        raise Exception("Unauthorized value")

    n = x.shape[0]

    # Shadow models

    existing_paths = []
    in_indices_shadow = []

    for i in range(_NUM_SHADOWS.value):
        result_path = os.path.join(EXP_PATH_BASE, f"Stats/stat_shadow_{i+1}.csv")
        existing_paths.append(os.path.exists(result_path))

    if sum(existing_paths) == _NUM_SHADOWS.value:  # Load stats if exists

        stat_shadow = []

        for i in range(_NUM_SHADOWS.value):
            stat = np.genfromtxt(
                f"{EXP_PATH_BASE}/Stats/stat_shadow_{i+1}.csv", delimiter=","
            )
            stat_shadow.append(stat[:, 1:])
            in_indices_shadow.append(stat[:, 0].astype("bool"))
            print(f"Loaded previously run stats from shadow model number {i+1}")

        model = utils.make_model(
            model=_MODEL.value,
            data=_DATASET.value,
            flip=_FLIP_IMG.value,
            nb_classes=NB_CLASSES,
        )
        test_acc, train_acc = utils.evaluate_model_for_sensecheck(
            x, y, in_indices_shadow[0], model, EXP_PATH_BASE, "model1"
        )

        print(
            f"Test accuracy of first fully trained shadow model is {test_acc:.4f}, train accuracy is {train_acc:.4f}"
        )

    else:  # Else train shadow models

        for i in range(_NUM_SHADOWS.value):
            in_indices_shadow.append(np.random.binomial(1, 0.5, n).astype(bool))

        stat_shadow, _ = _get_or_train_model(
            x,
            y,
            _NUM_SHADOWS.value,
            _EPOCHS.value,
            _BATCH.value,
            in_indices_shadow,
            n,
            seed,
        )

        for i, shadow in enumerate(stat_shadow):
            log_stat = np.concatenate([in_indices_shadow[i][:, None], shadow], 1)
            np.savetxt(
                f"{EXP_PATH_BASE}/Stats/stat_shadow_{i+1}.csv", log_stat, delimiter=","
            )

    # Base target model

    in_indices_target = []
    stat_path = os.path.join(EXP_PATH_BASE, f"Stats/stat_target_temp.csv")
    base_path = os.path.join(EXP_PATH_BASE, f"Models/model_target_temp.h5")

    if os.path.exists(stat_path) and os.path.exists(base_path):  # Load index if exists

        base = np.genfromtxt(stat_path, delimiter=",")
        in_indices_target.append(base[:, 0].astype("bool"))
        stat_base = [np.array(base[:, 1:])]

        print(f"Loaded in_indices from previously run target_temp model")

    else:

        if _DATASET.value == "BLOB":
            a = np.concatenate(
                [
                    np.ones(10).astype(bool),
                    np.random.binomial(1, 0.48, n - 10).astype(bool),
                ]
            )
            in_indices_target.append(a)
        else:
            in_indices_target.append(np.random.binomial(1, 0.5, n).astype(bool))
        stat_base, _ = _get_or_train_model(
            x,
            y,
            1,
            _EPOCHS.value,
            _BATCH.value,
            in_indices_target,
            n,
            seed,
            last_step=False,
            custom_name="model_target_temp",
        )
        log_stat = np.concatenate([in_indices_target[0][:, None], stat_base[0]], 1)
        np.savetxt(stat_path, log_stat, delimiter=",")

    # Get worse-classified or most sensitive image

    k = 10  # Top k worse images

    index_in_indices = list(np.where(in_indices_target[0])[0])

    # Check base performmamce of the attack

    stat_base_plus_one, _ = _get_or_train_model(
        x,
        y,
        1,
        _EPOCHS.value,
        _BATCH.value,
        in_indices_target,
        n,
        seed,
        n_1_last_steps=False,
        last_step=True,
        custom_name="model_target_temp",
        save_name="model_target_plus_one",
    )

    base_path_plus_one = os.path.join(EXP_PATH_BASE, f"Models/model_target_temp.h5")
    base_scores = np.array(
        _run_full_attack(
            stat_base_plus_one + stat_shadow,
            in_indices_target + in_indices_shadow,
            0,
            1,
        )
    )
    utils.get_auc(in_indices_target[0], base_scores)

    # TESTING

    if _WORSE_STRAT.value == "random":
        sorted_base_scores = np.random.randint(0, n, 100)

    elif _WORSE_STRAT.value == "highest_loss":
        sorted_base_scores = _get_worse_classified_image(x, y, base_path_plus_one)

    else:
        sorted_base_scores = list(np.argsort(base_scores)[:100])

    worse_img_index = [i for i in sorted_base_scores if i in index_in_indices][:k]

    worse_img_index_in_train = [index_in_indices.index(i) for i in worse_img_index]

    print(f"The best attacked {k} images are at index {worse_img_index}")

    # Train different version of last step

    nb_spots = 10

    stat_target, losses_target = [], []

    for spot in range(nb_spots + 1):

        in_indices_target.append(in_indices_target[0])

        stat_target, losses_target = _move_worse_image_around(
            x,
            y,
            in_indices_target,
            stat_target,
            losses_target,
            worse_img_index_in_train,
            spot / nb_spots,
        )

    in_indices_target.pop(0)

    # Put together stats

    in_indices = in_indices_target + in_indices_shadow
    stat = stat_target + stat_shadow
    # losses = losses_target + losses_shadow

    a = np.array([s[1] for s in stat_shadow])
    l = [s[worse_img_index[0]] for s in in_indices_shadow]
    plt.hist(a[l], bins="doane", alpha=0.5, color="red", label="in")
    l = [np.invert(s)[worse_img_index[0]] for s in in_indices_shadow]
    plt.hist(a[l], bins="doane", alpha=0.5, color="blue", label="out")
    plt.legend()
    plt.savefig(
        f"{EXP_PATH_BASE}/Figures/shadow_stats_for_img_{worse_img_index[0]}.png"
    )
    plt.close()

    # Now we do the MIA for all versions of the target model

    scores, fprs, tprs = [], [], []

    for idx in range(nb_spots + 1):

        print(f"Target model is #{idx}")
        score = _run_full_attack(stat, in_indices, idx, nb_spots + 1)
        fpr, tpr = utils.get_auc(in_indices[idx], score)

        scores.append(score)
        fprs.append(fpr)
        tprs.append(tpr)

        print(
            f"Average score over the {k} worse images is {score[worse_img_index].mean()}"
        )

    fig_name = f"{EXP_PATH_BASE}/Figures/target_AUC_ROC_at_{_NUM_SPOTS.value}_spots.png"
    utils.plot_auc(fprs, tprs, fig_name)
    plt.close()

    # Plot the influence of order on the attack

    worse_scores = [score[worse_img_index] for score in scores]
    worse_scores_mean = [m.mean() for m in worse_scores]
    worse_losses = [a[worse_img_index].mean(axis=1) for a in losses_target]
    worse_losses_mean = [l.mean() for l in worse_losses]

    x_ax = np.array(
        [(i * n) / (2 * nb_spots) for i in list(range(len(worse_scores_mean)))]
    )
    plt.scatter(x_ax, worse_scores_mean)
    m, b = np.polyfit(x_ax, worse_scores_mean, 1)
    plt.plot(x_ax, m * x_ax + b, color="red")
    plt.savefig(
        f"{EXP_PATH_BASE}/Figures/test_of_worse_moved_{worse_img_index[0]}+{len(worse_img_index)-1}.png"
    )
    plt.close()

    x_ax = np.array(
        [(i * n) / (2 * nb_spots) for i in list(range(len(worse_losses_mean)))]
    )
    plt.scatter(x_ax, worse_losses_mean)
    m, b = np.polyfit(x_ax, worse_losses_mean, 1)
    plt.plot(x_ax, m * x_ax + b, color="red")
    plt.savefig(
        f"{EXP_PATH_BASE}/Figures/test_of_worse_moved_loss{worse_img_index[0]}+{len(worse_img_index)-1}.png"
    )
    plt.close()

    fig, axes = plt.subplots(5, 2, figsize=(8, 8))

    for idx, ax in enumerate(axes.flatten()):
        x_ax = np.array(range(np.array(worse_scores).shape[0]))
        y_ax = np.array(worse_scores)[:, idx]
        ax.scatter(x_ax, y_ax)
        m, b = np.polyfit(x_ax, y_ax, 1)
        ax.plot(x_ax, m * x_ax + b, color="red")

    fig.savefig(
        f"{EXP_PATH_BASE}/Figures/test_of_worse_moved_details_{worse_img_index[0]}+{len(worse_img_index)-1}.png"
    )
    plt.close()

    fig, axes = plt.subplots(5, 2, figsize=(8, 8))

    for idx, ax in enumerate(axes.flatten()):
        x_ax = np.array(range(np.array(worse_losses).shape[0]))
        y_ax = np.array(worse_losses)[:, idx]
        ax.scatter(x_ax, y_ax)
        m, b = np.polyfit(x_ax, y_ax, 1)
        ax.plot(x_ax, m * x_ax + b, color="red")

    fig.savefig(
        f"{EXP_PATH_BASE}/Figures/test_of_worse_moved_loss_details_{worse_img_index[0]}+{len(worse_img_index)-1}.png"
    )


if __name__ == "__main__":
    app.run(main)
