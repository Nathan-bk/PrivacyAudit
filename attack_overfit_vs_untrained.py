import datetime
import gc
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from absl import app, flags
from matplotlib.pyplot import cm

from Attacks import amia
from Attacks import unclassified_utils as utils

FLAGS = flags.FLAGS
_LR = flags.DEFINE_float("learning_rate", 0.02, "Learning rate for training")
_BATCH = flags.DEFINE_integer("batch_size", 250, "Batch size")
_EPOCHS = flags.DEFINE_integer("epochs", 21, "Number of epochs")
_NUM_SHADOWS = flags.DEFINE_integer("num_shadows", 40, "Number of shadow models.")
_MODEL_DIR = flags.DEFINE_string("model_dir", "./Experiments", "Experiment directory.")
_WORSE_STRAT = flags.DEFINE_string(
    "worse_strat", "highest_loss", "How to pick the worse example(s)"
)
_DATASET = flags.DEFINE_string("dataset", "BLOB", "CIFAR10, BLOB, SBLOB or MNIST")
_SEED = flags.DEFINE_integer("seed", 12, "Random seed")
_FLIP_IMG = flags.DEFINE_boolean("flip_img", False, "Flip the images or not")
_EXPE_DATE = flags.DEFINE_string("date", str(datetime.date.today()), "YYYY-MM-DD")
_MODEL = flags.DEFINE_string("type_model", "mlp", "cnn, mlp, mlp_basic, or logr")
_EXPE_TYPE = flags.DEFINE_string("type_expe", "fit", "fit or order")
_NUM_MID_SAVES = flags.DEFINE_integer(
    "mid_save", 7, "How many intermediary models to save"
)

FLAGS(sys.argv)
EXP_PATH_BASE = (
    f"{_MODEL_DIR.value}/{_DATASET.value}/{_EXPE_DATE.value}/{_EXPE_TYPE.value}/"
    + f"{_MODEL.value}_lr{_LR.value}_b{_BATCH.value}_e{_EPOCHS.value}_"
    + f"nbs{_NUM_SHADOWS.value}_sd{_SEED.value}_flp{_FLIP_IMG.value}"
)

if _DATASET.value == "CIFAR10" or _DATASET.value == "MNIST":
    NB_CLASSES = 10
elif _DATASET.value == "SBLOB":
    NB_CLASSES = 2
else:
    NB_CLASSES = 3

# Functional

# TODO : Make sure it slicing still works with augmentations in _run_attack
# TODO : Make sure the noisy SGD actually works

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
    nb_mid_saves=_NUM_MID_SAVES.value,
):

    losses, stat = [], []

    nb_epoch_i = int((nb_epoch / nb_mid_saves))

    # NoisySGD = add_gradient_noise(tf.keras.optimizers.SGD)

    model_init = utils.make_model(
        model=_MODEL.value,
        data=_DATASET.value,
        flip=_FLIP_IMG.value,
        nb_classes=NB_CLASSES,
    )

    model_init(x[:1])  # This is to build the weights > any more elegant solution ?

    for i in range(1, nb_shadow + 1):

        in_indices_i = in_indices[i - 1]

        model = utils.make_model(
            model=_MODEL.value,
            data=_DATASET.value,
            flip=_FLIP_IMG.value,
            nb_classes=NB_CLASSES,
            seed=_SEED.value,
        )

        model.compile(
            optimizer=tf.keras.optimizers.SGD(_LR.value, momentum=0.9),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        # Load if exist

        for idx in range(1, nb_mid_saves + 1):

            model_path = os.path.join(
                EXP_PATH_BASE, f"Models/model{i}_{idx}-{nb_mid_saves}.h5"
            )

            model.fit(
                x[in_indices_i],
                y[in_indices_i],
                validation_data=(x[~in_indices_i], y[~in_indices_i]),
                epochs=nb_epoch_i,
                batch_size=batchsize,
                verbose=1,
            )

            model.save_weights(model_path)

            print(f"Trained model #{i} : {idx}/{nb_mid_saves}.")

            # Get the statistics of the current model.

            s, l = amia.get_stat_and_loss_aug(model, x, y, flip=_FLIP_IMG.value)
            stat.append(s)
            losses.append(l)

        tf.keras.backend.clear_session()
        gc.collect()

        print(f"Trained model #{i} with {in_indices_i.sum()} examples.")

    return stat, losses


def _run_full_attack(stat, in_indices, i, idx, nb_mid_saves):

    idx = idx - 1

    stat_target = stat[i + idx]  # statistics of target model, shape (n, k)
    in_indices_target = in_indices[i]  # ground-truth membership, shape (n,)

    # `stat_shadow` contains statistics of the shadow models, with shape (num_shadows, n, k).
    stat_indices = [i * nb_mid_saves + idx for i in range(i + 1, len(in_indices))]
    stat_shadow = np.array([stat[i] for i in stat_indices])

    # `in_indices_shadow` contains membership of the shadow models, with shape (num_shadows, n)
    in_indices_shadow = np.array(in_indices[i + 1 :])

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


def get_row_max(arr):

    idxs = np.argmax(np.abs(arr), axis=1)
    idxs = np.expand_dims(idxs, axis=1)
    result = np.take_along_axis(arr, idxs, axis=1)
    result = np.squeeze(result, axis=1)
    return result


def main(unused_argv):

    del unused_argv

    # Check Exp dir

    parent_dir = EXP_PATH_BASE
    directories = ["Models", "IndicesAndStats", "Figures"]
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
    elif _DATASET.value == "SBLOB":
        x, y = utils.load_simple_blobs(NB_CLASSES)
    else:
        raise Exception("Unauthorized value for data")

    n = x.shape[0]

    # Shadow models

    paths = []
    in_indices_shadow = []
    nb_mid_saves = _NUM_MID_SAVES.value
    nb_cols = _NUM_SHADOWS.value * nb_mid_saves

    for i in range(1, _NUM_SHADOWS.value + 1):

        for idx in range(1, nb_mid_saves + 1):

            # Define model path

            model_path = os.path.join(
                EXP_PATH_BASE, f"Models/model{i}_{idx}-{nb_mid_saves}.h5"
            )
            paths.append(model_path)

    if sum([os.path.exists(p) for p in paths]) == nb_cols:  # Load stats if exists

        stat_shadow = list(
            np.load(f"{EXP_PATH_BASE}/IndicesAndStats/stat_shadows.npy").astype(float).T
        )
        in_indices_shadow = list(
            np.load(f"{EXP_PATH_BASE}/IndicesAndStats/indices_shadows.npy")
            .astype(bool)
            .T
        )
        print(f"Loaded previously run stats from shadow models")

        model = utils.make_model(
            model=_MODEL.value,
            data=_DATASET.value,
            flip=_FLIP_IMG.value,
            nb_classes=NB_CLASSES,
        )
        model_name = f"model1_{_NUM_MID_SAVES.value}-{_NUM_MID_SAVES.value}"

        test_acc, train_acc = utils.evaluate_model(
            x, y, in_indices_shadow[0], model, EXP_PATH_BASE, model_name
        )

        print(
            f"Test accuracy of first fully trained shadow model is {test_acc:.4f}, train accuracy is {train_acc:.4f}"
        )

    else:  # Else train shadow models

        for i in range(1, _NUM_SHADOWS.value + 1):

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
            nb_mid_saves,
        )

        for i, shadow in enumerate(stat_shadow):
            np.save(
                f"{EXP_PATH_BASE}/IndicesAndStats/indices_shadows",
                np.array(in_indices_shadow).T,
            )
            np.save(
                f"{EXP_PATH_BASE}/IndicesAndStats/stat_shadows", np.array(stat_shadow).T
            )

    # Now we do the MIA for all versions of the target model

    scores, fprs, tprs = [], [], []

    print("Target model is shadow model 1")

    for idx in range(1, nb_mid_saves + 1):

        score = _run_full_attack(stat_shadow, in_indices_shadow, 0, idx, nb_mid_saves)
        fpr, tpr = utils.get_auc(in_indices_shadow[0], score)

        scores.append(score)
        fprs.append(fpr)
        tprs.append(tpr)

    # Adding most confident score

    scores_arr = np.array(scores).T

    score = get_row_max(scores_arr)
    scores.append(score)
    fpr, tpr = utils.get_auc(in_indices_shadow[0], score)
    fprs.append(fpr)
    tprs.append(tpr)

    # scores_normzd = -1 + (scores_arr  - scores_arr.min(0))*2 / scores_arr.ptp(0)
    # score = get_row_max(scores_normzd)
    # scores.append(score)
    # fpr, tpr = utils.get_auc(in_indices_shadow[0],score)
    # fprs.append(fpr)
    # tprs.append(tpr)

    n = len(fprs)
    curve_names = [
        f"LIRA - intermediary model {idx+1}/{n-2}" for idx in range(n - 2)
    ] + ["LiRA - fully trained model", "MaxLiRA"]
    # 'LiRA - normliazed most confident']

    custom_colors = list(iter(cm.Blues(np.linspace(0.3, 1, n - 1)))) + ["red", "gold"]

    fig_name = f"{EXP_PATH_BASE}/Figures/target_AUC_ROC_{nb_mid_saves}.png"
    utils.plot_auc(
        fprs,
        tprs,
        fig_name,
        curve_names,
        custom_colors,
        title="ROC-AUC curve for different LiRA attacks on simple MLP with noisy optimizer",
    )
    plt.close()


if __name__ == "__main__":
    app.run(main)
