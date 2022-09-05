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
_LR = flags.DEFINE_float("learning_rate", 0.002, "Learning rate for training")
_BATCH = flags.DEFINE_integer("batch_size", 250, "Batch size")
_EPOCHS = flags.DEFINE_integer("epochs", 20, "Number of epochs")
_NUM_SHADOWS = flags.DEFINE_integer("num_shadows", 80, "Number of shadow models.")
_MODEL_DIR = flags.DEFINE_string("model_dir", "./Experiments", "Experiment directory.")
_DATASET = flags.DEFINE_string(
    "dataset",
    "SPARSE",
    "CIFAR10, BLOB, SBLOB, SPARSE, SBLOB&DIRAC, SBLOB&OUTLIER or MNIST",
)
_SEED = flags.DEFINE_integer("seed", 123, "Random seed")
_FLIP_IMG = flags.DEFINE_boolean("flip_img", False, "Flip the images or not")
_EXPE_DATE = flags.DEFINE_string("date", str(datetime.date.today()), "YYYY-MM-DD")
_MODEL = flags.DEFINE_string("type_model", "logr", "cnn, mlp, mlp_basic, or logr")
_EXPE_TYPE = flags.DEFINE_string("type_expe", "param", "grad, activ, param")
_OUTLIER_POLICY = flags.DEFINE_string("out_policy", "large", "large, flip, largeflip")

FLAGS(sys.argv)

EXP_PATH_BASE = (
    f"{_MODEL_DIR.value}/{_DATASET.value}/{_EXPE_DATE.value}/{_EXPE_TYPE.value}/"
    + f"{_MODEL.value}_lr{_LR.value}_b{_BATCH.value}_e{_EPOCHS.value}_"
    + f"nbs{_NUM_SHADOWS.value}_sd{_SEED.value}_flp{_FLIP_IMG.value}"
)

if _DATASET.value in ["CIFAR10", "MNIST"]:
    NB_CLASSES = 10
elif _DATASET.value in ["SBLOB", "SBLOB&DIRAC", "SPARSE", "SBLOB&OUTLIER"]:
    NB_CLASSES = 2
else:
    NB_CLASSES = 3

if _DATASET.value == "SBLOB&OUTLIER":
    EXP_PATH_BASE = EXP_PATH_BASE + f"_outpol_{_OUTLIER_POLICY.value}"

# Functional

# TODO collect params OK, _run_param_attack OK, amia.run_param_attack OK, _POI viz
# TODO switch to classes rather than purely functional


# Good Practices

# TODO : typing
# TODO : docstring
# TODO : unzip flags in a less verbose way

# Note

# Forcing POI to be in target and all POI are class 1
# Small l2 reg, 0.02 lr, 8 epochs, 80 shadow


def _get_or_train_model(x, y, nb_shadow, nb_epoch, batchsize, in_indices, seed=None):

    if seed:
        tf.random.set_seed(seed)

    losses, stats, params = [], [], []

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

        model_path = os.path.join(EXP_PATH_BASE, f"Models/model{i}.h5")

        x_tf = tf.convert_to_tensor(x)
        y_tf = tf.convert_to_tensor(y)

        model.fit(
            x_tf[in_indices_i],
            y_tf[in_indices_i],
            validation_data=(x_tf[~in_indices_i], y_tf[~in_indices_i]),
            epochs=nb_epoch,
            batch_size=batchsize,
            verbose=1,
        )

        model.save_weights(model_path)

        # Save the params #ATTACKSPECIFIC

        model_weights = model.trainable_weights
        model_weights_wbias = utils.concat_gradient_per_layer(model_weights)
        param = list(np.array(model_weights_wbias))

        print(f"Trained model #{i}.")

        # Get the statistics of the current model.

        stat, loss = amia.get_stat_and_loss_aug(model, x, y, flip=_FLIP_IMG.value)
        stats.append(stat)
        losses.append(loss)
        params.append(param)

        tf.keras.backend.clear_session()
        gc.collect()

        print(f"Trained model #{i} with {in_indices_i.sum()} examples.")

    # Transform params to list of np.arrays #ATTACKSPECIFIC

    if len(np.array(params).shape) == 4:

        params = list(np.transpose(np.array(params), (1, 0, 2, 3)))

    elif len(np.array(params).shape) == 2:
        params_transpose = np.transpose(np.array(params), (1, 0)).tolist()

        params = []

        for layer in params_transpose:
            params.append(np.array(layer))

    else:
        raise ValueError("Wrong params dimension")

    return stats, losses, params


def _run_loss_attack(stat, in_indices, i):

    stat_target = stat[i]  # statistics of target model, shape (n, k)
    in_indices_target = in_indices[i]  # ground-truth membership, shape (n,)

    # `stat_shadow` : statistics of the shadow models, with shape (num_shadows, n, k).
    stat_shadow = np.array(stat[:i] + stat[i + 1 :])

    # `in_indices_shadow` membership of the shadow models, with shape (num_shadows, n)
    in_indices_shadow = np.array(in_indices[:i] + in_indices[i + 1 :])

    # stat_in[j] (resp. stat_out[j]) is a (m, k) array, m the number of shadow models
    # trained with (resp. without) the j-th example,
    # k being the number of augmentations (2 in our case)

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
        f"Average score of in sample is {round(scores[in_indices_target].mean(),4)}",
        "\n",
        f"Average score of sample is {round(scores[~in_indices_target].mean(),4)}",
    )

    return scores


def _get_metric_in_out_target(metrics, in_indices, i):

    metrics_in, metrics_out, metrics_target = [], [], []

    for metric in metrics:

        metric_target = metric[i, :, :]  # statistics of target model, shape (n, k1, k2)

        # `stat_shadow` : shadow models stats, shape (num_shadows, n, k1, k2).
        metric_shadow = np.concatenate([metric[:i, :, :], metric[i + 1 :, :, :]])

        # `in_indices_shadow` : shadow models membership info, shape (num_shadows, n)
        in_indices_shadow = np.array(in_indices[:i] + in_indices[i + 1 :])

        # stat_in[j] (resp. stat_out[j]) is a (m, k1, k2) array,
        # m the number of shadow models trained with (resp. without) the j-th example,
        # and k1, k2 being the dimensions of the metric
        metric_in = [
            metric_shadow[in_indices_shadow[:, j]]
            for j in range(in_indices[i].shape[0])
        ]
        metric_out = [
            metric_shadow[~in_indices_shadow[:, j]]
            for j in range(in_indices[i].shape[0])
        ]

        metrics_in.append(metric_in)
        metrics_out.append(metric_out)
        metrics_target.append(metric_target)

    return metrics_in, metrics_out, metrics_target


def _run_param_attack(metrics, in_indices, i):

    avg_scores = []
    avg_wght_scores = []

    metrics_in, metrics_out, metrics_target = _get_metric_in_out_target(
        metrics, in_indices, i
    )

    # Compute the scores and use them for MIA

    for idx in range(len(metrics_target)):

        score, weights = amia.compute_score_param_attack(
            metrics_target[idx], metrics_in[idx], metrics_out[idx], fix_variance=True
        )  # That is where the attack properly happens

        # Average score

        avg_score = score.mean(axis=1)
        avg_scores.append(avg_score)

        # Linearly weighted scores

        avg_wght_score = np.average(score, 1, weights)
        avg_wght_scores.append(avg_wght_score)

    print(
        "Average score of in sample is",
        f"{round(avg_score[in_indices[i]].mean(),4)}",
        "\n",
        "Average score of out sample is",
        f"{round(avg_score[~in_indices[i]].mean(),4)}",
    )

    return avg_scores, avg_wght_scores


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

    select_pt = []

    if _DATASET.value == "MNIST":
        x, y = utils.load_mnist()
    elif _DATASET.value == "CIFAR10":
        x, y = utils.load_cifar10()
    elif _DATASET.value == "BLOB":
        x, y = utils.load_blobs(NB_CLASSES)
    elif _DATASET.value == "SBLOB&OUTLIER":
        x, y, select_pt = utils.load_blobs_outliers_data(
            NB_CLASSES, _SEED.value, _OUTLIER_POLICY.value
        )
    elif _DATASET.value == "SBLOB&DIRAC":
        x, y, select_pt = utils.load_blobs_plus_dirac_data(NB_CLASSES, _SEED.value)
    elif _DATASET.value == "SPARSE":
        x, y = utils.load_sparse(NB_CLASSES, _SEED.value)
    else:
        raise Exception("Unauthorized value for data")

    n = x.shape[0]

    # Shadow models

    paths = []
    in_indices_shadow, params_shadow = [], []
    nb_cols = _NUM_SHADOWS.value

    # Define model path

    for i in range(1, _NUM_SHADOWS.value + 1):

        model_path = os.path.join(EXP_PATH_BASE, f"Models/model{i}.h5")
        paths.append(model_path)

    # Load stats if exists

    if sum([os.path.exists(p) for p in paths]) == nb_cols:

        stat_shadow = list(
            np.load(f"{EXP_PATH_BASE}/IndicesAndStats/stat_shadows.npy").astype(float).T
        )
        in_indices_shadow = list(
            np.load(f"{EXP_PATH_BASE}/IndicesAndStats/indices_shadows.npy")
            .astype(bool)
            .T
        )
        filenames_saved_layers = [
            filename
            for filename in os.listdir(f"{EXP_PATH_BASE}/IndicesAndStats/")
            if filename.startswith("params_shadow_layer")
        ]
        filenames_saved_layers.sort()

        for filename in filenames_saved_layers:
            params_shadow.append(
                np.load(f"{EXP_PATH_BASE}/IndicesAndStats/{filename}").astype(float)
            )

        print("Loaded previously run stats from shadow models")

        model = utils.make_model(
            model=_MODEL.value,
            data=_DATASET.value,
            flip=_FLIP_IMG.value,
            nb_classes=NB_CLASSES,
        )
        model_name = "model1"

        test_acc, train_acc = utils.evaluate_model(
            x, y, in_indices_shadow[0], model, EXP_PATH_BASE, model_name
        )

        print(
            f"Test accuracy of first fully trained shadow model is {test_acc:.4f},\
             train accuracy is {train_acc:.4f}"
        )

    # Else train shadow models

    else:

        for i in range(1, _NUM_SHADOWS.value + 1):

            in_indices_shadow.append(np.random.binomial(1, 0.5, n).astype(bool))

        if len(select_pt) != 0:
            in_indices_shadow[0][select_pt] = True  # forcing modified point in

        stat_shadow, _, params_shadow = _get_or_train_model(
            x,
            y,
            _NUM_SHADOWS.value,
            _EPOCHS.value,
            _BATCH.value,
            in_indices_shadow,
            seed,
        )

        np.save(
            f"{EXP_PATH_BASE}/IndicesAndStats/indices_shadows",
            np.array(in_indices_shadow).T,
        )
        np.save(
            f"{EXP_PATH_BASE}/IndicesAndStats/stat_shadows", np.array(stat_shadow).T
        )
        for idx, layer in enumerate(params_shadow):
            np.save(
                f"{EXP_PATH_BASE}/IndicesAndStats/params_shadow_layer{idx}",
                np.array(layer),
            )

    # Now we do the MIA for all versions of the target model

    scores, fprs, tprs = [], [], []

    print("Target model is shadow model 1")

    # NONEED to Pop out points with no in or out shadow model

    # Record liRA attack

    loss_score = _run_loss_attack(stat_shadow, in_indices_shadow, 0)

    fpr, tpr = utils.get_auc(in_indices_shadow[0], loss_score)

    scores.append(loss_score)
    fprs.append(fpr)
    tprs.append(tpr)

    # Record my attack

    param_avg_scores, param_avg_wght_scores = _run_param_attack(
        params_shadow, in_indices_shadow, 0
    )

    for param_scores in [param_avg_scores, param_avg_wght_scores]:
        for score in param_scores:
            fpr, tpr = utils.get_auc(in_indices_shadow[0], score)
            scores.append(score)
            print(
                f"The scores for points with diracs are \
                    {score[select_pt].mean().round(3)}"
            )
            fprs.append(fpr)
            tprs.append(tpr)

    # Print attack results on POIs

    if len(select_pt) == 0:

        select_pt = np.argpartition(scores[-1], 10)[:10]

    for i, score in enumerate(scores):
        print(
            f"The avg score ({i}) for points with diracs are \
                {score[select_pt].mean().round(3)}"
        )

    # Compare in and out

    metrics_in, metrics_out, metrics_target = _get_metric_in_out_target(
        params_shadow, in_indices_shadow, 0
    )

    for pt in select_pt:

        plot_dict = {"marker": ".", "linestyle": "None", "alpha": 0.4}

        metrics_in_mean = metrics_in[-1][pt][:, :, 0].mean(axis=0)
        metrics_out_mean = metrics_out[-1][pt][:, :, 0].mean(axis=0)

        plt.plot(metrics_out_mean, label="out", **plot_dict)
        plt.plot(metrics_in_mean, label="in", **plot_dict)

        plot_dict["marker"] = "x"

        plt.plot(metrics_target[-1][:, 0], color="red", label="target", **plot_dict)
        plt.legend()
        plt.title(f"Point coordinates : {x[pt]}")
        plt.savefig(f"{EXP_PATH_BASE}/Figures/point{pt}_in_out_{_EXPE_TYPE.value}.png")
        plt.cla()

        plot_dict["marker"] = "."

        metrics_diff = metrics_out_mean - metrics_in_mean
        plt.plot(metrics_diff, label="diff", **plot_dict)
        plt.title(f"Point number : {pt}")
        plt.savefig(f"{EXP_PATH_BASE}/Figures/point{pt}_diff_grad.png")
        plt.cla()

    # Param attack

    # Visualize ROC-AUC

    n = len(fprs)
    curve_names = [
        "LiRA - fully trained model",
        "Layer 1 avg attack",
        "Layer 1 avg weighted attack",
        "Layer 1 avg exp weighted attack",
    ]

    custom_colors = ["red"] + list(iter(cm.Blues(np.linspace(0.3, 1, n - 1))))  # \/3
    # + list(iter(cm.Greens(np.linspace(0.3, 1, int((n-1)/3))))) \
    # + list(iter(cm.Blues(np.linspace(0.3, 1, int((n-1)/3)))))

    fig_name = f"{EXP_PATH_BASE}/Figures/target_AUC_ROC.png"
    utils.plot_auc(
        fprs,
        tprs,
        fig_name,
        curve_names,
        custom_colors,
        title="ROC-AUC curve for different attacks on simple MLP",
    )

    plt.close()

    for idx, score in enumerate(scores):
        plt.scatter(
            y=score,
            x=np.linspace(0, score.shape[0], score.shape[0]).astype(int),
            c=in_indices_shadow[0].astype(int),
            cmap="bwr",
            marker=".",
        )
        plt.title(
            f"{curve_names[idx]} : diracs avg score {score[select_pt].mean().round(3)}"
        )
        plt.savefig(f"{EXP_PATH_BASE}/Figures/POI_scores_{idx}.png")

    plt.close()


if __name__ == "__main__":
    app.run(main)
