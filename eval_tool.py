import os

# import tensorflow as tf
import warnings
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

"""

class MatthewsCorrelationCoefficient(tf.keras.metrics.Metric):
    def __init__(self, name="matthews_correlation_coefficient", **kwargs):
        super(MatthewsCorrelationCoefficient, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.tn = self.add_weight(name="tn", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true = tf.cast(y_true, tf.bool)
        # y_pred = tf.cast(y_pred, tf.bool)
        y_true = tf.cast(tf.squeeze(y_true), tf.bool)
        y_pred = tf.cast(tf.squeeze(tf.round(y_pred)), tf.bool)

        tp = tf.cast(tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True)), tf.float32)
        tn = tf.cast(tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, False)), tf.float32)
        fp = tf.cast(tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True)), tf.float32)
        fn = tf.cast(tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False)), tf.float32)

        self.tp.assign_add(tf.reduce_sum(tp))
        self.tn.assign_add(tf.reduce_sum(tn))
        self.fp.assign_add(tf.reduce_sum(fp))
        self.fn.assign_add(tf.reduce_sum(fn))

    def result(self):
        numerator = (self.tp * self.tn) - (self.fp * self.fn)
        denominator = tf.sqrt((self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn))
        return numerator / denominator
"""


def matthews_correlation(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    warnings.filterwarnings("default", category=RuntimeWarning)
    return mcc


def _unique_path(path: str) -> str:
    """
    If *path* already exists, append a numeric suffix before the extension
    (e.g. 'roc_curve.png' -> 'roc_curve_1.png', 'roc_curve_2.png' …)
    and return the first unused filename.
    """
    base, ext = os.path.splitext(path)
    counter = 1
    candidate = path
    while os.path.exists(candidate):
        candidate = f"{base}_{counter}{ext}"
        counter += 1
    return candidate


def binary_evaluate(y_real, y_pred, y_prob, save_path: str = "./roc_curve.png"):
    """
    Evaluate binary-classification metrics and save an ROC curve plot.
    Uses *y_prob* (continuous scores) for the ROC curve/AUC, while *y_pred*
    is still used for confusion-matrix based point metrics.

    Returns
    -------
    accuracy, sensitivity, specificity, ppv, npv, f1_score : float
    """
    # ---- confusion-matrix counts (@ threshold 0.5) --------------------------
    tp = tn = fp = fn = 0
    for actual, predict in zip(y_real, y_pred):
        if predict >= 0.5 and actual >= 1:
            tp += 1
        elif predict < 0.5 and actual == 0:
            tn += 1
        elif predict >= 0.5 and actual == 0:
            fp += 1
        elif predict < 0.5 and actual >= 1:
            fn += 1

    # ---- point metrics ------------------------------------------------------
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0
    ppv = tp / (tp + fp) if (tp + fp) else 0
    npv = tn / (tn + fn) if (tn + fn) else 0
    f1_score = 2 * ppv * sensitivity / (ppv + sensitivity) if (ppv + sensitivity) else 0

    # ---- ROC curve & AUC (using continuous y_prob) --------------------------
    fpr, tpr, _ = roc_curve(y_real, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.tight_layout()

    final_path = _unique_path(save_path)
    # plt.savefig(final_path, dpi=300)
    plt.close()

    return accuracy, sensitivity, specificity, ppv, npv, f1_score


def export_perform(output_path, info_dict, perform_lst, verbose=False):
    key_lst = list(info_dict.keys())
    if not os.path.exists(output_path):
        if len(key_lst) > 0:
            head = ",".join(key_lst)
            head = f"{head}, accuracy, sensitivity, specificity, ppv, npv, f1_score\n"
        else:
            head = f"accuracy, sensitivity, specificity, ppv, npv, f1_score\n"

        with open(output_path, "w") as f:
            f.write(head)

    info = ""
    for key in key_lst:
        val = info_dict[key]
        if len(info) == 0:
            info = str(val)
        else:
            info = f"{info},{val}"

    line = ""
    for val in perform_lst:
        if len(line) == 0:
            line = str(val)
        else:
            line = f"{line},{val}"

    with open(output_path, "a") as f:
        f.write(f"{info},{line}\n")
    print(info)
    if verbose:
        line = info
        for val in perform_lst:
            line = f"{line},{val=:.3f}"
        print(line)
