# threshold_utils.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def threshold_table(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    fpr, tpr, thr = roc_curve(y_true, y_score)
    rows = []
    for t in thr:
        y_pred = (y_score >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        acc = accuracy_score(y_true, y_pred)
        sen = recall_score(y_true, y_pred, zero_division=0)
        spe = tn / (tn + fp) if (tn+fp) > 0 else 0.0
        ppv = precision_score(y_true, y_pred, zero_division=0)
        npv = tn / (tn + fn) if (tn+fn) > 0 else 0.0
        f1  = f1_score(y_true, y_pred, zero_division=0)
        youden = sen + spe - 1.0
        rows.append({
            "threshold": float(t),
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
            "Acc": acc, "Sen": sen, "Spe": spe, "PPV": ppv, "NPV": npv, "F1": f1, "YoudenJ": youden
        })
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)

def choose_threshold_by_youden(df):
    best = df.loc[df["YoudenJ"].idxmax()]
    return float(best["threshold"]), best.to_dict()

def choose_threshold_for_target_sen(df, target_sen=0.90):
    cand = df[df["Sen"] >= target_sen].copy()
    if len(cand) == 0:
        best = df.loc[df["Sen"].idxmax()]
    else:
        best = cand.loc[cand["Spe"].idxmax()]
    return float(best["threshold"]), best.to_dict()

def plot_roc(y_true, y_score, chosen_threshold=None, out_path=None, title="ROC"):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("1 - Specificity (FPR)")
    plt.ylabel("Sensitivity (TPR)")
    plt.title(title)
    plt.legend(loc="lower right")
    if chosen_threshold is not None:
        import numpy as np
        idx = int(np.argmin(np.abs(thr - chosen_threshold)))
        plt.scatter([fpr[idx]], [tpr[idx]], marker='o')
    if out_path:
        import os
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()
