# -*- coding: utf-8 -*-
"""
합성 데이터 데모: 5-fold GBDT 학습 → CV 평가 → 외부(확증) 검증.

Run:
  python data/make_synthetic.py   # 먼저 합성 데이터 생성
  python demo_train.py

Writes:
  ml_result/use_all/GBDT/{fold:03d}.pkl   -- eval_cv_average()가 읽는 포맷
  external_fold_roc_curves.png            -- 외부 검증 fold별 ROC
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from common import ConfigManager, check_directoty, save_obj
from ml_manager import DataPurpose, TabularDataManager, eval_cv_average
from threshold_utils import threshold_table, choose_threshold_by_youden


def train_folds():
    cfg = ConfigManager("./config/config.ini").load()
    check_directoty("./ml_result/use_all/GBDT")

    for fold in range(5):
        cfg["train_folds"] = cfg[f"train_folds.{fold}"]
        cfg["val_folds"] = cfg[f"val_folds.{fold}"]
        cfg["test_folds"] = cfg[f"test_folds.{fold}"]

        x_tr, y_tr, _ = TabularDataManager(DataPurpose.TRAIN, cfg).load_data()
        x_val, y_val, _ = TabularDataManager(DataPurpose.VAL, cfg).load_data()

        model = GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)
        model.fit(x_tr.astype(float), y_tr.astype(int))

        p_val = model.predict_proba(x_val.astype(float))[:, 1]
        thr, _ = choose_threshold_by_youden(threshold_table(y_val.astype(int), p_val))

        save_obj(f"./ml_result/use_all/GBDT/{fold:03d}.pkl",
                 {"model": model, "scaler": None, "pca": None,
                  "params": {"thr": float(thr)}})
        print(f"[Fold {fold}] trained GBDT: train n={len(y_tr)} "
              f"(pos={int(y_tr.sum())}), thr={thr:.3f}")


if __name__ == "__main__":
    train_folds()
    print()
    eval_cv_average()
    print()
    from external_eval import evaluate_external
    evaluate_external("./data/csv/external.csv")
