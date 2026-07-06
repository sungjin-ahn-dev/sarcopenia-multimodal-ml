# -*- coding: utf-8 -*-
# 실제 피험자 데이터는 repo에 안 올림. 대신 같은 스키마의 합성 데이터를 찍어내서
# 학습/평가 파이프라인이 그 위에서 돌아가게 함.
# 양성군은 고령 + 저SMI + 저악력 쪽으로 분포 shift, 터치/스와이프에도 약한 신호만.
# datasetA.csv = 5-fold 용, external.csv = 외부검증 용.
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "csv"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(7)

FEATURES = ["SEX", "Age", "Weight", "BMI", "SMI", "IBgrip_MAX",
            "swipe_horizontal_rms_distance",
            "touch_horizontal_one_press_consistency",
            "touch_horizontal_one_left_mean",
            "touch_horizontal_two_balance_ratio"]


def make_cohort(n, pos_rate, sites, prefix):
    n_pos = int(round(n * pos_rate))
    labels = np.array([1] * n_pos + [0] * (n - n_pos))
    RNG.shuffle(labels)

    sex = RNG.integers(0, 2, n)  # 0=남, 1=여
    # 양성군: 고령, 저체중/저BMI/저SMI/저악력 경향 + 잡음
    age = np.clip(RNG.normal(71 + 6 * labels, 7), 55, 92).round(0)
    bmi = np.clip(RNG.normal(24.5 - 2.2 * labels, 2.8), 16, 35).round(1)
    height = np.where(sex == 0, RNG.normal(167, 6, n), RNG.normal(154, 6, n))
    weight = np.clip(bmi * (height / 100) ** 2 + RNG.normal(0, 2, n), 35, 95).round(1)
    smi = np.clip(RNG.normal(7.3 - 0.9 * labels - 0.7 * sex, 0.75), 4.2, 9.5).round(2)
    grip = np.clip(RNG.normal(30 - 6.5 * labels - 8 * sex, 4.5), 8, 48).round(1)

    swipe_rms = np.clip(RNG.normal(0.42 + 0.05 * labels, 0.09, n), 0.1, 0.9).round(4)
    press_consist = np.clip(RNG.normal(0.71 - 0.06 * labels, 0.11, n), 0.2, 0.98).round(4)
    left_mean = np.clip(RNG.normal(0.48 + 0.02 * labels, 0.07, n), 0.2, 0.8).round(4)
    balance = np.clip(RNG.normal(1.02 + 0.08 * labels, 0.16, n), 0.5, 1.8).round(4)

    site = RNG.choice(sites, n)
    ids = [f"{s}_{prefix}{i:03d}" for i, s in enumerate(site, start=1)]

    df = pd.DataFrame({
        "sub no.": range(1, n + 1),
        "PDT": ids,
        "Sarcopenia_label": labels,
        "fold": -1,
        "SEX": sex, "Age": age, "Weight": weight, "BMI": bmi,
        "SMI": smi, "IBgrip_MAX": grip,
        "swipe_horizontal_rms_distance": swipe_rms,
        "touch_horizontal_one_press_consistency": press_consist,
        "touch_horizontal_one_left_mean": left_mean,
        "touch_horizontal_two_balance_ratio": balance,
    })
    return df


def assign_folds(df, k=5):
    """클래스 층화 5-fold 배정."""
    for label in (0, 1):
        idx = df.index[df["Sarcopenia_label"] == label].to_numpy()
        RNG.shuffle(idx)
        for i, ix in enumerate(idx):
            df.loc[ix, "fold"] = i % k
    return df


def main():
    ds_a = assign_folds(make_cohort(283, 44 / 283, ["SITE_A", "SITE_B"], "A"))
    ds_a.to_csv(OUT / "datasetA.csv", index=False)
    print(f"wrote datasetA.csv: {len(ds_a)} rows, pos={ds_a.Sarcopenia_label.sum()}, "
          f"folds={sorted(ds_a.fold.unique())}")

    ext = make_cohort(279, 48 / 279, ["SITE_A", "SITE_C"], "Z").drop(columns=["fold"])
    ext.to_csv(OUT / "external.csv", index=False)
    print(f"wrote external.csv: {len(ext)} rows, pos={ext.Sarcopenia_label.sum()}")


if __name__ == "__main__":
    main()
