import numpy as np
import pandas as pd

SAMP_SIZE = 20


def export_fft_features(swipe_path, touch_path, fft_path):
    swipe_df = pd.read_csv(swipe_path)
    record_dict = {"hor_swipe": dict(), "ver_swipe": dict()}
    for _, row in swipe_df.iterrows():
        dist, hor, uid = row.iloc[3], row.iloc[5], row.iloc[6]
        col_name = "hor_swipe" if hor else "ver_swipe"
        id2val = record_dict[col_name]
        if uid not in id2val:
            id2val[uid] = []
        id2val[uid].append(dist)

    touch_df = pd.read_csv(touch_path)
    record_dict["hor_one_touch"] = dict()
    record_dict["ver_one_touch"] = dict()
    record_dict["hor_two_touch"] = dict()

    for _, row in touch_df.iterrows():
        left, right, _, hor, one, uid = row.iloc[3:]
        if hor and one:
            col_name = "hor_one_touch"
        elif one:
            col_name = "ver_one_touch"
        elif hor:
            col_name = "hor_two_touch"
        id2val = record_dict[col_name]
        cnt = left + right
        if uid not in id2val:
            id2val[uid] = []
        id2val[uid].append(cnt)

    result_dict = {"uid": []}
    freqs = np.fft.rfftfreq(SAMP_SIZE)
    prd_lst = np.where(freqs != 0.0, 1.0 / freqs, 0.0)
    uid_lst = None
    for col_name, id2val in record_dict.items():
        print(f"{col_name=}, length={len(id2val)}")

        if uid_lst is None:
            uid_lst = list(id2val.keys())

        for i in range(1, len(prd_lst)):
            name = f"{col_name}_amp_{prd_lst[i]:.1f}"
            result_dict[name] = []

    for uid in uid_lst:
        result_dict["uid"].append(uid)
        for col_name, id2val in record_dict.items():
            val_lst = id2val[uid]
            if len(val_lst) > SAMP_SIZE:
                val_lst = val_lst[-SAMP_SIZE:]

            amp_lst = np.fft.rfft(val_lst - np.mean(val_lst))
            for i in range(1, len(prd_lst)):
                name = f"{col_name}_amp_{prd_lst[i]:.1f}"
                result_dict[name].append(np.abs(amp_lst[i]))

    pd.DataFrame(result_dict).to_csv(fft_path, index=False)
    print("done!")


if __name__ == "__main__":
    print("statistic with touch and swipe test")
    swipe_path = "./inputs/swipe_record.csv"
    touch_path = "./inputs/touch_record.csv"
    fft_path = "./fft_features.csv"
    export_fft_features(swipe_path, touch_path, fft_path)
