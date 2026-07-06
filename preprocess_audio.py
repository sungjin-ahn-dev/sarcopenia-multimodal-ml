import csv
import pandas as pd
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import parselmouth
import opensmile
from scipy import signal


def fill_nan_with_mean(data):
    # numpy 배열로 변환
    arr = np.array(data)

    # 각 행의 평균값을 계산하여 nan 값을 대체
    column_means = np.nanmean(arr, axis=0)  # 각 열의 평균값 계산
    inds = np.where(np.isnan(arr))  # nan 값의 인덱스 찾기
    arr[inds] = np.take(column_means, inds[1])  # 해당 열의 평균값으로 nan 대체

    return arr


def f_low(y, sr):
    b, a = signal.butter(10, 4096 / (sr / 2), btype="lowpass")
    yf = signal.lfilter(b, a, y)
    return yf


def normalize_signal(signal):
    return signal / np.mean(np.abs(signal))


def moving_average(signal, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(signal, window, "same")


def process_file(row, audio_path):
    file_list = os.listdir(audio_path)
    file_flags = 0
    for file_indiv in file_list:
        if row["PDT"] in file_indiv:
            # load audio file
            print(audio_path, file_indiv)
            y, sr = librosa.load(os.path.join(audio_path, file_indiv))
            # y = f_low(y, sr)
            # y = moving_average(y, 5)
            # y = normalize_signal(y)

            """
            # audio noise reduction
            D = librosa.stft(y)

            magnitude, _ = librosa.magphase(D)
            # 목소리와 배경 분리를 위한 마스크 생성
            # 예를 들어, median filter를 사용하여 노이즈 제거
            S_filter = librosa.decompose.nn_filter(
                magnitude,
                aggregate=np.median,
                metric="cosine",
                width=int(librosa.time_to_frames(0.4, sr=sr)),
            )
            S_filter = np.minimum(magnitude, S_filter)

            # 목소리 마스크 생성
            margin_i, margin_v = 2, 10
            power = 2

            mask_i = librosa.util.softmask(
                S_filter, margin_i * (magnitude - S_filter), power=power
            )

            mask_v = librosa.util.softmask(
                magnitude - S_filter, margin_v * S_filter, power=power
            )

            # 목소리와 배경 추출
            S_foreground = mask_v * D
            S_background = mask_i * D

            # 추출된 목소리 신호를 시간 도메인으로 변환
            y_foreground = librosa.istft(S_foreground)
            y_background = librosa.istft(S_background)

            y = y_background
            """
            # silence trimming
            non_silent_intervals = librosa.effects.split(y, top_db=30)
            y_trimmed = []
            for start, end in non_silent_intervals:
                y_trimmed.append(y[start:end])
            y_trimmed = np.concatenate(y_trimmed)

            # voice normalize
            # rms = np.sqrt(np.mean(y_trimmed**2))
            # y_trimmed = y_trimmed / rms * 0.1  # RMS를 0.1로 정규화

            # not preprocessed
            # y_trimmed = y

            # mel spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(y=y_trimmed, sr=sr)
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

            # save mel spectrogram
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(
                mel_spectrogram_db, sr=sr, x_axis="time", y_axis="mel"
            )
            plt.colorbar(format="%+2.0f dB")
            plt.title(
                "PDT: " + row["PDT"] + " Sarcopenia: " + str(row["Sarcopenia_label"])
            )
            plt.savefig(
                audio_path + "/mel/" + row["PDT"] + "_" + str(row["Sarcopenia_label"])
            )
            plt.close()

            # sound featrue extraction using parselmouth
            sound = parselmouth.Sound(y_trimmed, sampling_frequency=sr)

            # extract pitch
            pitch = sound.to_pitch()
            pitch_values = pitch.selected_array["frequency"]

            # extract intensity
            intensity = sound.to_intensity()
            intensity_values = intensity.values.T[0]

            # extract formants
            formants = sound.to_formant_burg()
            num_points = pitch.n_frames
            formant_values = np.array(
                [
                    [formants.get_value_at_time(i, t) for i in range(1, 4)]
                    for t in pitch.xs()
                ]
            )

            # 조음점(Jitter, Shimmer) 추출
            # jitter = sound.to_jitter()
            # shimmer = sound.to_shimmer()

            # 기타 통계 값 계산
            pitch_mean = np.nanmean(pitch_values[pitch_values != 0])  # 0 제외 평균
            pitch_std = np.nanstd(pitch_values[pitch_values != 0])  # 0 제외 표준 편차

            intensity_mean = np.nanmean(intensity_values)
            intensity_std = np.nanstd(intensity_values)

            formants = sound.to_formant_burg()
            num_points = pitch.n_frames
            formant_values = np.array(
                [
                    [formants.get_value_at_time(i, t) for i in range(1, 4)]
                    for t in pitch.xs()
                ]
            )

            formant_frequencies_mean = np.nanmean(formant_values)
            formant_frequencies_std = np.nanstd(formant_values)

            mfccs_mean = np.mean(librosa.feature.mfcc(y=y_trimmed, sr=sr))
            chroma_mean = np.mean(librosa.feature.chroma_stft(y=y_trimmed, sr=sr))
            contrast_mean = np.mean(
                librosa.feature.spectral_contrast(y=y_trimmed, sr=sr)
            )
            rolloff_mean = np.mean(librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr))
            bandwidth_mean = np.mean(
                librosa.feature.spectral_bandwidth(y=y_trimmed, sr=sr)
            )
            centroid_mean = np.mean(
                librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)
            )
            flatness_mean = np.mean(librosa.feature.spectral_flatness(y=y_trimmed))

            # 결과 저장
            return [
                mfccs_mean,
                chroma_mean,
                contrast_mean,
                rolloff_mean,
                bandwidth_mean,
                centroid_mean,
                flatness_mean,
                pitch_mean,
                pitch_std,
                intensity_mean,
                intensity_std,
                num_points,
                formant_frequencies_mean,
                formant_frequencies_std,
            ]

            file_flags = 1

    if file_flags == 0:
        return [np.nan] * 14


if __name__ == "__main__":

    file_name = "data/csv/tabular_data.csv"
    df = pd.read_csv(file_name)

    task_list = ["1", "2", "3", "4", "5", "6"]
    for task in task_list:
        audio_path = f"data/speech/record/{task}"
        processed_data = []

        df_part = df[["PDT", "Sarcopenia_label"]]
        df_part["audio_path"] = audio_path
        df_part["features"] = df_part.apply(
            lambda row: process_file(row, row["audio_path"]), axis=1
        )

        test = df_part["features"].tolist()

        df_feature = fill_nan_with_mean(df_part["features"].tolist()).tolist()

        df_speech = pd.DataFrame(
            df_feature,
            columns=[
                f"MFCCs_{task}",
                f"Chroma_{task}",
                f"Contrast_{task}",
                f"Rolloff_{task}",
                f"Bandwidth_{task}",
                f"Centroid_{task}",
                f"Flatness_{task}",
                f"Pitch_mean_{task}",
                f"Pitch_std_{task}",
                f"Intensity_mean_{task}",
                f"Intensity_std_{task}",
                f"Num_points",
                f"formant_frequencies_mean_{task}",
                f"formant_frequencies_std_{task}",
            ],
        )
        df_merge = pd.concat([df_part, df_speech], axis=1)
        df_merge.to_csv(f"results/speech_task_{task}.csv", index=False)
