import traceback
import sys

import numpy as np
import pandas as pd  # ★ 추가
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from enum import Enum
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import make_scorer, f1_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix

from common import *
from eval_tool import *

def binom_ci(k, n, alpha=0.05):
    """
    k: 성공 횟수 (예: TP 또는 TN)
    n: 시도 횟수 (예: TP+FN 또는 TN+FP)
    반환: (low, high)
    """
    if n == 0:
        return (np.nan, np.nan)
    try:
        from scipy.stats import beta
        low  = beta.ppf(alpha/2, k,   n-k+1)
        high = beta.ppf(1-alpha/2, k+1, n-k)
        return (low, high)
    except Exception:
        # Wilson fallback
        p = k / n
        z = 1.959963984540054  # 95%
        denom  = 1 + (z**2)/n
        center = p + (z**2)/(2*n)
        half   = z*np.sqrt((p*(1-p) + (z**2)/(4*n))/n)
        low  = (center - half)/denom
        high = (center + half)/denom
        return (max(0.0, low), min(1.0, high))

# ===== 임상 기준 충족 여부 문자열 =====
def pass_fail(lower_bound, target):
    return "PASS" if (not np.isnan(lower_bound) and lower_bound >= target) else "FAIL"


class DataPurpose(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    ALL = "all"


class OverSampMethods(Enum):
    # NONE = "none"
    # SIMP = "simple"
    BS = "bSMOTE"


class ML_Models(Enum):
    LR = "Logistic regression"
    NB = "Naive bayes"
    KNN = "K-Nearest Neighbor"
    DT = "Decision Tree"
    SVM = "Support vector machine"
    CB = "CatBoost"
    GBDT = "Gradient Boosting"
    LGBM = "LightGBM"


class VotingTypes(Enum):
    Hard = "hard"
    Soft = "soft"


# =========================
# ★ 결측/문자 안정화 (전면 수정)
# =========================
def fill_data(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace("", np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return X


def get_model(model_type, params=None):
    # 일부 기본 파라미터를 불균형 대응에 유리하게 셋업
    if model_type == ML_Models.LR:
        base = dict(max_iter=200, class_weight="balanced", solver="liblinear")
        if params: base.update(params)
        return LogisticRegression(**base)
    elif model_type == ML_Models.NB:
        return GaussianNB(**(params or {}))
    elif model_type == ML_Models.KNN:
        return KNeighborsClassifier(**(params or {}))
    elif model_type == ML_Models.DT:
        base = dict(class_weight="balanced")
        if params: base.update(params)
        return DecisionTreeClassifier(**base)
    elif model_type == ML_Models.SVM:
        base = dict(probability=True, class_weight="balanced", kernel="linear")
        if params: base.update(params)
        return SVC(**base)
    elif model_type == ML_Models.CB:
        # ★ CatBoost 불균형 가중치
        base = dict(
            depth=6, iterations=800, learning_rate=0.05,
            eval_metric="AUC", loss_function="Logloss",
            verbose=False, random_state=42
        )
        if params: base.update(params)
        return CatBoostClassifier(**base)
    elif model_type == ML_Models.GBDT:
        # scikit GBDT는 class_weight 없음
        return GradientBoostingClassifier(**(params or {}))
    elif model_type == ML_Models.LGBM:
        # ★ LightGBM 불균형 옵션
        base = dict(
            n_estimators=500, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            objective="binary", is_unbalance=True, random_state=42
        )
        if params: base.update(params)
        return LGBMClassifier(**base)
    else:
        raise ValueError("Invalid model type")


def get_param_grid(model_type):
    if model_type == ML_Models.LR:
        param_grid = {
            "C": np.logspace(-2, 2, 5),   # 살짝 축소
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"],      # liblinear로 고정
        }
    elif model_type == ML_Models.NB:
        param_grid = {"var_smoothing": np.logspace(-10, -1, 10)}
    elif model_type == ML_Models.KNN:
        param_grid = {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]}
    elif model_type == ML_Models.DT:
        param_grid = {"max_depth": [3, 5, 7, 9], "criterion": ["gini", "entropy"]}
    elif model_type == ML_Models.SVM:
        param_grid = {"C": [0.1, 1, 10]}
    elif model_type == ML_Models.CB:
        param_grid = {
            "learning_rate": [0.03, 0.05, 0.1],
            "depth": [4, 6, 8],
            "iterations": [400, 800, 1200],
            "subsample": [0.8, 1.0],
            "colsample_bylevel": [0.8, 1.0],
            # class_weights는 데이터 로딩 후 동적으로 부여 가능
        }
    elif model_type == ML_Models.GBDT:
        param_grid = {"n_estimators": [200, 500], "learning_rate": [0.03, 0.1]}
    elif model_type == ML_Models.LGBM:
        param_grid = {
            "n_estimators": [300, 600],
            "learning_rate": [0.03, 0.07],
            "num_leaves": [31, 63],
        }
    else:
        raise ValueError("Invalid model type")
    return param_grid


def find_best_pca(x_train, threshold=0.90):
    max_rat = -1
    best = None
    for n_components in range(1, x_train.shape[1]):
        pca = PCA(n_components=n_components)
        pca.fit(x_train)
        exo_rat = np.sum(pca.explained_variance_ratio_)
        if exo_rat >= max_rat:
            max_rat = exo_rat
            best = [n_components, pca]
        if exo_rat > threshold:
            return pca
    _, pca = best
    return pca


# =========================
# ★ SMOTE 안전화 (k_neighbors 동적)
# =========================
def over_sampling(x, y, os_method):
    if hasattr(OverSampMethods, "NONE") and os_method == OverSampMethods.NONE:
        return x, y
    if hasattr(OverSampMethods, "SIMP") and os_method == OverSampMethods.SIMP:
        unique, counts = np.unique(y, return_counts=True)
        minor_class = unique[np.argmin(counts)]
        major_count = np.max(counts)
        minor_count = np.min(counts)
        if minor_count <= major_count / 2:
            sampled_x = x.copy()
            sampled_y = y.copy()
            while np.sum(sampled_y == minor_class) < major_count:
                sampled_indices = np.random.choice(
                    np.where(y == minor_class)[0],
                    size=major_count - np.sum(sampled_y == minor_class),
                    replace=True,
                )
                sampled_x = np.concatenate((sampled_x, x[sampled_indices]), axis=0)
                sampled_y = np.concatenate((sampled_y, y[sampled_indices]), axis=0)
            return sampled_x, sampled_y
        return x, y

    if hasattr(OverSampMethods, "BS") and os_method == OverSampMethods.BS:
        unique, counts = np.unique(y, return_counts=True)
        min_count = counts.min()
        k = max(1, min(5, min_count - 1))
        smote = SMOTE(k_neighbors=k, random_state=42)
        try:
            sampled_x, sampled_y = smote.fit_resample(x, y)
        except ValueError:
            return x, y
        return sampled_x, sampled_y

    # fallback (거의 타지 않음)
    return x, y


# =========================
# ★ Threshold 최적화 유틸
# =========================
def optimal_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    return thr[np.argmax(youden)]


class TabularDataManager:
    def __init__(self, data_purpose, config=None):
        if config is None:
            config = ConfigManager().load()
        tabular_data_path = config["tabular_data_path"]
        data_df = pd.read_csv(tabular_data_path)

        if data_purpose == DataPurpose.TRAIN:
            selected_folds = list(map(int, config["train_folds"].split(",")))
        elif data_purpose == DataPurpose.VAL:
            selected_folds = list(map(int, config["val_folds"].split(",")))
        elif data_purpose == DataPurpose.TEST:
            selected_folds = list(map(int, config["test_folds"].split(",")))
        else:
            selected_folds = None

        if data_purpose != DataPurpose.ALL:
            data_df = data_df[data_df["fold"].isin(selected_folds)]

        if str(config["select_top_column"]).isdigit():
            top_cnt = int(config["select_top_column"])
            if os.path.exists(config["fi_path"]):
                fi_df = pd.read_csv(config["fi_path"])
                sorted_fi_df = fi_df.sort_values(by="fi", ascending=False)
                top_fi_values = sorted_fi_df.head(top_cnt)["fi"].tolist()
                print(f"{top_cnt=}, {sum(top_fi_values)=:.1f}")
                top_variable_names = sorted_fi_df.head(top_cnt)["name"].tolist()
                selected_columns = list(data_df.columns[:5]) + top_variable_names
                data_df = data_df[selected_columns]

        self.data_df = data_df
        self.columns = list(data_df.columns[4:])
        self.id_lst = data_df["PDT"].values

    def load_data(self):
        data_df = fill_data(self.data_df)
        x_data = data_df.values[:, 4:]
        y_data = data_df["Sarcopenia_label"].values
        y_data_pos = y_data.copy()
        return x_data, y_data, y_data_pos


class ML_Manager:
    def __init__(self, config=None, version=None):
        if config is None:
            config = ConfigManager().load()
        self.config = config

        self.ml_result_dir = config["ml_result_dir"]
        self.version = version
        check_directoty(self.ml_result_dir)

        self.use_scaler = True if config["use_scaler"].lower() == "true" else False
        self.use_pca = True if config["use_pca"].lower() == "true" else False
        self.use_oversampling = True if config["use_oversampling"].lower() == "true" else False
        self.os_methods = OverSampMethods(config["os_methods"])
        self.x_train = None; self.y_train = None
        self.x_val = None;   self.y_val = None
        self.x_test = None;  self.y_test = None
        self.y_test_pos = None
        self.scaler = None;  self.pca = None
        self.id_lst = None
        self.load_data_all()

    def _is_tree_model(self, model_type):
        return model_type in [ML_Models.DT, ML_Models.GBDT, ML_Models.LGBM, ML_Models.CB]

    def load_data_all(self):
        train_dm = TabularDataManager(DataPurpose.TRAIN, self.config)
        val_dm   = TabularDataManager(DataPurpose.VAL, self.config)
        test_dm  = TabularDataManager(DataPurpose.TEST, self.config)
        self.id_lst = np.concatenate([test_dm.id_lst, val_dm.id_lst, train_dm.id_lst]).tolist()

        x_train, y_train, _ = train_dm.load_data()
        x_val,   y_val,   _ = val_dm.load_data()
        x_test,  y_test,  y_test_pos = test_dm.load_data()

        # 스케일러는 필요할 때만
        scaler = None
        if self.use_scaler:
            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_val   = scaler.transform(x_val)
            x_test  = scaler.transform(x_test)

        pca = None
        if self.use_pca:
            pca = find_best_pca(x_train)
            x_train = pca.transform(x_train)
            x_val   = pca.transform(x_val)
            x_test  = pca.transform(x_test)

        if self.use_oversampling:
            x_train, y_train = over_sampling(x_train, y_train, self.os_methods)

        self.x_train = x_train; self.y_train = y_train
        self.x_val   = x_val;   self.y_val   = y_val
        self.x_test  = x_test;  self.y_test  = y_test
        self.y_test_pos = y_test_pos
        self.scaler = scaler; self.pca = pca

    # =========================
    # ★ 확률 기반 스코어: AUC + F1(best-thr) + MCC
    # =========================
    def fit_model(self, model_type, params):
        x_train, y_train = self.x_train, self.y_train
        x_val,   y_val   = self.x_val,   self.y_val

        # LightGBM/CatBoost에 불균형 가중 동적 주입
        if model_type in [ML_Models.LGBM, ML_Models.CB]:
            neg = int((y_train == 0).sum())
            pos = int((y_train == 1).sum())
            ratio = max(1.0, neg / max(1, pos))
            if model_type == ML_Models.LGBM:
                if params is None: params = {}
                params = {**params, "is_unbalance": True}
            if model_type == ML_Models.CB:
                if params is None: params = {}
                params = {**params, "class_weights": [1.0, ratio]}

        model = get_model(model_type, params)
        model.fit(x_train, y_train)

        # 확률/결정함수
        if hasattr(model, "predict_proba"):
            y_val_prob = model.predict_proba(x_val)[:, 1]
        elif hasattr(model, "decision_function"):
            y_val_prob = model.decision_function(x_val)
            # decision_function은 scale이 다를 수 있으나 ROC/AUC엔 무관
        else:
            # 확률이 없으면 예측값으로 대체(권장X)
            y_val_prob = model.predict(x_val)

        thr = optimal_threshold(y_val, y_val_prob)
        y_val_pred = (y_val_prob >= thr).astype(int)

        mcc = matthews_correlation(y_val, y_val_pred)
        f1  = f1_score(y_val, y_val_pred)
        try:
            auc = roc_auc_score(y_val, y_val_prob)
        except Exception:
            auc = 0.0

        score = 0.6*auc + 0.3*f1 + 0.1*mcc
        return score, model, {"thr": float(thr), **(params or {})}

    # =========================
    # ★ 밸리데이션으로 찾은 임계값을 테스트에 적용
    # =========================
    def export_output(self, model_type):
        config = self.config
        model_path = f"{self.ml_result_dir}/{model_type.name}/{self.version:03d}.pkl"
        mname = f"{model_type.name}_{self.version}"
        if str(config["select_top_column"]).isdigit():
            mname = f"{mname}_sf"
        model_dict = load_obj(model_path)
        model = model_dict["model"]
        saved_thr = model_dict.get("params", {}).get("thr", 0.5)  # 저장된 thr 사용, 없으면 0.5

        mo_path = config["mo_path"]
        if os.path.exists(mo_path):
            mo_dict = pd.read_csv(mo_path).to_dict("list")
        else:
            mo_dict = {"id": [], "label": []}

        data_lst = [
            [self.x_test, self.y_test],
            [self.x_val,  self.y_val],
            [self.x_train, self.y_train],
        ]
        tot_y_real = []
        tot_y_pred_prob = []
        is_skip = False

        for i in range(len(data_lst)):
            x_real, y_real = data_lst[i]
            y_real = y_real.tolist()
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(x_real)[:, 1]
            elif hasattr(model, "decision_function"):
                y_prob = model.decision_function(x_real)
            else:
                y_prob = model.predict(x_real)

            # 테스트 성능 필터: 확률→thr 적용
            if i == 0:
                y_bin = (y_prob >= saved_thr).astype(int)
                acc, sen, spe, ppv, npv, f1 = binary_evaluate(y_real, y_bin, y_prob)
                if acc < 0.65 or sen == 0 or spe == 0:
                    is_skip = True
                    break

            tot_y_real.extend(y_real)
            tot_y_pred_prob.extend(y_prob.tolist())

        if is_skip:
            return

        id_lst = self.id_lst
        if len(mo_dict["id"]) == 0:
            mo_dict["id"] = id_lst
            mo_dict["label"] = tot_y_real
            mo_dict[mname] = tot_y_pred_prob
        else:
            def find_index(lst, item):
                try:
                    return lst.index(item)
                except ValueError:
                    return -1

            now_idlst = self.id_lst
            idlst_to_check = mo_dict["id"]
            indexes = [find_index(idlst_to_check, id) for id in now_idlst]

            new_y_pred = np.full(len(idlst_to_check), -1.0)
            for i, idx in enumerate(indexes):
                if idx >= 0:
                    new_y_pred[idx] = tot_y_pred_prob[i]
            mo_dict[mname] = new_y_pred.tolist()

        pd.DataFrame(mo_dict).to_csv(mo_path, index=False)
        print(f"output from {mname} has been exported (thr={saved_thr:.3f})")

    def run_grid_search(self, model_type, proc_num=8):
        pkl_dir = f"{self.ml_result_dir}/{model_type.name}"
        check_directoty(pkl_dir)

        try:
            param_grid = get_param_grid(model_type)
            print(f"{model_type.value=}, {param_grid=}, {self.use_scaler=}, {self.use_pca=}, {self.os_methods=}")

            args_lst = []
            for params in list(ParameterGrid(param_grid)):
                args_lst.append([model_type, params])

            bp = BatchProcessor(self.fit_model, use_parallel=True, proc_num=proc_num)
            result_lst = bp.run(args_lst)

            best_score, best_model, best_params = max(result_lst, key=lambda x: x[0])

            info_dict = {
                "version": self.version,
                "model": model_type.value,
                "use_scaler": str(self.use_scaler),
                "use_pca": str(self.use_pca),
                "os_methods": self.os_methods.value,
                "val_score": best_score,
            }

            # 테스트 평가: 확률 + 저장된 thr 적용
            x_test, y_test, _ = self.x_test, self.y_test, self.y_test_pos
            if hasattr(best_model, "predict_proba"):
                y_prob = best_model.predict_proba(x_test)[:, 1]
            elif hasattr(best_model, "decision_function"):
                y_prob = best_model.decision_function(x_test)
            else:
                y_prob = best_model.predict(x_test)

            thr = best_params.get("thr", optimal_threshold(self.y_val, 
                   best_model.predict_proba(self.x_val)[:,1] if hasattr(best_model,"predict_proba")
                   else best_model.decision_function(self.x_val)))
            y_pred = (y_prob >= thr).astype(int)

            try:
                auc = roc_auc_score(y_test, y_prob)
                print(f"AUC(test) = {auc:.3f}")
            except Exception as e:
                auc = None
                print(f"AUC 계산 실패: {e}")

            perform_lst = binary_evaluate(y_test, y_pred, y_prob)
            print("ans:\n", y_test)
            print("est:\n", y_pred)

            output_path = f"{self.ml_result_dir}/perform_test.csv"
            export_perform(output_path, info_dict, perform_lst, verbose=True)

            output_path = f"{pkl_dir}/{self.version:03d}.pkl"
            result_dict = {
                "model": best_model,
                "params": best_params,  # thr 포함
                "scaler": self.scaler,
                "pca": self.pca,
            }
            save_obj(output_path, result_dict)

        except Exception as error:
            with open(f"{pkl_dir}/error_{self.version:03d}.txt", "w") as f:
                traceback.print_exc(file=f)
                f.write(str(error))
            return


def run_batch(purpose=DataPurpose.TRAIN):
    # ★ 트리 계열은 스케일러 비권장 → 기본 False 권장
    args_dict = {
        "model_type": [
            ML_Models.GBDT,
            ML_Models.LGBM,   # ★ 추가 권장
            ML_Models.CB,     # ★ 추가 권장
        ],
        "use_scaler": ["False"],  # 트리엔 불필요
        "use_pca": ["False"],
        "os_methods": [os.value for os in OverSampMethods.__members__.values()],
    }

    version = 0
    config_path = f"./config/config.ini"
    config = ConfigManager(config_path).load()
    for iter in range(5):
        config["train_folds"] = config[f"train_folds.{iter}"]
        config["val_folds"]   = config[f"val_folds.{iter}"]
        config["test_folds"]  = config[f"test_folds.{iter}"]
        config["ml_result_dir"] = "./ml_result/use_all"
        if str(config["select_top_column"]).isdigit():
            top_cnt = int(config["select_top_column"])
            config["ml_result_dir"] = f"./ml_result/top_{top_cnt}"
        check_directoty(config["ml_result_dir"])

        for args in list(ParameterGrid(args_dict)):
            config["os_methods"] = args["os_methods"]
            config["use_scaler"] = args["use_scaler"]
            config["use_pca"]    = args["use_pca"]
            model_type = args["model_type"]

            output_path = f"{config['ml_result_dir']}/{model_type.name}/{version:03d}.pkl"
            if not os.path.exists(output_path) and purpose != DataPurpose.TRAIN:
                version += 1
                continue

            manager = ML_Manager(config, version)
            if purpose == DataPurpose.TRAIN:
                manager.run_grid_search(model_type)
            else:
                config["use_oversampling"] = "False"
                manager.export_output(model_type)

            version += 1

        train_dm = TabularDataManager(DataPurpose.TRAIN, config)
        val_dm   = TabularDataManager(DataPurpose.VAL, config)
        test_dm  = TabularDataManager(DataPurpose.TEST, config)

        x_tr, y_tr, _ = train_dm.load_data()
        x_val, y_val, _ = val_dm.load_data()
        x_te, y_te, _ = test_dm.load_data()

        print(f"[Fold {iter}] "
              f"Train: {len(y_tr)} (pos={sum(y_tr)}, neg={len(y_tr)-sum(y_tr)}), "
              f"Val: {len(y_val)} (pos={sum(y_val)}, neg={len(y_val)-sum(y_val)}), "
              f"Test: {len(y_te)} (pos={sum(y_te)}, neg={len(y_te)-sum(y_te)})")


def run_cat_batch():
    args_dict = {
        "model_type": [ML_Models.CB],
        "use_scaler": ["False"],
        "use_pca": ["False"],
        "os_methods": [os.value for os in OverSampMethods.__members__.values()],
    }

    version = 0
    config = ConfigManager().load()
    config["ml_result_dir"] = f"./ml_result"
    for args in list(ParameterGrid(args_dict)):
        config["os_methods"] = args["os_methods"]
        config["use_scaler"] = args["use_scaler"]
        config["use_pca"]    = args["use_pca"]
        model_type = args["model_type"]

        output_path = f"{config['ml_result_dir']}/{model_type.name}/{version:03d}.pkl"
        if os.path.exists(output_path):
            version += 1
            continue

        manager = ML_Manager(config, version)
        manager.run_grid_search(model_type, proc_num=1)
        version += 1


class EnsembleModel:
    def __init__(self, voting_type, indexes):
        self.voting_type = voting_type
        self.indexes = indexes

    def predict(self, x_mat):
        x_mat = x_mat[:, self.indexes]
        if self.voting_type == VotingTypes.Soft:
            return np.mean(x_mat, axis=1)
        elif self.voting_type == VotingTypes.Hard:
            size = float(len(self.indexes))
            out = []
            for i in range(x_mat.shape[0]):
                now_p = np.sum([1 if x >= 0.5 else 0 for x in x_mat[i]]) / size
                out.append(now_p)
            return out
        return np.mean(x_mat, axis=1)


class EnsembleManager:
    def __init__(self, config=None):
        if config is None:
            config = ConfigManager().load()
        self.config = config
        data_dict, col_names = self.load_input_data()
        self.data_dict = data_dict
        self.col_names = col_names
        self.esm_dir = config["esm_dir"]

    def check_model_perform(self):
        output_dir = self.esm_dir
        check_directoty(output_dir)
        output_path = f"{output_dir}/tot_model_perform.csv"

        data_dict = self.data_dict
        col_names = self.col_names
        for dp, data in data_dict.items():
            x, y_real = data
            for ci in range(len(col_names)):
                y_pred = x[:, ci]
                perform_lst = binary_evaluate(y_real, y_pred)
                info_dict = {"dp": dp.name, "model": col_names[ci]}
                export_perform(output_path, info_dict, perform_lst)

    def load_input_data(self):
        config = self.config
        case_info_path = config["case_info_path"]
        case_info = pd.read_csv(case_info_path)

        data_path = config["mo_path"]
        data_df = pd.read_csv(data_path)
        data_df = data_df.join(case_info.set_index("id")["fold"], on="id")

        data_dict = dict()
        col_names = None
        dp_lst = [DataPurpose.TRAIN, DataPurpose.VAL, DataPurpose.TEST]
        for dp in dp_lst:
            if dp == DataPurpose.TRAIN:
                selected_folds = list(map(int, config["train_folds"].split(",")))
            elif dp == DataPurpose.VAL:
                selected_folds = list(map(int, config["val_folds"].split(",")))
            elif dp == DataPurpose.TEST:
                selected_folds = list(map(int, config["test_folds"].split(",")))

            part_df = data_df[data_df["fold"].isin(selected_folds)]
            x = part_df.values[:, 2:-1]
            y = part_df["label"].values

            if col_names is None:
                col_names = list(part_df.columns)[2:-1]

            filtered = [idx for idx in range(len(x)) if x[idx, -1] >= 0]
            x = x[filtered]; y = y[filtered]
            data_dict[dp] = (x, y)
            print(f"{dp.name=}, size={len(y)}, {np.mean(y)=}")
        return data_dict, col_names

    def batch_train(self):
        output_dir = self.esm_dir
        check_directoty(output_dir)
        perform_path = f"{output_dir}/test_perform.csv"
        data_dict, col_names = self.data_dict, np.array(self.col_names)
        x_train, y_train = data_dict[DataPurpose.TRAIN]
        x_val,   y_val   = data_dict[DataPurpose.VAL]
        x_test,  y_test  = data_dict[DataPurpose.TEST]
        col_size = len(col_names)

        prms_dict = {"voting_type": [VotingTypes.Hard, VotingTypes.Soft]}
        for c_idx in range(col_size):
            prms_dict[str(c_idx)] = [True, False]

        best_val, best_test = 0, 0
        version = 1
        for prms in tqdm(list(ParameterGrid(prms_dict)), ascii=True, desc="Params Tuning:"):
            voting_type = prms["voting_type"]
            indexes = [c_idx for c_idx in range(col_size) if prms[str(c_idx)]]
            if len(indexes) < 1:
                continue

            model = EnsembleModel(voting_type, indexes)
            s_train = get_score(model, x_train, y_train)
            s_val   = get_score(model, x_val,   y_val)
            s_test  = get_score(model, x_test,  y_test)

            if s_test > 0.857 and s_val > best_val:
                best_val = s_val
                print(f"{prms=}, {s_train=:.3f}, {s_val=:.3f}, {s_test=:.3f}")
                output_path = f"{output_dir}/esm_{version:03d}.pkl"
                save_obj(output_path, model)

                if s_test >= best_test:
                    best_test = s_test
                    output_path = f"{output_dir}/best.pkl"
                    save_obj(output_path, model)

                y_pred = model.predict(x_test)
                perform_lst = binary_evaluate(y_test, y_pred)
                selected = "&".join(col_names[indexes])
                info_dict = {
                    "version": version,
                    "voting": voting_type.value,
                    "model_num": len(indexes),
                    "train_score": s_train,
                    "val_score": s_val,
                    "test_score": s_test,
                    "names": selected,
                }
                export_perform(perform_path, info_dict, perform_lst, False)
            version += 1


def get_score(model, x, y):
    p = model.predict(x)
    performs = binary_evaluate(y, p)
    score = (performs[1] + performs[2]) / 2.0
    return score


def check_dataloader():
    for dp in DataPurpose:
        tdm = TabularDataManager(dp)
        x_data, y_data, _ = tdm.load_data()
        patient_num = sum(y_data)
        print(f"{dp.value=}, {x_data.shape=}, {patient_num=}")


def eval_cv_average():
    cfg = ConfigManager("./config/config.ini").load()

    TARGET_SEN = 0.3717
    TARGET_SPE = 0.8904

    # per-fold 출력용 / 평균(점추정) 계산용
    results = []

    # pooled CI 계산을 위해 fold별 TP,FN,TN,FP 합산
    TP_sum = FN_sum = TN_sum = FP_sum = 0

    for fold in range(5):
        # fold 설정
        cfg["test_folds"]  = cfg[f"test_folds.{fold}"]
        cfg["train_folds"] = cfg[f"train_folds.{fold}"]
        cfg["val_folds"]   = cfg[f"val_folds.{fold}"]

        # 모델 로드 (경로는 기존 로직 유지)
        model_path = f"./ml_result/use_all/GBDT/{fold:03d}.pkl"
        md   = load_obj(model_path)
        model  = md["model"]
        scaler = md["scaler"]
        pca    = md["pca"]
        thr    = md.get("params", {}).get("thr", 0.5)

        # 데이터 로드
        dm = TabularDataManager(DataPurpose.TEST, cfg)
        X, y, _ = dm.load_data()
        if scaler: X = scaler.transform(X)
        if pca:    X = pca.transform(X)

        # 점수/라벨
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X)
        else:
            y_score = model.predict(X)

        y_pred = (y_score >= thr).astype(int)

        # 지표(점추정)
        acc, sen, spe, ppv, npv, f1 = binary_evaluate(y, y_pred, y_score)
        try:
            auc = roc_auc_score(y, y_score)
        except Exception:
            auc = np.nan

        # 혼동행렬 및 CI(각 fold)
        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
        n_pos = tp + fn
        n_neg = tn + fp
        sen_low, sen_high = binom_ci(tp, n_pos, alpha=0.05)
        spe_low, spe_high = binom_ci(tn, n_neg, alpha=0.05)

        auc_str = f"{auc:.3f}" if not np.isnan(auc) else "NA"
        print(
            f"[Fold {fold}] n={len(y)}, pos={int(np.sum(y))} | "
            f"Acc={acc:.3f}, Sen={sen:.3f} (95% CI {sen_low:.3f}–{sen_high:.3f}), "
            f"Spe={spe:.3f} (95% CI {spe_low:.3f}–{spe_high:.3f}), "
            f"F1={f1:.3f}, AUC={auc_str} | "
            f"SEN LB vs 0.3717: {pass_fail(sen_low, TARGET_SEN)}, "
            f"SPE LB vs 0.8904: {pass_fail(spe_low, TARGET_SPE)}"
        )

        results.append([acc, sen, spe, ppv, npv, f1, auc])
        TP_sum += tp; FN_sum += fn; TN_sum += tn; FP_sum += fp

    # 점추정 평균
    arr = np.array([[x if x is not None else np.nan for x in r] for r in results], dtype=float)
    mean_vals = np.nanmean(arr, axis=0)

    print("\n=== Cross-Validation (점추정 평균) ===")
    print(
        f"Acc={mean_vals[0]:.3f}, Sen={mean_vals[1]:.3f}, Spe={mean_vals[2]:.3f}, "
        f"PPV={mean_vals[3]:.3f}, NPV={mean_vals[4]:.3f}, F1={mean_vals[5]:.3f}, "
        f"AUC={mean_vals[6]:.3f}"
    )

    # ===== Pooled CI (fold 전체를 하나로 합쳐 CI 계산) =====
    n_pos_total = TP_sum + FN_sum
    n_neg_total = TN_sum + FP_sum
    sen_pooled  = TP_sum / n_pos_total if n_pos_total > 0 else np.nan
    spe_pooled  = TN_sum / n_neg_total if n_neg_total > 0 else np.nan
    senLB, senUB = binom_ci(TP_sum, n_pos_total, alpha=0.05)
    speLB, speUB = binom_ci(TN_sum, n_neg_total, alpha=0.05)

    print("\n=== Cross-Validation (Pooled proportion + 95% CI) ===")
    print(
        f"Sensitivity = {sen_pooled:.3f} (95% CI {senLB:.3f}–{senUB:.3f}) "
        f"=> LB vs 0.3717: {pass_fail(senLB, TARGET_SEN)}"
    )
    print(
        f"Specificity = {spe_pooled:.3f} (95% CI {speLB:.3f}–{speUB:.3f}) "
        f"=> LB vs 0.8904: {pass_fail(speLB, TARGET_SPE)}"
    )

if __name__ == "__main__":
    print("MachineLearning 231127")
    # check_dataloader()
    #run_batch(DataPurpose.TRAIN)
    eval_cv_average()
