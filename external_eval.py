import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score, roc_curve, auc
from threshold_utils import threshold_table, choose_threshold_by_youden, choose_threshold_for_target_sen, plot_roc
from sklearn.metrics import confusion_matrix
from scipy.stats import beta
import matplotlib.pyplot as plt
# ===== 평가 지표 유틸 =====

def binom_ci(k, n, alpha=0.05):
    if n == 0:
        return (np.nan, np.nan)
    try:
        low  = beta.ppf(alpha / 2, k, n - k + 1)
        high = beta.ppf(1 - alpha / 2, k + 1, n - k)
        return (low, high)
    except Exception:
        p = k / n
        z = 1.96
        denom = 1 + z**2 / n
        center = p + z**2 / (2 * n)
        half = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
        low = (center - half) / denom
        high = (center + half) / denom
        return (max(0.0, low), min(1.0, high))

def auc_ci_bootstrap(y_true, y_score, n_bootstraps=2000, alpha=0.05):
    """
    Bootstrap을 사용하여 AUC의 95% 신뢰구간 계산
    """
    rng = np.random.RandomState(42)
    auc_scores = []
    
    for _ in range(n_bootstraps):
        # 부트스트랩 샘플 생성
        indices = rng.choice(len(y_true), len(y_true), replace=True)
        y_true_boot = y_true[indices]
        y_score_boot = y_score[indices]
        
        # AUC 계산 (클래스가 하나만 있으면 skip)
        if len(np.unique(y_true_boot)) == 2:
            auc_boot = roc_auc_score(y_true_boot, y_score_boot)
            auc_scores.append(auc_boot)
    
    auc_scores = np.array(auc_scores)
    ci_lower = np.percentile(auc_scores, (alpha/2)*100)
    ci_upper = np.percentile(auc_scores, (1-alpha/2)*100)
    
    return ci_lower, ci_upper

def pass_fail(lb, target):
    return "PASS" if (not np.isnan(lb) and lb >= target) else "FAIL"

def binary_evaluate(y_true, y_pred_label, y_score=None):
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
    acc = accuracy_score(y_true, y_pred_label)
    sen = recall_score(y_true, y_pred_label, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_label, labels=[0,1]).ravel()
    spe = tn / (tn + fp) if (tn+fp) > 0 else 0.0
    ppv = precision_score(y_true, y_pred_label, zero_division=0)
    npv = tn / (tn + fn) if (tn+fn) > 0 else 0.0
    f1  = f1_score(y_true, y_pred_label, zero_division=0)
    return acc, sen, spe, ppv, npv, f1

def fill_data(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
    return df

def load_obj(path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)

# ===== 경로/설정 =====
#EXTERNAL_PATH = "./data/csv/datasetZ.csv"
#EXTERNAL_PATH = "./data/csv/site_c.csv"
EXTERNAL_PATH = "./data/csv/external.csv"

MODEL_DIR = "./ml_result/use_all/GBDT"
MODEL_TEMPLATE = "./ml_result/use_all/GBDT/{:03d}.pkl"

# site_b.csv 에서 사용할 정확한 피처 목록 (요청하신 6개만 사용)
EXTERNAL_FEATURES = ["SEX", "Age", "Weight", "BMI", "SMI", "IBgrip_MAX",'swipe_horizontal_rms_distance','touch_horizontal_one_press_consistency','touch_horizontal_one_left_mean','touch_horizontal_two_balance_ratio']
LABEL_COL = "Sarcopenia_label"

# 대소문자/스페이스 대응(선택)
NORM_MAP = {
    "sex": "SEX",
    "age": "Age",
    "weight": "Weight",
    "bmi": "BMI",
    "smi": "SMI",
    'swipe_horizontal_rms_distance':'swipe_horizontal_rms_distance',
    'touch_horizontal_one_press_consistency':'touch_horizontal_one_press_consistency',
    'touch_horizontal_one_left_mean':'touch_horizontal_one_left_mean',
    'touch_horizontal_two_balance_ratio':'touch_horizontal_two_balance_ratio',
    "ibgrip_max": "IBgrip_MAX",
    "sarcopenia": LABEL_COL,
    "sarcopenia_label": LABEL_COL,
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 컬럼을 정규화 (소문자 -> 타깃 이름으로 매핑)
    def norm_one(c):
        k = c.strip().lower()   
        return NORM_MAP.get(k, c.strip())
    df.columns = [norm_one(c) for c in df.columns]
    return df

def load_external_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_columns(df)
    # 라벨 존재 확인
    if LABEL_COL not in df.columns:
        raise ValueError(f"External file must contain label column '{LABEL_COL}'. Found: {list(df.columns)}")
    # 피처 존재 확인
    missing = [c for c in EXTERNAL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"External file missing required features {missing}. Available: {list(df.columns)}")
    return df


def evaluate_external(EXTERNAL_PATH: str):
    TARGET_SEN = 0.3717
    TARGET_SPE = 0.8904

    ext = load_external_table(EXTERNAL_PATH)
    X_ext = ext[EXTERNAL_FEATURES].copy()
    y_ext = pd.to_numeric(ext[LABEL_COL], errors="coerce").fillna(0).astype(int).values
    X_ext = fill_data(X_ext)
    X_eval_raw = X_ext.values

    print(f"[Eval] External shape = {X_eval_raw.shape}, pos={int(np.sum(y_ext))}, neg={len(y_ext)-int(np.sum(y_ext))}")

    results = []
    fold_auc_results = []  # fold별 AUC 결과 저장
    roc_data_list = []  # ROC curve 데이터 저장
    TP_sum = FN_sum = TN_sum = FP_sum = 0

    for fold in range(5):
        model_path = MODEL_TEMPLATE.format(fold)
        if not os.path.exists(model_path):
            print(f"[Skip] Model not found: {model_path}")
            continue

        md = load_obj(model_path)
        model = md.get("model", None)
        scaler = md.get("scaler", None)
        pca = md.get("pca", None)

        # ✅ threshold를 항상 0.5로 고정
        thr = 0.5  

        X_eval = X_eval_raw.copy()
        if scaler is not None:
            X_eval = scaler.transform(X_eval)
        if pca is not None:
            X_eval = pca.transform(X_eval)

        # 예측
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_eval)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_eval)
        else:
            y_score = model.predict(X_eval)
        y_pred_label = (y_score >= thr).astype(int)
        
        # ===== 디버깅: y_score 분포 확인 =====
        unique_scores = np.unique(y_score)
        print(f"[Fold {fold}] y_score range: {y_score.min():.6f} ~ {y_score.max():.6f}, unique values: {len(unique_scores)}")

        acc, sen, spe, ppv, npv, f1 = binary_evaluate(y_ext, y_pred_label, y_score)
        auc_score = roc_auc_score(y_ext, y_score) if y_score is not None else np.nan
        
        # ===== AUC의 95% 신뢰구간 계산 =====
        auc_ci_low, auc_ci_high = auc_ci_bootstrap(y_ext, y_score, n_bootstraps=2000)

        # CI 계산
        tn, fp, fn, tp = confusion_matrix(y_ext, y_pred_label, labels=[0, 1]).ravel()
        n_pos, n_neg = tp + fn, tn + fp
        sen_low, sen_high = binom_ci(tp, n_pos)
        spe_low, spe_high = binom_ci(tn, n_neg)

        auc_str = f"{auc_score:.3f} (95% CI {auc_ci_low:.3f}–{auc_ci_high:.3f})" if not np.isnan(auc_score) else "NA"
        print(
            f"[External Fold {fold}] n={len(y_ext)}, pos={int(np.sum(y_ext))} | "
            f"Acc={acc:.3f}, Sen={sen:.3f} (95% CI {sen_low:.3f}–{sen_high:.3f}), "
            f"Spe={spe:.3f} (95% CI {spe_low:.3f}–{spe_high:.3f}), "
            f"F1={f1:.3f}, AUC={auc_str}, thr={thr:.3f} | "
            f"SEN LB vs 0.3717: {pass_fail(sen_low, TARGET_SEN)}, "
            f"SPE LB vs 0.8904: {pass_fail(spe_low, TARGET_SPE)}"
        )

        results.append([acc, sen, spe, ppv, npv, f1, auc_score])
        
        # ===== Fold별 AUC 저장 =====
        fold_auc_results.append({
            'fold': fold,
            'auc': auc_score,            'auc_ci_low': auc_ci_low,
            'auc_ci_high': auc_ci_high,            'accuracy': acc,
            'sensitivity': sen,
            'specificity': spe,
            'ppv': ppv,
            'npv': npv,
            'f1': f1
        })
        
        # ===== ROC curve 데이터 저장 =====
        fpr, tpr, _ = roc_curve(y_ext, y_score)
        roc_data_list.append({
            'fold': fold,
            'fpr': fpr,
            'tpr': tpr,
            'auc': auc_score
        })
        
        TP_sum += tp; FN_sum += fn; TN_sum += tn; FP_sum += fp

    if not results:
        print("No fold results produced.")
        return

    arr = np.array(results, dtype=float)
    mean_vals = np.nanmean(arr, axis=0)

    print("\n=== External Validation (mean across folds) ===")
    print(
        f"Acc={mean_vals[0]:.3f}, Sen={mean_vals[1]:.3f}, Spe={mean_vals[2]:.3f}, "
        f"PPV={mean_vals[3]:.3f}, NPV={mean_vals[4]:.3f}, F1={mean_vals[5]:.3f}, "
        f"AUC={mean_vals[6]:.3f}"
    )

    # ===== Pooled CI =====
    n_pos_total = TP_sum + FN_sum
    n_neg_total = TN_sum + FP_sum
    sen_pooled = TP_sum / n_pos_total if n_pos_total > 0 else np.nan
    spe_pooled = TN_sum / n_neg_total if n_neg_total > 0 else np.nan
    senLB, senUB = binom_ci(TP_sum, n_pos_total)
    speLB, speUB = binom_ci(TN_sum, n_neg_total)

    print("\n=== External Validation (Pooled + 95% CI) ===")
    print(
        f"Sensitivity = {sen_pooled:.3f} (95% CI {senLB:.3f}–{senUB:.3f}) "
        f"=> LB vs 0.3717: {pass_fail(senLB, TARGET_SEN)}"
    )
    print(
        f"Specificity = {spe_pooled:.3f} (95% CI {speLB:.3f}–{speUB:.3f}) "
        f"=> LB vs 0.8904: {pass_fail(speLB, TARGET_SPE)}"
    )
    
    # ===== Fold별 AUC 결과를 CSV로 저장 =====
    _save_fold_auc_results(fold_auc_results)
    
    # ===== Fold별 ROC curve 그리기 및 저장 =====
    _plot_fold_roc_curves(roc_data_list, mean_vals[6])


def _save_fold_auc_results(fold_auc_results):
    """Fold별 AUC 및 평가 지표를 CSV로 저장"""
    output_path = "./external_fold_auc_results.csv"
    
    df = pd.DataFrame(fold_auc_results)
    df.to_csv(output_path, index=False)
    
    print(f"\n[Saved] Fold별 AUC 결과: {output_path}")
    print(df.to_string(index=False))
    
    # 평균값 추가 출력
    mean_auc = df['auc'].mean()
    std_auc = df['auc'].std()
    print(f"\nMean AUC across folds: {mean_auc:.3f} ± {std_auc:.3f}")


def _plot_fold_roc_curves(roc_data_list, mean_auc):
    """Fold별 ROC curve를 그려서 저장 (개별 파일)"""
    if not roc_data_list:
        print("[Warning] No ROC data to plot.")
        return
    
    # 폴더 생성 (ROC curve 이미지 저장용)
    output_dir = "./external_roc_curves"
    os.makedirs(output_dir, exist_ok=True)
    
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    # ===== 각 fold별로 개별 ROC curve 저장 =====
    for i, roc_data in enumerate(roc_data_list):
        fold = roc_data['fold']
        fpr = roc_data['fpr']
        tpr = roc_data['tpr']
        auc_score = roc_data['auc']
        
        # FPR로 정렬 (부드러운 곡선을 위해)
        sort_idx = np.argsort(fpr)
        fpr_sorted = fpr[sort_idx]
        tpr_sorted = tpr[sort_idx]
        
        # ===== 중복된 FPR 값 제거 =====
        unique_idx = np.concatenate(([True], np.diff(fpr_sorted) > 1e-10))
        fpr_unique = fpr_sorted[unique_idx]
        tpr_unique = tpr_sorted[unique_idx]
        
        # 개별 figure 생성
        plt.figure(figsize=(10, 8))
        
        color = colors[i % len(colors)]
        plt.plot(fpr_unique, tpr_unique, color=color, lw=2.5, 
                 label=f'AUC = {auc_score:.3f}')
        
        # 대각선 그리기
        plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance (AUC = 0.500)')
        
        # 레이블 및 타이틀 설정
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve', fontsize=14)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # 개별 파일로 저장
        fold_path = os.path.join(output_dir, f"roc_fold_{fold:03d}.png")
        plt.savefig(fold_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[Saved] {fold_path}")
    
    # ===== 모든 fold를 하나의 그래프에 그리기 =====
    plt.figure(figsize=(10, 8))
    
    for i, roc_data in enumerate(roc_data_list):
        fold = roc_data['fold']
        fpr = roc_data['fpr']
        tpr = roc_data['tpr']
        auc_score = roc_data['auc']
        
        # FPR로 정렬
        sort_idx = np.argsort(fpr)
        fpr_sorted = fpr[sort_idx]
        tpr_sorted = tpr[sort_idx]
        
        # ===== 중복된 FPR 값 제거 =====
        unique_idx = np.concatenate(([True], np.diff(fpr_sorted) > 1e-10))
        fpr_unique = fpr_sorted[unique_idx]
        tpr_unique = tpr_sorted[unique_idx]
        
        color = colors[i % len(colors)]
        plt.plot(fpr_unique, tpr_unique, color=color, lw=2.5, 
                 label=f'AUC = {auc_score:.3f}')
    
    # 대각선 그리기
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance (AUC = 0.500)')
    
    # 레이블 및 타이틀 설정
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve by Fold (Mean AUC = {mean_auc:.3f})', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # 통합 파일 저장
    combined_path = os.path.join(output_dir, "roc_all_folds_combined.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[Saved] {combined_path}")




if __name__ == '__main__':
    evaluate_external(EXTERNAL_PATH)  