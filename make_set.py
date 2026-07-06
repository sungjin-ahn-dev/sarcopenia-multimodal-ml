# build_datasets.py
import os
import re
import math
import json
import random
import numpy as np
import pandas as pd
from collections import defaultdict

# -----------------------------
# 0) 경로/출력 이름
# -----------------------------
SITE_A_SITE_B_CSV = "./data/csv/tabular_data_site_ab.csv"
SITE_C_CSV        = "./data/csv/site_c.csv"
OUT_DIR         = "./_new_splits"   # 출력 폴더

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# 1) 유틸: 컬럼 표준화 & 병원 태깅
# -----------------------------
NORM_MAP = {
    "sub no.": "PDT",
    "sub_no": "PDT",
    "sub_no.": "PDT",
    "id": "PDT",
    "sex": "SEX", "gender": "SEX",
    "sarcopenia": "Sarcopenia_label",
    "sarcopenia_label": "Sarcopenia_label",
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    def norm_one(c):
        k = c.strip().lower()
        return NORM_MAP.get(k, c.strip())
    df.columns = [norm_one(c) for c in df.columns]
    return df

def infer_hospital_from_pdt(pdt: str) -> str:
    if not isinstance(pdt, str):
        return "UNKNOWN"
    u = pdt.strip().upper()
    if "SITE_A" in u:   return "SITE_A"
    if "SITE_B" in u: return "SITE_B"
    if "SITE_C" in u:   return "SITE_C"
    # 자주 쓰는 패턴 보정
    if u.startswith("SITE_A"):  return "SITE_A"
    if u.startswith("SITE_B"): return "SITE_B"
    if u.startswith("SITE_C"):  return "SITE_C"
    return "UNKNOWN"


PDT_ALIASES = {"pdt", "id", "sub no.", "sub_no", "sub_no."}

def _first_non_empty(row):
    for v in row:
        s = str(v).strip()
        if s and s.lower() != "nan" and s.lower() != "none":
            return s
    return np.nan

def add_hospital_tag(df: pd.DataFrame, fallback_hospital: str = None) -> pd.DataFrame:
    df = df.copy()

    # 1) 컬럼명 정규화(공백/소문자 기준으로 비교)
    norm_map = {c: c.strip() for c in df.columns}
    df.rename(columns=norm_map, inplace=True)

    # 2) PDT 후보 컬럼 모으기 (동일 이름 중복 포함, alias 포함)
    pdt_like_cols = [c for c in df.columns if c.strip().lower() in PDT_ALIASES or c.strip().upper() == "PDT"]

    if not pdt_like_cols:
        raise ValueError("PDT를 나타내는 컬럼을 찾지 못했습니다. (가능 후보: PDT, ID, sub no., sub_no, sub_no.)")

    # 3) 중복 PDT 처리: 행 단위로 가장 먼저 나온 비어있지 않은 값 선택하여 단일 'PDT' 생성
    #    (여러 PDT 유사 컬럼이 있을 때, 비어있지 않은 첫 값을 선택)
    pdt_block = df[pdt_like_cols].astype(str)
    df["PDT"] = pdt_block.apply(_first_non_empty, axis=1)

    # 혹시 또 DataFrame으로 남아있을 가능성 방지
    if isinstance(df["PDT"], pd.DataFrame):
        df["PDT"] = df["PDT"].apply(_first_non_empty)

    # 4) 병원 추정 함수(문자열 하나만 반환하도록!)
    def infer_hospital_from_pdt(pdt_val: str) -> str:
        s = str(pdt_val).lower().strip()
        if "site_b" in s:
            return "SITE_B"
        if "site_a" in s:
            return "SITE_A"
        if "site_c" in s:
            return "SITE_C"
        return fallback_hospital  # 못찾으면 지정한 기본 병원명(또는 None)

    # 5) hospital 컬럼 생성 (반드시 Series를 RHS로 보장)
    df["hospital"] = df["PDT"].astype(str).map(infer_hospital_from_pdt)

    return df
def load_and_prepare(path: str, fallback_hospital: str = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_columns(df)
    if "PDT" not in df.columns:
        # 혹시 sub no.만 있을 때를 대비
        for cand in ["sub no.", "sub_no", "ID", "id"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "PDT"})
                break
    if "Sarcopenia_label" not in df.columns:
        raise ValueError(f"{path} 에 'Sarcopenia_label' 라벨 컬럼이 없습니다. 컬럼을 확인해 주세요.")
    df = add_hospital_tag(df, fallback_hospital)
    return df

# -----------------------------
# 2) 데이터 로드
#    - site_a+site_b 합본
#    - site_c 별도
# -----------------------------
df_bt = load_and_prepare(SITE_A_SITE_B_CSV)     # SITE_A + SITE_B 혼합
df_d  = load_and_prepare(SITE_C_CSV, "SITE_C")    # SITE_C

# 병원 표기 보정(혹시 모르니)
df_bt.loc[df_bt["hospital"].eq("UNKNOWN"), "hospital"] = df_bt["PDT"].astype(str).map(infer_hospital_from_pdt)
df_d.loc[df_d["hospital"].eq("UNKNOWN"),  "hospital"]  = "SITE_C"

# -----------------------------
# 3) 기본 통계 출력(참고)
# -----------------------------
def quick_counts(df, name):
    pos = int(df["Sarcopenia_label"].sum())
    n   = len(df)
    print(f"[{name}] n={n}, pos={pos}, neg={n-pos}, by hospital=", df.groupby("hospital")["Sarcopenia_label"].agg(['count','sum']))

quick_counts(df_bt, "SITE_A+SITE_B")
quick_counts(df_d,  "SITE_C")

# -----------------------------
# 4) 샘플링 유틸
# -----------------------------
def stratified_pick(df: pd.DataFrame,
                    total_n: int,
                    pos_n: int,
                    hospitals: list,
                    keep_hospital_mix=True,
                    random_state=42) -> pd.DataFrame:
    """
    - class(양/음)별, 병원별 비율을 가능한 유지하면서
      total_n/pos_n 타깃 수를 맞춰 샘플링
    """
    rng = np.random.default_rng(random_state)

    # 가용 수 확인
    total_avail = len(df)
    pos_avail   = int(df["Sarcopenia_label"].sum())
    neg_avail   = total_avail - pos_avail
    if total_n > total_avail:
        raise ValueError(f"요청 수(total_n={total_n})가 가용 수({total_avail})보다 큽니다.")
    if pos_n > pos_avail:
        raise ValueError(f"요청 양성(pos_n={pos_n})이 가용 양성({pos_avail})보다 큽니다.")

    neg_n = total_n - pos_n
    if neg_n > neg_avail:
        raise ValueError(f"요청 음성(neg_n={neg_n})이 가용 음성({neg_avail})보다 큽니다.")

    # 병원리스트 제한(필요 시)
    if hospitals:
        df = df[df["hospital"].isin(hospitals)].copy()
        if len(df) < total_n:
            raise ValueError(f"지정 병원 {hospitals} 내 가용 수({len(df)})가 total_n={total_n} 보다 적습니다.")

    # 병원별/라벨별 가용 수
    grouped = df.groupby(["hospital", "Sarcopenia_label"])
    avail_map = { (h,c): len(g) for (h,c), g in grouped }

    # 타깃 병원 분배 비율(가능하면 원본 분포 유지)
    if keep_hospital_mix:
        mix = df["hospital"].value_counts(normalize=True).to_dict()
    else:
        mix = {h: 1/len(hospitals) for h in hospitals}

    # 병원별 타깃 총/양성 수 1차 할당(반올림)
    target_total_by_h = {h: int(round(total_n * mix.get(h,0))) for h in mix}
    # 합이 어긋날 수 있어 보정
    while sum(target_total_by_h.values()) != total_n:
        # 가장 큰 병원에 +/- 보정
        diff = total_n - sum(target_total_by_h.values())
        key  = max(target_total_by_h, key=lambda k: mix.get(k,0))
        target_total_by_h[key] += diff

    pos_rate = df["Sarcopenia_label"].mean()  # 병원별이 아니라 전체 비율로 1차 배분
    target_pos_by_h = {h: int(round(target_total_by_h[h]*pos_rate)) for h in target_total_by_h}
    # pos 총합 보정
    while sum(target_pos_by_h.values()) != pos_n:
        diff = pos_n - sum(target_pos_by_h.values())
        key  = max(target_pos_by_h, key=lambda k: target_total_by_h[k])
        target_pos_by_h[key] += diff

    # 병원/라벨 단에서 실제 샘플링
    picks = []
    for h in target_total_by_h:
        need_pos = target_pos_by_h[h]
        need_neg = target_total_by_h[h] - need_pos

        df_h = df[df["hospital"]==h]
        pos_pool = df_h[df_h["Sarcopenia_label"]==1]
        neg_pool = df_h[df_h["Sarcopenia_label"]==0]

        take_pos = min(need_pos, len(pos_pool))
        take_neg = min(need_neg, len(neg_pool))
        pos_sel = pos_pool.sample(n=take_pos, random_state=random_state)
        neg_sel = neg_pool.sample(n=take_neg, random_state=random_state)

        picks.append(pd.concat([pos_sel, neg_sel], axis=0))

    out = pd.concat(picks, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # 혹시 정밀 타깃과 차이가 나면(병원 가용에 막힌 경우) 전체에서 추가 보정
    need_total = total_n - len(out)
    if need_total > 0:
        remained = df.drop(out.index)  # 간단히 빠른 보정
        # 부족한 라벨 계산
        cur_pos = int(out["Sarcopenia_label"].sum())
        need_pos_more = max(0, pos_n - cur_pos)
        need_neg_more = need_total - need_pos_more

        if need_pos_more > 0:
            add_pos = remained[remained["Sarcopenia_label"]==1].sample(n=need_pos_more, random_state=random_state)
        else:
            add_pos = remained.iloc[0:0]
        if need_neg_more > 0:
            add_neg = remained[remained["Sarcopenia_label"]==0].sample(n=need_neg_more, random_state=random_state)
        else:
            add_neg = remained.iloc[0:0]
        out = pd.concat([out, add_pos, add_neg], axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # 최종 검증
    assert len(out)==total_n, f"샘플링 길이 불일치: {len(out)} != {total_n}"
    assert int(out["Sarcopenia_label"].sum())==pos_n, f"양성 수 불일치: {int(out['Sarcopenia_label'].sum())} != {pos_n}"
    return out

def add_stratified_folds(df: pd.DataFrame, n_folds=5, seed=42):
    """
    라벨 + 병원 기준으로 층화하여 fold(0~n_folds-1) 부여
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["fold"] = -1

    for (h, y), g in df.groupby(["hospital","Sarcopenia_label"]):
        idx = g.sample(frac=1.0, random_state=seed).index.tolist()
        for i, ix in enumerate(idx):
            df.loc[ix, "fold"] = i % n_folds
    return df

def save_with_summary(df: pd.DataFrame, path: str, name: str):
    df.to_csv(path, index=False, encoding="utf-8-sig")
    n = len(df); pos = int(df["Sarcopenia_label"].sum()); neg = n - pos
    print(f"[SAVE] {name}: n={n}, pos={pos}, neg={neg}, file={path}")
    print(df.groupby(["hospital","Sarcopenia_label"]).size().unstack(fill_value=0))
    print("-"*60)

# -----------------------------
# 5) 만들 것들
#   A) 모델 학습용(5-fold): 총 283명 (neg 239, pos 44) - 3병원(가능하면 유지)
#   B) 내부 성능평가 16명 (neg 13, pos 3) - 3병원
#   C) 확증임상용 모델 테스트(=datasetZ): 총 279명 (neg 231, pos 48) - SITE_A + SITE_C 만
# -----------------------------

# (A) 5-fold 학습용: 283 (pos 44)
pool_A = pd.concat([df_bt, df_d], axis=0, ignore_index=True)   # 요청: "3개 병원에서 모델과 성능평가 데이터"
ds_A   = stratified_pick(pool_A, total_n=283, pos_n=44, hospitals=["SITE_A","SITE_B","SITE_C"], keep_hospital_mix=True, random_state=7)
ds_A   = add_stratified_folds(ds_A, n_folds=5, seed=7)
save_with_summary(ds_A, os.path.join(OUT_DIR, "datasetA_train_5fold.csv"), "DatasetA(5fold)")

# (B) 내부 성능평가용: 16 (pos 3)
#    주: A와 중복 피하고 싶으면 pool에서 제외 후 뽑기
pool_B = pool_A[~pool_A["PDT"].isin(ds_A["PDT"])].copy()
ds_B   = stratified_pick(pool_B, total_n=16, pos_n=3, hospitals=["SITE_A","SITE_B","SITE_C"], keep_hospital_mix=True, random_state=11)
save_with_summary(ds_B, os.path.join(OUT_DIR, "datasetB_internal_eval_16.csv"), "DatasetB(Internal-16)")

# (C) 확증임상 모델 테스트: 279 (pos 48), SITE_A + SITE_C만
pool_C = pd.concat([df_bt[df_bt["hospital"]=="SITE_A"], df_d], axis=0, ignore_index=True)
# 중복 방지: A, B에 쓴 대상 제거
used_ids = set(pd.concat([ds_A["PDT"], ds_B["PDT"]], axis=0).tolist())
pool_C   = pool_C[~pool_C["PDT"].isin(used_ids)].copy()
ds_C     = stratified_pick(pool_C, total_n=279, pos_n=48, hospitals=["SITE_A","SITE_C"], keep_hospital_mix=True, random_state=19)
save_with_summary(ds_C, os.path.join(OUT_DIR, "datasetZ_confirmatory_test_279.csv"), "DatasetZ(Confirmatory)")

# 보조: 요약표 CSV도 함께 저장
summary_rows = []
for nm, d in [("DatasetA(5fold)", ds_A), ("DatasetB(Internal-16)", ds_B), ("DatasetZ(Confirmatory)", ds_C)]:
    row = {"name": nm, "n": len(d), "pos": int(d["Sarcopenia_label"].sum()), "neg": len(d) - int(d["Sarcopenia_label"].sum())}
    # 병원별/양성 분할
    ctab = d.pivot_table(index="hospital", columns="Sarcopenia_label", values="PDT", aggfunc="count", fill_value=0)
    for h in ["SITE_A","SITE_B","SITE_C","UNKNOWN"]:
        if h in ctab.index:
            row[f"{h}_neg"] = int(ctab.loc[h, 0]) if 0 in ctab.columns else 0
            row[f"{h}_pos"] = int(ctab.loc[h, 1]) if 1 in ctab.columns else 0
        else:
            row[f"{h}_neg"] = 0; row[f"{h}_pos"] = 0
    summary_rows.append(row)

pd.DataFrame(summary_rows).to_csv(os.path.join(OUT_DIR, "split_summary.csv"), index=False, encoding="utf-8-sig")
print("[DONE] 모든 산출물이 저장되었습니다.")
