import pandas as pd
import numpy as np
import os

# ── 경로 설정 ─────────────────────────────────────────
base_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(base_dir, "wafer_train_data.pkl")
output_path = os.path.join(base_dir, "wafer_train_data_B.pkl")

DEFECT_THRESHOLD = 0.15  # 15% 기준


# ── 불량 비율 계산 함수 ───────────────────────────────
def get_defect_ratio(wm):
    arr = np.array(wm)
    total = int(np.sum(arr > 0))
    if total == 0:
        return 0.0
    return int(np.sum(arr == 2)) / total


# ── 라벨 추출 함수 ────────────────────────────────────
def extract_label(x):
    if isinstance(x, (list, np.ndarray)) and len(x) > 0:
        inner = x[0]
        if isinstance(inner, (list, np.ndarray)) and len(inner) > 0:
            return str(inner[0])
        return str(inner)
    return "None"


# ── 메인 ─────────────────────────────────────────────
if not os.path.exists(input_path):
    print(f"❌ 에러: {input_path} 파일이 없습니다.")
else:
    print(f"▶ [{input_path}] 로딩 중...")
    df = pd.read_pickle(input_path)
    print(f"▶ 로딩 완료 — 전체: {len(df):,}개")

    # 라벨 추출
    df["label"] = df["failureType"].apply(extract_label)

    # 불량 비율 계산
    df["defect_ratio"] = df["waferMap"].apply(get_defect_ratio)

    # ── 제거 조건 ─────────────────────────────────────
    # 라벨이 None/none/빈값 이고 불량 비율이 15% 이상인 행
    is_none_label = df["label"].str.strip().str.lower().isin(["none", ""])
    is_over_threshold = df["defect_ratio"] >= DEFECT_THRESHOLD
    mask_remove = is_none_label & is_over_threshold

    removed = int(mask_remove.sum())
    df_cleaned = df[~mask_remove].copy()

    # 임시 컬럼 제거 (원본 구조 유지)
    df_cleaned = df_cleaned.drop(columns=["label", "defect_ratio"])

    # ── 저장 ─────────────────────────────────────────
    df_cleaned.to_pickle(output_path)

    print("-" * 40)
    print(f"✅ 완료!")
    print(f"   원본 샘플 수  : {len(df):,}개")
    print(f"   제거된 샘플 수: {removed:,}개  (None/빈값 & 불량 15% 이상)")
    print(f"   정제 후 샘플 수: {len(df_cleaned):,}개")
    print(f"   저장 경로     : {output_path}")
    print("-" * 40)
