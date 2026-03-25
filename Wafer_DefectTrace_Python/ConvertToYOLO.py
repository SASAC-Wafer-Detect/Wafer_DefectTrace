"""
ConvertToYOLO.py
────────────────
wafer_train_data_A_sampled.pkl / wafer_train_data_B_sampled.pkl 을
YOLO11 cls 모드 폴더 구조로 변환

출력 구조:
dataset_A/
├── train/
│   ├── Center/
│   ├── Donut/
│   └── ...
└── val/
    ├── Center/
    ├── Donut/
    └── ...
"""

import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

# ── 설정 ─────────────────────────────────────────────
RANDOM_SEED = 42
VAL_RATIO = 0.2  # train 80% / val 20%
IMG_SIZE = 128  # 128×128 리사이즈
IMG_EXT = ".png"

base_dir = os.path.dirname(os.path.abspath(__file__))
sample_dir = os.path.join(base_dir, "SampleDataSet")

FILES = {
    "A": os.path.join(sample_dir, "wafer_train_data_A_sampled.pkl"),
    "B": os.path.join(sample_dir, "wafer_train_data_B_sampled.pkl"),
}


# ── 라벨 추출 함수 ────────────────────────────────────
def extract_label(x):
    if isinstance(x, (list, np.ndarray)) and len(x) > 0:
        inner = x[0]
        if isinstance(inner, (list, np.ndarray)) and len(inner) > 0:
            return str(inner[0]).strip()
    return "None"


# ── 웨이퍼맵 → 컬러 png 변환 ─────────────────────────
def wm_to_img(wm_array, size=IMG_SIZE):
    """
    픽셀값 0/1/2 → RGB 이미지 변환 후 128×128 리사이즈
    0: 배경 (검정)
    1: 정상 (회색)
    2: 불량 (빨강)
    """
    wm = np.array(wm_array, dtype=np.uint8)
    h, w = wm.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[wm == 1] = [200, 200, 200]  # 정상: 회색
    img[wm == 2] = [50, 50, 220]  # 불량: 빨강 (BGR 순서)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)


# ── 폴더 생성 함수 ────────────────────────────────────
def make_dirs(dataset_dir, labels):
    """train / val 하위에 클래스별 폴더 생성"""
    for split in ["train", "val"]:
        for label in labels:
            os.makedirs(os.path.join(dataset_dir, split, label), exist_ok=True)


# ── 메인 ─────────────────────────────────────────────
for key, file_path in FILES.items():
    if not os.path.exists(file_path):
        print(f"❌ 파일 없음: {file_path}")
        continue

    print(f"\n{'='*60}")
    print(f"▶ [{key}안] 로딩 중: {file_path}")
    df = pd.read_pickle(file_path)
    df = df.reset_index(drop=True)  # 인덱스 초기화
    print(f"   전체 샘플: {len(df):,}개")

    # 라벨 추출
    df["label"] = df["failureType"].apply(extract_label)
    labels = sorted(df["label"].unique())
    print(f"   클래스: {labels}")

    # 출력 폴더 생성
    dataset_dir = os.path.join(base_dir, f"dataset_{key}")
    make_dirs(dataset_dir, labels)

    # train / val Stratified Split
    train_df, val_df = train_test_split(
        df,
        test_size=VAL_RATIO,
        random_state=RANDOM_SEED,
        stratify=df["label"],  # 클래스 비율 유지
    )
    print(f"   train: {len(train_df):,}개 / val: {len(val_df):,}개")

    # 이미지 저장
    for split, split_df in [("train", train_df), ("val", val_df)]:
        print(f"\n   [{split}] 이미지 저장 중...")
        count = 0
        for df_idx, row in split_df.iterrows():
            label = row["label"]
            img = wm_to_img(row["waferMap"])
            filename = f"{df_idx:06d}{IMG_EXT}"
            save_path = os.path.join(dataset_dir, split, label, filename)
            cv2.imwrite(save_path, img)
            count += 1
            if count % 5000 == 0:
                print(f"      {count:,}개 완료...")
        print(f"      [{split}] 총 {count:,}개 저장 완료")

    # 결과 요약 출력
    print(f"\n   {'클래스':<14} {'train':>8} {'val':>8}")
    print(f"   {'-'*35}")
    for label in labels:
        train_cnt = len(os.listdir(os.path.join(dataset_dir, "train", label)))
        val_cnt = len(os.listdir(os.path.join(dataset_dir, "val", label)))
        print(f"   {label:<14} {train_cnt:>8,} {val_cnt:>8,}")
    print(f"   {'-'*35}")
    print(f"✅ 저장 완료: {dataset_dir}")

print(f"\n{'='*60}")
print("모든 변환 완료!")
print(f"{'='*60}")
