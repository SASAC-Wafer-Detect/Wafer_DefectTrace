"""
SamplingDataSet.py
──────────────────
wafer_train_data_A.pkl / wafer_train_data_B.pkl 에 대해
- 2만개 이상 클래스 → 5,000개로 다운샘플링
- 500개 이하 클래스 → 원본 유지 + 부족분은 복원 추출 후 증강으로 500개 업샘플링
결과를 wafer_train_data_A_sampled.pkl / wafer_train_data_B_sampled.pkl 로 저장
"""

import pandas as pd
import numpy as np
import os
import albumentations as A

# ── 설정 ─────────────────────────────────────────────
RANDOM_SEED = 42
DOWN_THRESHOLD = 20000  # 이 개수 이상이면 다운샘플링
DOWN_TARGET = 5000  # 다운샘플링 목표 개수
UP_THRESHOLD = 500  # 이 개수 이하면 업샘플링
UP_TARGET = 500  # 업샘플링 목표 개수

base_dir = os.path.dirname(os.path.abspath(__file__))
sample_dir = os.path.join(base_dir)

FILES = {
    "A": os.path.join(sample_dir, "wafer_train_data_A.pkl"),
    "B": os.path.join(sample_dir, "wafer_train_data_B.pkl"),
}

# ── 증강 파이프라인 ───────────────────────────────────
# 웨이퍼는 방향이 의미 없으므로 회전/반전 모두 허용
augmentor = A.Compose(
    [
        A.Rotate(limit=180, p=0.8),  # 180도 범위 회전
        A.HorizontalFlip(p=0.5),  # 좌우 반전
        A.VerticalFlip(p=0.5),  # 상하 반전
    ]
)


# ── 라벨 추출 함수 ────────────────────────────────────
def extract_label(x):
    if isinstance(x, (list, np.ndarray)) and len(x) > 0:
        inner = x[0]
        if isinstance(inner, (list, np.ndarray)) and len(inner) > 0:
            return str(inner[0]).strip()
    return "None"


# ── 웨이퍼맵 증강 함수 ───────────────────────────────
def augment_wafermap(wm_array):
    """
    웨이퍼맵(0/1/2 값) 에 증강 적용
    원본 크기 그대로 유지 (크기 통일은 YOLOv8 학습 단계에서 처리)
    1. float32 변환 후 3채널로 확장 (albumentations 입력 형식)
    2. 증강 적용
    3. 반올림으로 픽셀값 0/1/2 복원
    """
    wm = np.array(wm_array, dtype=np.float32)
    wm_3ch = np.stack([wm, wm, wm], axis=-1)  # HWC 3채널
    aug = augmentor(image=wm_3ch)["image"]
    wm_out = np.round(aug[:, :, 0]).astype(np.uint8)
    wm_out = np.clip(wm_out, 0, 2)
    return wm_out.tolist()


# ── 업샘플링 함수 (원본 유지 + 부족분 증강) ──────────
def upsample_with_augmentation(group, target_n, label):
    """
    1단계: 원본 그대로 유지
    2단계: 부족한 만큼 복원 추출 후 증강 적용
    → 완전 중복 방지 + 다양성 확보
    """
    original_n = len(group)
    need_n = target_n - original_n

    if need_n <= 0:
        return group

    # 부족한 만큼 복원 추출
    extra = group.sample(n=need_n, replace=True, random_state=RANDOM_SEED).copy()

    # 증강 적용
    print(
        f"      [{label}] 증강 중... ({original_n}개 → {target_n}개, +{need_n}개 생성)"
    )
    extra["waferMap"] = extra["waferMap"].apply(augment_wafermap)

    # 원본 + 증강 합치기
    result = pd.concat([group, extra], ignore_index=True)
    return result


# ── 샘플링 함수 ───────────────────────────────────────
def apply_sampling(df, label_col="failure_label"):
    groups = []
    summary = []

    for label, group in df.groupby(label_col):
        n = len(group)

        if n >= DOWN_THRESHOLD:
            # 다운샘플링
            sampled = group.sample(n=DOWN_TARGET, random_state=RANDOM_SEED)
            action = f"다운샘플링 ({n:,} → {DOWN_TARGET:,})"

        elif n <= UP_THRESHOLD:
            # 업샘플링 (원본 + 증강)
            sampled = upsample_with_augmentation(group, UP_TARGET, label)
            action = f"업샘플링+증강 ({n:,} → {UP_TARGET:,})"

        else:
            # 그대로 유지
            sampled = group
            action = f"유지 ({n:,}개)"

        groups.append(sampled)
        summary.append(
            {
                "Pattern": label,
                "Before": n,
                "After": len(sampled),
                "Action": action,
            }
        )

    df_sampled = (
        pd.concat(groups)
        .sample(frac=1, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )
    return df_sampled, summary


# ── 메인 ─────────────────────────────────────────────
for key, file_path in FILES.items():
    if not os.path.exists(file_path):
        print(f"❌ 파일 없음: {file_path}")
        continue

    print(f"\n{'='*60}")
    print(f"▶ [{key}안] 로딩 중: {file_path}")
    df = pd.read_pickle(file_path)
    print(f"   전체 샘플: {len(df):,}개")

    # 라벨 추출
    df["failure_label"] = df["failureType"].apply(extract_label)

    # 샘플링 적용
    df_sampled, summary = apply_sampling(df)

    # 결과 출력
    print(f"\n   {'Pattern':<14} {'Before':>10} {'After':>8}  Action")
    print(f"   {'-'*60}")
    for s in sorted(summary, key=lambda x: -x["Before"]):
        print(
            f"   {s['Pattern']:<14} {s['Before']:>10,} {s['After']:>8,}  {s['Action']}"
        )
    print(f"   {'-'*60}")
    print(f"   총합: {len(df):,}개 → {len(df_sampled):,}개")

    # 임시 컬럼 제거 후 저장
    df_sampled = df_sampled.drop(columns=["failure_label"])
    out_path = os.path.join(sample_dir, f"wafer_train_data_{key}_sampled.pkl")
    df_sampled.to_pickle(out_path)
    print(f"\n✅ 저장 완료: {out_path}")

print(f"\n{'='*60}")
print("모든 파일 처리 완료!")
print(f"{'='*60}")
