import pandas as pd
import numpy as np
import cv2
import os

# ── 설정 ─────────────────────────────────────────────
SAMPLES_PER_CLASS = 20  # 패턴별 추출 개수
IMG_SIZE = 128  # 128×128 리사이즈
RANDOM_SEED = 42
INCLUDE_NONE = True  # None 샘플 포함 여부 (WPF 스킵 테스트)
NONE_SAMPLES = 30  # None 은 따로 개수 지정

base_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(base_dir, "wafer_test_data.pkl")
OUTPUT_DIR = os.path.join(base_dir, "test_images")


# ── 라벨 추출 (none/빈값 → None 통일) ───────────────
def extract_label(x):
    if isinstance(x, (list, np.ndarray)) and len(x) > 0:
        inner = x[0]
        if isinstance(inner, (list, np.ndarray)) and len(inner) > 0:
            label = str(inner[0]).strip()
            if label.lower() == "none" or label == "":
                return "None"
            return label
    return "None"


# ── 웨이퍼맵 → 컬러 이미지 변환 ──────────────────────
def wm_to_img(wm_array, size=IMG_SIZE):
    """
    픽셀값 0/1/2 → BGR 이미지
    0: 배경 (검정)
    1: 정상 (회색)
    2: 불량 (빨강) → BGR: [50, 50, 220]
    """
    wm = np.array(wm_array, dtype=np.uint8)
    h, w = wm.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[wm == 1] = [200, 200, 200]
    img[wm == 2] = [50, 50, 220]  # BGR 순서
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)


# ── 메인 ─────────────────────────────────────────────
if not os.path.exists(INPUT_PATH):
    print(f"❌ 파일 없음: {INPUT_PATH}")
else:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"▶ 로딩 중: {INPUT_PATH}")
    df = pd.read_pickle(INPUT_PATH)
    df = df.reset_index(drop=True)
    df["label"] = df["failureType"].apply(extract_label)

    print(f"▶ 전체 샘플: {len(df):,}개")
    print(f"▶ 패턴 분포:")
    print(df["label"].value_counts().to_string())

    # 패턴별 샘플링
    selected = []

    for label, group in df.groupby("label"):
        if label == "None":
            if not INCLUDE_NONE:
                continue
            n = min(NONE_SAMPLES, len(group))
        else:
            n = min(SAMPLES_PER_CLASS, len(group))

        sampled = group.sample(n=n, random_state=RANDOM_SEED)
        selected.append(sampled)
        print(f"   {label:<12}: {n}개 선택")

    df_selected = (
        pd.concat(selected)
        .sample(frac=1, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )

    # 이미지 저장
    print(f"\n▶ 이미지 저장 중... (총 {len(df_selected)}개)")
    saved = 0
    for i, (_, row) in enumerate(df_selected.iterrows()):
        label = row["label"]
        img = wm_to_img(row["waferMap"])

        # 파일명: wafer_000001_Scratch.png (라벨 포함 → 정답 확인용)
        fname = f"wafer_{i+1:06d}_{label}.png"
        cv2.imwrite(os.path.join(OUTPUT_DIR, fname), img)
        saved += 1

    print(f"\n{'='*45}")
    print(f"✅ 완료!")
    print(f"   저장 경로: {OUTPUT_DIR}")
    print(f"   저장 개수: {saved}개")
    print(f"   패턴별 구성:")
    label_counts = df_selected["label"].value_counts()
    for label, cnt in label_counts.items():
        print(f"      {label:<12}: {cnt}개")
    print(f"{'='*45}")
    print(f"\n💡 WPF 에서 이 폴더를 선택해서 테스트하세요.")
    print(f"   None 파일은 자동 스킵됩니다.")
