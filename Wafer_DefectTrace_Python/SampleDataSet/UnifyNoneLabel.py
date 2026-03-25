import pandas as pd
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

FILES = [
    os.path.join(base_dir, "wafer_train_data_A.pkl"),
    os.path.join(base_dir, "wafer_train_data_B.pkl"),
]


def unify_none_label(failureType):
    """
    failureType 컬럼 값을 분석해서
    none / 빈값 / 공백 → [['None']] 으로 통일
    나머지는 그대로 유지
    """
    # 빈 리스트 또는 빈 중첩 리스트인 경우
    if not isinstance(failureType, (list, np.ndarray)) or len(failureType) == 0:
        return [["None"]]

    inner = failureType[0]

    # 내부도 비어있는 경우
    if not isinstance(inner, (list, np.ndarray)) or len(inner) == 0:
        return [["None"]]

    label = str(inner[0]).strip()

    # none / 빈문자열 → None 으로 통일
    if label.lower() == "none" or label == "":
        return [["None"]]

    # 나머지는 원본 유지
    return failureType


for file_path in FILES:
    if not os.path.exists(file_path):
        print(f"❌ 파일 없음: {file_path}")
        continue

    print(f"\n▶ 로딩 중: {file_path}")
    df = pd.read_pickle(file_path)
    print(f"   전체 샘플: {len(df):,}개")

    # 변경 전 분포 확인
    def extract_label(x):
        if isinstance(x, (list, np.ndarray)) and len(x) > 0:
            inner = x[0]
            if isinstance(inner, (list, np.ndarray)) and len(inner) > 0:
                return str(inner[0]).strip()
        return ""

    before_labels = df["failureType"].apply(extract_label)
    none_before = int((before_labels.str.lower() == "none").sum())
    empty_before = int((before_labels == "").sum())

    print(f"   변경 전 — none: {none_before:,}개 / 빈값: {empty_before:,}개")

    # 통일 적용
    df["failureType"] = df["failureType"].apply(unify_none_label)

    # 변경 후 확인
    after_labels = df["failureType"].apply(extract_label)
    none_after = int((after_labels == "None").sum())

    print(f"   변경 후 — None: {none_after:,}개")
    print(f"   변경된 샘플 수: {none_before + empty_before:,}개 → None 으로 통일")

    # 덮어쓰기 저장
    df.to_pickle(file_path)
    print(f"✅ 저장 완료: {file_path}")

print("\n" + "=" * 40)
print("모든 파일 처리 완료!")
print("=" * 40)
