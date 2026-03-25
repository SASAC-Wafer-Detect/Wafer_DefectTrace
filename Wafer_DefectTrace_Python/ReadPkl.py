import pandas as pd
import numpy as np
import os

# 현재 파일(ReadPkl.py)의 폴더 경로를 자동으로 계산합니다.
base_dir = os.path.dirname(os.path.abspath(__file__))
# 그 폴더 경로와 파일 이름을 합칩니다.
file_path = os.path.join(base_dir, "SampleDataSet/wafer_train_data_A_sampled.pkl")

if not os.path.exists(file_path):
    print(f"❌ 에러: {file_path} 파일이 현재 폴더에 없습니다.")
else:
    print(f"▶ [{file_path}] 로딩 시작...")
    train_data = pd.read_pickle(file_path)

    print("\n" + "=" * 50)
    print("▶ Train 데이터셋 구조 확인")
    print("-" * 50)
    print(train_data.info())  # 전체적인 컬럼 구성과 데이터 타입 확인
    print("-" * 50)

    # 라벨 추출
    def extract_label(x):
        if isinstance(x, (list, np.ndarray)) and len(x) > 0:
            inner = x[0]
            if isinstance(inner, (list, np.ndarray)) and len(inner) > 0:
                return str(inner[0]).strip()
        return "None"

    train_data["failure_label"] = train_data["failureType"].apply(extract_label)

    # 실패 타입별 개수 출력
    print("▶ 실패 타입별 샘플 수")
    print("-" * 50)
    count_df = train_data["failure_label"].value_counts().reset_index()
    count_df.columns = ["Pattern", "Count"]
    count_df["Ratio (%)"] = (count_df["Count"] / len(train_data) * 100).round(2)

    print(count_df.to_string(index=False))
    print("-" * 50)
    print(f"전체 샘플 수: {len(train_data):,}개")
    print("=" * 50)
