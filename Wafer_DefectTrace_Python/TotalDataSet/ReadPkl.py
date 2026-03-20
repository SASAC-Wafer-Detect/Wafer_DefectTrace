import pandas as pd
import os

# 현재 파일(ReadPkl.py)의 폴더 경로를 자동으로 계산합니다.
base_dir = os.path.dirname(os.path.abspath(__file__))
# 그 폴더 경로와 파일 이름을 합칩니다.
file_path = os.path.join(base_dir, "wafer_test_data.pkl")

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
    print(train_data.head())  # 실제 데이터 샘플 5행 출력
    print("=" * 50)
