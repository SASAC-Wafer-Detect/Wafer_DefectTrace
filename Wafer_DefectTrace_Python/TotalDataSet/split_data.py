import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 1. 파일 경로 및 존재 여부 확인
file_name = "LSWMD_Ori.pkl"

if not os.path.exists(file_name):
    print(f"❌ 에러: {file_name} 파일이 현재 폴더에 없습니다.")
else:
    print(f"▶ [{file_name}] 로딩 시작...")

    data = pd.read_pickle(file_name)
    print(f"▶ 로딩 완료. 총 데이터 개수: {len(data)}개")

    # 2. 데이터 분할 (70% : 30%)
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

    # 3. 분할된 데이터 저장 (Pandas 전용 to_pickle 사용)
    print("▶ 분할 데이터 저장 중... 잠시만 기다려 주세요.")

    train_data.to_pickle("wafer_train_data.pkl")
    test_data.to_pickle("wafer_test_data.pkl")

    print("-" * 30)
    print(f"✅ 모든 작업 완료!")
    print(f"- Train 세트: {len(train_data)}개 저장됨")
    print(f"- Test 세트 : {len(test_data)}개 저장됨")
    print("-" * 30)
