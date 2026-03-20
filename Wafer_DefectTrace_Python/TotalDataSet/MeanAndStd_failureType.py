import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def get_stats(wafer_map):
    # 배경(0) 제외, 1:정상, 2:불량
    mask = wafer_map > 0
    total = np.sum(mask)
    if total == 0:
        return np.nan, np.nan

    defect_count = np.sum(wafer_map == 2)
    defect_ratio = (defect_count / total) * 100
    yield_rate = 100 - defect_ratio
    return yield_rate, defect_ratio


base_dir = os.path.dirname(os.path.abspath(__file__))
file_name = os.path.join(base_dir, "LSWMD_Ori.pkl")

if not os.path.exists(file_name):
    print(f"❌ 에러: {file_name} 파일이 현재 폴더에 없습니다.")
else:
    print(f"▶ [{file_name}] 로딩 시작...")

    df_withpattern = pd.read_pickle(file_name)

    # 1. 데이터 준비 및 수치 계산
    df_target = df_withpattern.copy()
    # 리스트 형태의 라벨을 문자열로 변환
    df_target["failure_label"] = df_target["failureType"].apply(
        lambda x: (
            x[0][0]
            if isinstance(x, (list, np.ndarray)) and len(x) > 0 and len(x[0]) > 0
            else "None"
        )
    )

    stats_results = df_target["waferMap"].apply(get_stats)
    df_target["yield"] = [x[0] for x in stats_results]
    df_target["defect_ratio"] = [x[1] for x in stats_results]

    # 2. 통계 요약 (평균, 표준편차, 최소, 최대)
    summary = (
        df_target.groupby("failure_label")
        .agg({"yield": ["mean", "std"], "defect_ratio": ["mean", "std", "min", "max"]})
        .reset_index()
    )

    summary.columns = [
        "Pattern",
        "Avg_Yield",
        "Yield_Std",
        "Avg_Defect",
        "Defect_Std",
        "Min",
        "Max",
    ]
    print(summary.sort_values(by="Avg_Defect", ascending=False))

    # 3. 시각화 (데이터가 얼마나 퍼져있는지 확인)
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="failure_label", y="defect_ratio", data=df_target, palette="Set3")
    plt.title("Defect Ratio Distribution by Failure Type")
    plt.ylabel("Defect Ratio (%)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()
