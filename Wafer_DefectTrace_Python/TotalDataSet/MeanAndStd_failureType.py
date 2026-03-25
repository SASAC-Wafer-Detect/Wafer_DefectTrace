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
file_name = os.path.join(base_dir, "SampleDataSet/wafer_train_data_B.pkl")

if not os.path.exists(file_name):
    print(f"❌ 에러: {file_name} 파일이 현재 폴더에 없습니다.")
else:
    print(f"▶ [{file_name}] 로딩 시작...")

    df_withpattern = pd.read_pickle(file_name)

    # 1. 데이터 준비 및 수치 계산
    df_target = df_withpattern.copy()

    # 리스트 형태의 라벨을 문자열로 변환 (빈 값 -> None로 처리)
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

    # 2. 통계 요약 (평균, 표준편차, 최소, 최대) + 개수 추가
    summary = (
        df_target.groupby("failure_label")
        .agg(
            Count=("failure_label", "count"),  # 개수 추가
            Avg_Yield=("yield", "mean"),
            Yield_Std=("yield", "std"),
            Avg_Defect=("defect_ratio", "mean"),
            Defect_Std=("defect_ratio", "std"),
            Min=("defect_ratio", "min"),
            Max=("defect_ratio", "max"),
        )
        .reset_index()
    )

    summary.columns = [
        "Pattern",
        "Count",
        "Avg_Yield",
        "Yield_Std",
        "Avg_Defect",
        "Defect_Std",
        "Min",
        "Max",
    ]

    print("\n" + "=" * 70)
    print(summary.sort_values(by="Avg_Defect", ascending=False).to_string(index=False))
    print("=" * 70)

    # 3. 박스플롯 (불량 비율 분포)
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="failure_label", y="defect_ratio", data=df_target, palette="Set3")
    plt.title("Defect Ratio Distribution by Failure Type")
    plt.ylabel("Defect Ratio (%)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    # 4. 패턴별 개수 바차트 추가
    count_data = summary.sort_values(by="Count", ascending=False)
    plt.figure(figsize=(12, 5))
    bars = plt.bar(
        count_data["Pattern"],
        count_data["Count"],
        color=sns.color_palette("Set3", len(count_data)),
    )
    plt.bar_label(bars, fmt="%d", padding=3, fontsize=9)
    plt.title("Sample Count by Failure Pattern")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
