"""
EvaluateYOLO.py
───────────────
학습된 best.pt 로 평가만 진행
- Macro F1 계산
- 클래스별 F1 출력
- Confusion Matrix 시각화 (개수 / 비율 2가지)
- A안 vs B안 최종 비교
"""

from ultralytics import YOLO
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import platform


# ── 한글 폰트 ─────────────────────────────────────────
def set_korean_font():
    system = platform.system()
    if system == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    elif system == "Darwin":
        plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False


set_korean_font()

# ── 설정 ─────────────────────────────────────────────
base_dir = os.path.dirname(os.path.abspath(__file__))

TARGETS = {
    "A": {
        "model": os.path.join(base_dir, "runs", "wafer_A", "weights", "best.pt"),
        "dataset": os.path.join(base_dir, "dataset_A"),
    },
    "B": {
        "model": os.path.join(base_dir, "runs", "wafer_B", "weights", "best.pt"),
        "dataset": os.path.join(base_dir, "dataset_B"),
    },
}


# ── Confusion Matrix 시각화 ───────────────────────────
def plot_confusion_matrix(y_true, y_pred, labels, key):
    """
    Confusion Matrix 히트맵 시각화
    - 왼쪽: 개수 버전
    - 오른쪽: 비율(%) 버전 (행 기준 정규화)
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float)
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1  # 0으로 나누기 방지
    cm_norm = cm_norm / row_sum * 100

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 왼쪽: 개수
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[0],
        linewidths=0.5,
    )
    axes[0].set_title(f"[{key}안] Confusion Matrix (개수)")
    axes[0].set_xlabel("예측 클래스")
    axes[0].set_ylabel("실제 클래스")
    axes[0].tick_params(axis="x", rotation=45)

    # 오른쪽: 비율 (%)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[1],
        linewidths=0.5,
        vmin=0,
        vmax=100,
    )
    axes[1].set_title(f"[{key}안] Confusion Matrix (%)")
    axes[1].set_xlabel("예측 클래스")
    axes[1].set_ylabel("실제 클래스")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()

    # 저장
    save_path = os.path.join(
        base_dir, "runs", f"wafer_{key}", f"confusion_matrix_{key}.png"
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"   Confusion Matrix 저장: {save_path}")
    plt.show()
    plt.close()


# ── 평가 함수 ─────────────────────────────────────────
def evaluate(model_path, dataset_path, key):
    """
    val 폴더에서 이미지 수집 후 예측
    → Macro F1 / 클래스별 F1 / Confusion Matrix 출력
    """
    if not os.path.exists(model_path):
        print(f"❌ 모델 없음: {model_path}")
        return None

    print(f"\n{'='*55}")
    print(f"▶ [{key}안] 평가 시작")
    print(f"   모델: {model_path}")
    print(f"{'='*55}")

    model = YOLO(model_path)
    val_path = os.path.join(dataset_path, "val")

    # 클래스별 폴더에서 이미지 경로 + 실제 라벨 수집
    img_paths, y_true = [], []
    for class_name in sorted(os.listdir(val_path)):
        class_dir = os.path.join(val_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                img_paths.append(os.path.join(class_dir, fname))
                y_true.append(class_name)

    print(f"   평가 이미지 수: {len(img_paths):,}개")

    # 예측
    results = model.predict(img_paths, verbose=False)
    y_pred = [r.names[r.probs.top1] for r in results]

    # 클래스 목록
    labels = sorted(list(set(y_true)))
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # 클래스별 F1 출력
    print("\n" + "=" * 55)
    print(f"[{key}안] 클래스별 F1-Score")
    print("=" * 55)
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    print(f">>> Macro F1-Score: {macro_f1:.4f}")
    print("=" * 55)

    # Confusion Matrix 시각화
    plot_confusion_matrix(y_true, y_pred, labels, key)

    return macro_f1


# ── 메인 ─────────────────────────────────────────────
if __name__ == "__main__":
    results_summary = {}

    for key, target in TARGETS.items():
        macro_f1 = evaluate(target["model"], target["dataset"], key)
        if macro_f1 is not None:
            results_summary[key] = macro_f1

    # A안 vs B안 최종 비교
    if results_summary:
        print(f"\n{'='*55}")
        print("A안 vs B안 Macro F1 최종 비교")
        print(f"{'='*55}")
        for key, f1 in results_summary.items():
            print(f"   {key}안 Macro F1: {f1:.4f}")

        best_key = max(results_summary, key=results_summary.get)
        print(
            f"\n>>> 최적 데이터셋: {best_key}안"
            f" (Macro F1: {results_summary[best_key]:.4f})"
        )
        print(f"{'='*55}")
