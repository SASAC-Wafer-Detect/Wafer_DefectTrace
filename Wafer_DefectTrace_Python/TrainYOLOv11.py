"""
TrainYOLO.py
────────────
YOLO11n-cls 파인튜닝 (학습만)
- A안 / B안 데이터셋 각각 학습
- patience=10 (EarlyStopping)
- 학습 완료 후 best.pt → ONNX 변환 (opset=13)
- 평가는 EvaluateYOLO.py 에서 별도 진행
"""

from ultralytics import YOLO
import os

# ── 설정 ─────────────────────────────────────────────
EPOCHS = 50
IMG_SIZE = 128
BATCH = 64
PATIENCE = 10
DEVICE = 0  # GPU 없으면 "cpu"
RANDOM_SEED = 42

base_dir = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    "A": os.path.join(base_dir, "dataset_A"),
    "B": os.path.join(base_dir, "dataset_B"),
}

# ── 메인 ─────────────────────────────────────────────
if __name__ == "__main__":
    for key, dataset_path in DATASETS.items():
        if not os.path.exists(dataset_path):
            print(f"❌ 데이터셋 없음: {dataset_path}")
            continue

        print(f"\n{'='*55}")
        print(f"▶ [{key}안] 학습 시작")
        print(f"   데이터셋: {dataset_path}")
        print(f"   epochs={EPOCHS}, imgsz={IMG_SIZE}, patience={PATIENCE}")
        print(f"{'='*55}")

        # YOLO11n-cls 사전학습 가중치 로드
        model = YOLO("yolo11n-cls.pt")

        # 학습
        model.train(
            data=dataset_path,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH,
            patience=PATIENCE,
            device=DEVICE,
            seed=RANDOM_SEED,
            project=os.path.join(base_dir, "runs"),
            name=f"wafer_{key}",
            exist_ok=True,
            # ── 기하학적 증강 (웨이퍼 방향 무관 → 유지) ──
            degrees=90.0,  # 90도 단위 회전
            fliplr=0.5,  # 좌우 반전
            flipud=0.5,  # 상하 반전
            # ── 색상/손실 증강 (범주형 데이터 → 전부 끄기) ──
            hsv_h=0.0,  # 색조 변환 끄기 (빨강=불량 의미 보존)
            hsv_s=0.0,  # 채도 변환 끄기
            hsv_v=0.0,  # 밝기 변환 끄기
            erasing=0.0,  # 랜덤 지우기 끄기 (Scratch/Loc 패턴 소실 방지)
            auto_augment=False,  # 추가 자동 색상 증강 끄기
        )

        print(f"\n▶ [{key}안] 학습 완료")

        # best.pt → ONNX 변환
        best_pt = os.path.join(base_dir, "runs", f"wafer_{key}", "weights", "best.pt")
        best_model = YOLO(best_pt)
        best_model.export(format="onnx", opset=13, imgsz=IMG_SIZE)
        print(f"✅ ONNX 변환 완료: {best_pt.replace('.pt', '.onnx')}")

    print(f"\n{'='*55}")
    print("모든 학습 완료! 평가는 EvaluateYOLO.py 를 실행하세요.")
    print(f"{'='*55}")
