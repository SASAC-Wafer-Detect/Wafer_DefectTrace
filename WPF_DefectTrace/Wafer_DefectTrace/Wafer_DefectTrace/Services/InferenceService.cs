using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Generic;

// System.Drawing.Size 와 충돌 방지
using CvSize = OpenCvSharp.Size;
using System.IO;
using System.Linq;

namespace Wafer_DefectTrace.Services
{
    public class InferenceService : IDisposable
    {
        private readonly InferenceSession _session;

        // YOLO11 분류 클래스 순서 (dataset 폴더 알파벳 순)
        // ConvertToYOLO.py 에서 sorted() 로 정렬한 순서와 동일해야 함
        private static readonly string[] Labels = new[]
        {
            "Center", "Donut", "Edge-Loc", "Edge-Ring",
            "Loc", "Near-full", "None", "Random", "Scratch"
        };

        private const int IMG_SIZE = 128; // 학습 시 사용한 해상도

        public InferenceService(string modelPath)
        {
            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"ONNX 모델 파일을 찾을 수 없습니다: {modelPath}");

            _session = new InferenceSession(modelPath);
        }

        // ── Letterbox 리사이즈 ────────────────────────
        // YOLO 학습 시 내부 전처리와 동일하게 구현
        // 비율 유지하면서 목표 크기에 맞추고 남는 부분은 패딩
        private Mat Letterbox(Mat src, int targetSize = 128)
        {
            int srcH = src.Rows;
            int srcW = src.Cols;

            // 비율 계산 (가로/세로 중 작은 쪽 기준)
            float ratio = Math.Min((float)targetSize / srcH,
                                   (float)targetSize / srcW);

            int newW = (int)Math.Round(srcW * ratio);
            int newH = (int)Math.Round(srcH * ratio);

            // 비율 유지 리사이즈
            using var resized = new Mat();
            Cv2.Resize(src, resized, new CvSize(newW, newH),
                       interpolation: InterpolationFlags.Linear);

            // 패딩 크기 계산 (중앙 정렬)
            int padTop = (targetSize - newH) / 2;
            int padBottom = targetSize - newH - padTop;
            int padLeft = (targetSize - newW) / 2;
            int padRight = targetSize - newW - padLeft;

            // 회색(114) 패딩 추가 (YOLO 기본값)
            var result = new Mat();
            Cv2.CopyMakeBorder(resized, result,
                padTop, padBottom, padLeft, padRight,
                BorderTypes.Constant,
                new Scalar(114, 114, 114));

            return result;
        }

        // ── 추론 ──────────────────────────────────────
        public (string label, float confidence, float[] allProbs) Predict(string imagePath)
        {
            // 1. OpenCvSharp4 로 이미지 로드
            using var src = Cv2.ImRead(imagePath, ImreadModes.Color);
            if (src.Empty())
                throw new InvalidOperationException($"이미지 로드 실패: {imagePath}");

            // 2. Letterbox 리사이즈 (YOLO 학습 시와 동일한 전처리)
            using var letterboxed = Letterbox(src, IMG_SIZE);

            // 3. float32 변환 + /255.0 정규화
            using var floatMat = new Mat();
            letterboxed.ConvertTo(floatMat, MatType.CV_32FC3, 1.0 / 255.0);

            // 4. HWC(BGR) → CHW(RGB) 변환
            float[] tensor = new float[3 * IMG_SIZE * IMG_SIZE];
            for (int c = 0; c < 3; c++)
                for (int h = 0; h < IMG_SIZE; h++)
                    for (int w = 0; w < IMG_SIZE; w++)
                        tensor[c * IMG_SIZE * IMG_SIZE + h * IMG_SIZE + w]
                            = floatMat.At<Vec3f>(h, w)[2 - c]; // BGR→RGB

            // 5. ONNX 추론
            var inputTensor = new DenseTensor<float>(tensor, new[] { 1, 3, IMG_SIZE, IMG_SIZE });
            var inputName = _session.InputMetadata.Keys.First();

            using var results = _session.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
            });

            // 6. 확률 추출 (YOLO11 ONNX 는 내부적으로 Softmax 적용 후 출력)
            var probs = results[0].AsEnumerable<float>().ToArray();

            // 7. 최고 확률 클래스 반환
            int maxIdx = Array.IndexOf(probs, probs.Max());
            return (Labels[maxIdx], probs[maxIdx], probs);
        }

        public void Dispose() => _session?.Dispose();
    }
}