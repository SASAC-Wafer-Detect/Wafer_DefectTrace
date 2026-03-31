using System.Collections.Generic;
using Wafer_DefectTrace.Models;

namespace Wafer_DefectTrace.Services
{
    public static class ProcessMapper
    {
        private static readonly Dictionary<string, (string Process, string[] Causes, int Severity)> _map
            = new()
            {
                ["Center"] = (
                "CVD / 박막 증착",
                new[] { "증착 헤드 중앙부 불균일", "챔버 내 가스 흐름 불균형", "웨이퍼 척 온도 불균일" },
                3),
                ["Donut"] = (
                "포토레지스트 / CMP",
                new[] { "레지스트 재증착 현상", "CMP 연마 패드 경화", "슬러리 공급 불균형" },
                2),
                ["Edge-Loc"] = (
                "포토리소그래피",
                new[] { "에지 노광 정렬 불량", "포커스 맵 에지 편차", "레티클 에지 오염" },
                2),
                ["Edge-Ring"] = (
                "건식 에칭 / 어닐링",
                new[] { "플라즈마 불균일 (에지 집중)", "급속 어닐링 온도 에지 편차", "에지 링 마모 또는 오염" },
                3),
                ["Loc"] = (
                "이온 주입",
                new[] { "국소 파티클 오염", "마스크 핀홀 불량", "이온 빔 국소 집중" },
                2),
                ["Near-full"] = (
                "CVD / 배치 전체",
                new[] { "배치 단위 대규모 오염", "챔버 세정 불충분", "전구체 가스 오염" },
                4),
                ["Random"] = (
                "클린룸 / ESD",
                new[] { "정전기 방전(ESD) 손상", "클린룸 파티클 낙하", "화학 오염(미량)" },
                1),
                ["Scratch"] = (
                "웨이퍼 이송 장비",
                new[] { "로봇 암 / 척 접촉 결함", "웨이퍼 이송 속도 불량", "척 클램핑 압력 이상" },
                4),
                ["Normal"] = ("—", new string[] { }, 0),
                ["None"] = ("—", new string[] { }, 0),
            };

        public static DiagnosisResult GetResult(string pattern, float confidence)
        {
            if (!_map.TryGetValue(pattern, out var info))
                return new DiagnosisResult
                {
                    PatternName = pattern,
                    Confidence = confidence,
                    ProcessName = "알 수 없음",
                    CauseOptions = new string[] { },
                    Severity = 0
                };

            return new DiagnosisResult
            {
                PatternName = pattern,
                Confidence = confidence,
                ProcessName = info.Process,
                CauseOptions = info.Causes,
                Severity = info.Severity,
            };
        }

        public static string GetSeverityText(int severity) => severity switch
        {
            4 => "★★★★  매우 심각 — 즉시 라인 스탑",
            3 => "★★★☆  심각 — 긴급 점검 필요",
            2 => "★★☆☆  주의 — 점검 권고",
            1 => "★☆☆☆  경미 — 모니터링",
            _ => "정상 웨이퍼",
        };

        public static string GetSeverityColor(int severity) => severity switch
        {
            4 => "#A32D2D",
            3 => "#C45911",
            2 => "#C4A011",
            1 => "#375623",
            _ => "#555555",
        };
    }
}