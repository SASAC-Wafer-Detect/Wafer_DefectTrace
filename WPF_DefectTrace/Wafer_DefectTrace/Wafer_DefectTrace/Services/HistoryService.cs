using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Wafer_DefectTrace.Models;

namespace Wafer_DefectTrace.Services
{
    public class HistoryService
    {
        private readonly string _csvPath;
        private const string Header = "진단시간,패턴명,신뢰도,선택한원인,파일명";

        public HistoryService(string csvPath = "wafer_history.csv")
            => _csvPath = csvPath;

        // 이력 추가 저장
        public void Append(HistoryItem item)
        {
            bool exists = File.Exists(_csvPath);
            using var sw = new StreamWriter(_csvPath, append: true, Encoding.UTF8);
            if (!exists) sw.WriteLine(Header);
            sw.WriteLine($"{E(item.Time)},{E(item.Pattern)},{E(item.Confidence)},{E(item.SelectedCause)},{E(item.FileName)}");
        }

        // 전체 로드 (최신순)
        public List<HistoryItem> LoadAll()
        {
            if (!File.Exists(_csvPath)) return new();
            var lines = File.ReadAllLines(_csvPath, Encoding.UTF8);
            var result = lines.Skip(1).Select(Parse).Where(x => x != null).Cast<HistoryItem>().ToList();
            result.Reverse();
            return result;
        }

        // 특정 패턴만 필터
        public List<HistoryItem> LoadByPattern(string pattern)
            => LoadAll().Where(h => h.Pattern == pattern).ToList();

        // ── 내부 헬퍼 ─────────────────────────────────
        private string E(string v)
        {
            if (string.IsNullOrEmpty(v)) return "";
            return (v.Contains(',') || v.Contains('"') || v.Contains('\n'))
                ? $"\"{v.Replace("\"", "\"\"")}\"" : v;
        }

        private HistoryItem? Parse(string line)
        {
            var parts = SplitCsv(line);
            if (parts.Length < 5) return null;
            return new HistoryItem
            {
                Time = parts[0],
                Pattern = parts[1],
                Confidence = parts[2],
                SelectedCause = parts[3],
                FileName = parts[4]
            };
        }

        private string[] SplitCsv(string line)
        {
            var fields = new List<string>();
            var sb = new StringBuilder();
            bool inQ = false;
            for (int i = 0; i < line.Length; i++)
            {
                char c = line[i];
                if (c == '"') { if (inQ && i + 1 < line.Length && line[i + 1] == '"') { sb.Append('"'); i++; } else inQ = !inQ; }
                else if (c == ',' && !inQ) { fields.Add(sb.ToString()); sb.Clear(); }
                else sb.Append(c);
            }
            fields.Add(sb.ToString());
            return fields.ToArray();
        }
    }
}