using System;
using System.ComponentModel;

namespace Wafer_DefectTrace.Models
{
    // ── 추론 결과 ─────────────────────────────────────
    public class DiagnosisResult
    {
        public string PatternName { get; set; } = string.Empty;
        public float Confidence { get; set; }
        public string ProcessName { get; set; } = string.Empty;
        public int Severity { get; set; }
        public string[] CauseOptions { get; set; } = Array.Empty<string>();
    }

    // ── 이력 항목 ─────────────────────────────────────
    public class HistoryItem
    {
        public string Time { get; set; } = string.Empty;
        public string Pattern { get; set; } = string.Empty;
        public string Confidence { get; set; } = string.Empty;
        public string SelectedCause { get; set; } = string.Empty;
        public string FileName { get; set; } = string.Empty;
    }

    // ── 원인 선택 버튼 항목 ───────────────────────────
    public class CauseOption : INotifyPropertyChanged
    {
        public string Label { get; set; } = string.Empty;

        private bool _isSelected;
        public bool IsSelected
        {
            get => _isSelected;
            set
            {
                _isSelected = value;
                OnPropertyChanged(nameof(IsSelected));
                OnPropertyChanged(nameof(DisplayLabel));
                OnPropertyChanged(nameof(TextColor));
            }
        }

        // 선택 상태에 따라 표시 텍스트 변경
        public string DisplayLabel => IsSelected ? $"✓  {Label}" : $"    {Label}";

        // 선택 상태에 따라 텍스트 색상 변경
        public string TextColor => IsSelected ? "#FFFFFF" : "#A0B4C8";

        public event PropertyChangedEventHandler? PropertyChanged;
        protected void OnPropertyChanged(string name)
            => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
    }
}