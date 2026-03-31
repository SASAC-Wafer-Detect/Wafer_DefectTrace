
using Microsoft.Win32;
using System;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media.Imaging;
using Wafer_DefectTrace.Commands;
using Wafer_DefectTrace.Models;
using Wafer_DefectTrace.Services;

using WpfMessageBox = System.Windows.MessageBox;
using WpfOpenFileDialog = Microsoft.Win32.OpenFileDialog;

namespace Wafer_DefectTrace.ViewModels
{
    public class MainViewModel : INotifyPropertyChanged
    {

        #region 앱 시작
        // ── 서비스 ────────────────────────────────────
        private readonly InferenceService _inference;
        private readonly HistoryService _historyService = new();

        // ── 이미지 파일 목록 ──────────────────────────
        private System.Collections.Generic.List<string> _imageFiles = new();
        private int _currentIndex = 0;
        private const string NonePattern = "None";

        // ── 생성자 ────────────────────────────────────
        public MainViewModel()
        {
            _inference = new InferenceService("wafer_model.onnx");

            // Command 초기화
            SelectFolderCommand = new RelayCommand(ExecuteSelectFolder);
            SaveCommand = new RelayCommand(_ => ExecuteSave(null), _ => CanSave);
            SkipCommand = new RelayCommand(_ => ExecuteSkip(null), _ => HasResult);
            ResetCommand = new RelayCommand(ExecuteReset);
            CauseSelectCommand = new RelayCommand(ExecuteCauseSelect);

            // 기존 이력 로드
            AllHistory = new ObservableCollection<HistoryItem>(
                _historyService.LoadAll());
        }

        // ══════════════════════════════════════════════
        // Commands
        // ══════════════════════════════════════════════

        public ICommand SelectFolderCommand { get; }
        public ICommand SaveCommand { get; }
        public ICommand SkipCommand { get; }
        public ICommand ResetCommand { get; }
        public ICommand CauseSelectCommand { get; }


        // ── 폴더 선택 ────────────────────────────────
        private void ExecuteSelectFolder(object? _)
        {
            var dlg = new WpfOpenFileDialog
            {
                Title = "폴더 안의 아무 이미지 파일을 선택하세요",
                Filter = "이미지 파일|*.png;*.jpg;*.jpeg;*.bmp",
                CheckFileExists = false,
                FileName = "폴더 선택",
            };
            if (dlg.ShowDialog() != true) return;

            var folder = Path.GetDirectoryName(dlg.FileName);
            if (folder == null) return;

            LoadFolder(folder);
            ProcessNext();
        }

        // ── 저장 후 다음 ─────────────────────────────
        private void ExecuteSave(object? _)
        {
            if (SelectedCause == null)
            {
                WpfMessageBox.Show("원인을 선택한 후 저장해주세요.",
                    "알림", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }
            SaveCurrentResult();
            ProcessNext();
        }

        // ── 건너뛰기 ─────────────────────────────────
        private void ExecuteSkip(object? _)
        {
            SkippedCount++;
            ProcessNext();
        }

        // ── 초기화 ───────────────────────────────────
        private void ExecuteReset(object? _)
        {
            PreviewImage = null;
            Reset();
        }

        // ── 원인 버튼 선택 ────────────────────────────
        private void ExecuteCauseSelect(object? parameter)
        {
            if (parameter is not CauseOption selected) return;

            // 단일 선택 토글
            foreach (var opt in CauseOptions)
                opt.IsSelected = false;
            selected.IsSelected = true;

            OnPropertyChanged(nameof(SelectedCause));
            OnPropertyChanged(nameof(CanSave));
            ((RelayCommand)SaveCommand).RaiseCanExecuteChanged();
        }

        #endregion

        #region 텍스트 데이터 바인딩 속성
        // ══════════════════════════════════════════════
        // 통계 카운터 : `OnPropertyChanged("PatternName")` 호출 시 XAML 이 자동으로 화면 갱신
        // ══════════════════════════════════════════════

        private int _totalCount;
        public int TotalCount
        {
            get => _totalCount;
            set { _totalCount = value; OnPropertyChanged(nameof(TotalCount)); RefreshProgress(); }
        }

        private int _skippedCount;
        public int SkippedCount
        {
            get => _skippedCount;
            set { _skippedCount = value; OnPropertyChanged(nameof(SkippedCount)); RefreshProgress(); }
        }

        private int _noneSkippedCount;
        public int NoneSkippedCount
        {
            get => _noneSkippedCount;
            set { _noneSkippedCount = value; OnPropertyChanged(nameof(NoneSkippedCount)); }
        }

        private int _defectCount;
        public int DefectCount
        {
            get => _defectCount;
            set { _defectCount = value; OnPropertyChanged(nameof(DefectCount)); RefreshProgress(); }
        }

        private int _savedCount;
        public int SavedCount
        {
            get => _savedCount;
            set { _savedCount = value; OnPropertyChanged(nameof(SavedCount)); RefreshProgress(); }
        }

        private double _progressBarMaxWidth = 220;
        public string ProgressText
            => _totalCount == 0 ? "0 / 0" : $"{_currentIndex} / {_totalCount}";
        public double ProgressBarWidth
            => _totalCount == 0 ? 0 : _progressBarMaxWidth * _currentIndex / _totalCount;

        private void RefreshProgress()
        {
            OnPropertyChanged(nameof(ProgressText));
            OnPropertyChanged(nameof(ProgressBarWidth));
        }

        #endregion

        // ══════════════════════════════════════════════
        // 현재 이미지
        // ══════════════════════════════════════════════

        private string _currentFileName = "";
        public string CurrentFileName
        {
            get => _currentFileName;
            set
            {
                _currentFileName = value;
                OnPropertyChanged(nameof(CurrentFileName));
                OnPropertyChanged(nameof(ImageVisibility));
                OnPropertyChanged(nameof(NoImageVisibility));
            }
        }

        private BitmapImage? _previewImage;
        public BitmapImage? PreviewImage
        {
            get => _previewImage;
            set { _previewImage = value; OnPropertyChanged(nameof(PreviewImage)); }
        }

        // ══════════════════════════════════════════════
        // 진단 결과
        // ══════════════════════════════════════════════

        private string _patternName = "";
        public string PatternName
        {
            get => _patternName;
            set { _patternName = value; OnPropertyChanged(nameof(PatternName)); }
        }

        private string _confidence = "";
        public string Confidence
        {
            get => _confidence;
            set { _confidence = value; OnPropertyChanged(nameof(Confidence)); }
        }

        private string _severityText = "";
        public string SeverityText
        {
            get => _severityText;
            set { _severityText = value; OnPropertyChanged(nameof(SeverityText)); }
        }

        private string _severityColor = "#555555";
        public string SeverityColor
        {
            get => _severityColor;
            set { _severityColor = value; OnPropertyChanged(nameof(SeverityColor)); }
        }

        // ══════════════════════════════════════════════
        // 원인 선택
        // ══════════════════════════════════════════════

        private ObservableCollection<CauseOption> _causeOptions = new();
        public ObservableCollection<CauseOption> CauseOptions
        {
            get => _causeOptions;
            set { _causeOptions = value; OnPropertyChanged(nameof(CauseOptions)); }
        }

        public string? SelectedCause
            => CauseOptions.FirstOrDefault(c => c.IsSelected)?.Label;

        public bool CanSave => SelectedCause != null;

        // ══════════════════════════════════════════════
        // UI 상태 (Visibility)
        // ══════════════════════════════════════════════

        private enum UiState { Waiting, Result, Done }
        private UiState _uiState = UiState.Waiting;
        private bool HasResult => _uiState == UiState.Result;

        public Visibility WaitingVisibility
            => _uiState == UiState.Waiting ? Visibility.Visible : Visibility.Collapsed;
        public Visibility ResultVisibility
            => _uiState == UiState.Result ? Visibility.Visible : Visibility.Collapsed;
        public Visibility DoneVisibility
            => _uiState == UiState.Done ? Visibility.Visible : Visibility.Collapsed;
        public Visibility ImageVisibility
            => _currentFileName != "" ? Visibility.Visible : Visibility.Collapsed;
        public Visibility NoImageVisibility
            => _currentFileName == "" ? Visibility.Visible : Visibility.Collapsed;
        public Visibility NoHistoryVisibility
            => FilteredHistory.Count == 0 && _uiState == UiState.Result
               ? Visibility.Visible : Visibility.Collapsed;
        public Visibility HistoryBadgeVisibility
            => FilteredHistory.Count > 0 ? Visibility.Visible : Visibility.Collapsed;

        private void SetUiState(UiState state)
        {
            _uiState = state;
            OnPropertyChanged(nameof(WaitingVisibility));
            OnPropertyChanged(nameof(ResultVisibility));
            OnPropertyChanged(nameof(DoneVisibility));
            OnPropertyChanged(nameof(ImageVisibility));
            OnPropertyChanged(nameof(NoImageVisibility));
            OnPropertyChanged(nameof(NoHistoryVisibility));
            OnPropertyChanged(nameof(HistoryBadgeVisibility));
            ((RelayCommand)SkipCommand).RaiseCanExecuteChanged();
            ((RelayCommand)SaveCommand).RaiseCanExecuteChanged();
        }

        // ══════════════════════════════════════════════
        // 이력
        // ══════════════════════════════════════════════

        public ObservableCollection<HistoryItem> AllHistory { get; }

        private ObservableCollection<HistoryItem> _filteredHistory = new();
        public ObservableCollection<HistoryItem> FilteredHistory
        {
            get => _filteredHistory;
            set { _filteredHistory = value; OnPropertyChanged(nameof(FilteredHistory)); }
        }

        public string HistoryBadgeText
            => $"{PatternName} 패턴 {FilteredHistory.Count}회 발생";
        public string DoneSummaryText
            => $"불량 감지 {_defectCount}건 / 저장 완료 {_savedCount}건";

        // ══════════════════════════════════════════════
        // 내부 로직
        // ══════════════════════════════════════════════

        private void LoadFolder(string folderPath)
        {
            var exts = new[] { ".png", ".jpg", ".jpeg", ".bmp" };
            _imageFiles = System.IO.Directory.GetFiles(folderPath)
                .Where(f => exts.Contains(Path.GetExtension(f).ToLower()))
                .OrderBy(f => f)
                .ToList();

            _currentIndex = 0;
            TotalCount = _imageFiles.Count;
            SkippedCount = DefectCount = SavedCount = 0;
        }

        private void ProcessNext()
        {
            while (true)
            {
                if (_currentIndex >= _imageFiles.Count)
                {
                    CurrentFileName = "";
                    PreviewImage = null;
                    OnPropertyChanged(nameof(DoneSummaryText));
                    SetUiState(UiState.Done);
                    return;
                }

                var path = _imageFiles[_currentIndex++];
                RefreshProgress();

                try
                {
                    // 이미지 미리보기
                    PreviewImage = new BitmapImage(new Uri(path));
                    CurrentFileName = Path.GetFileName(path);

                    // 추론
                    var (label, confidence, _) = _inference.Predict(path);

                    // None → 자동 스킵
                    if (label == NonePattern)
                    {
                        NoneSkippedCount++; 
                        continue;
                    }

                    // 불량 → 결과 표시
                    var result = ProcessMapper.GetResult(label, confidence);
                    ShowResult(result);
                    return;
                }
                catch (Exception)
                {
                    SkippedCount++;
                    continue;
                }
            }
        }

        private void ShowResult(DiagnosisResult result)
        {
            PatternName = result.PatternName;
            Confidence = $"{result.Confidence * 100:F1}%";
            SeverityText = ProcessMapper.GetSeverityText(result.Severity);
            SeverityColor = ProcessMapper.GetSeverityColor(result.Severity);

            CauseOptions = new ObservableCollection<CauseOption>(
                result.CauseOptions.Select(c => new CauseOption { Label = c }));

            DefectCount++;
            SetUiState(UiState.Result);
            RefreshFilteredHistory();
        }

        private void SaveCurrentResult()
        {
            if (SelectedCause == null) return;

            var item = new HistoryItem
            {
                Time = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"),
                Pattern = PatternName,
                Confidence = Confidence,
                SelectedCause = SelectedCause,
                FileName = CurrentFileName,
            };

            _historyService.Append(item);
            AllHistory.Insert(0, item);
            SavedCount++;
            RefreshFilteredHistory();
        }

        private void Reset()
        {
            _imageFiles.Clear();
            _currentIndex = 0;
            TotalCount = SkippedCount = DefectCount = SavedCount = NoneSkippedCount = 0;
            CurrentFileName = "";
            PatternName = Confidence = SeverityText = "";
            SeverityColor = "#555555";
            CauseOptions = new();
            FilteredHistory = new();
            SetUiState(UiState.Waiting);
        }

        private void RefreshFilteredHistory()
        {
            FilteredHistory = new ObservableCollection<HistoryItem>(
                AllHistory.Where(h => h.Pattern == PatternName));
            OnPropertyChanged(nameof(HistoryBadgeText));
            OnPropertyChanged(nameof(HistoryBadgeVisibility));
            OnPropertyChanged(nameof(NoHistoryVisibility));
        }

        // ══════════════════════════════════════════════
        // INotifyPropertyChanged
        // ══════════════════════════════════════════════

        public event PropertyChangedEventHandler? PropertyChanged;
        protected void OnPropertyChanged(string name)
            => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
    }
}