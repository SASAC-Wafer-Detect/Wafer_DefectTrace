
using System.Windows;
using Wafer_DefectTrace.ViewModels;

namespace Wafer_DefectTrace
{
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            DataContext = new MainViewModel();
        }
    }
}
