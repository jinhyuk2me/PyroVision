from typing import Optional, Dict, Any

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit, QHBoxLayout
from PyQt6.QtCore import Qt


try:
    from gui.plot_widget import RollingPlot
except ImportError:
    RollingPlot = None


class MonitorPanel(QWidget):
    """상태/로그 모니터링 패널"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.status_label = QLabel("Status: -")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.status_label.setWordWrap(True)
        self.fusion_info = QLabel("-")
        self.fusion_info.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.fusion_info.setWordWrap(True)

        self.det_plot = RollingPlot(title="Det FPS") if RollingPlot else None
        self.rgb_plot = RollingPlot(title="RGB FPS") if RollingPlot else None
        self.ir_plot = RollingPlot(title="IR FPS") if RollingPlot else None

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(140)

        layout.addWidget(self.status_label)
        layout.addWidget(self.fusion_info)
        if self.det_plot or self.rgb_plot or self.ir_plot:
            plot_row = QHBoxLayout()
            if self.det_plot:
                plot_row.addWidget(self.det_plot)
            if self.rgb_plot:
                plot_row.addWidget(self.rgb_plot)
            if self.ir_plot:
                plot_row.addWidget(self.ir_plot)
            layout.addLayout(plot_row)
        layout.addWidget(self.log_view)

    def append_log(self, text: str) -> None:
        self.log_view.append(text)

    def get_plots(self) -> Dict[str, Optional[Any]]:
        return {
            'det_plot': self.det_plot,
            'rgb_plot': self.rgb_plot,
            'ir_plot': self.ir_plot,
        }
