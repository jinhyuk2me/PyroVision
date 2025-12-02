from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen
from PyQt6.QtWidgets import QWidget


class RollingPlot(QWidget):
    def __init__(self, max_points=120, title="Metric", parent=None):
        super().__init__(parent)
        self.max_points = max_points
        self.title = title
        self.values = []
        self.setMinimumHeight(80)
        self.setMaximumHeight(120)

    def update_value(self, value):
        self.values.append(value)
        if len(self.values) > self.max_points:
            self.values.pop(0)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 20, 20))
        painter.setPen(QColor(200, 200, 200))
        painter.drawText(10, 15, f"{self.title}: {self.values[-1]:.1f}" if self.values else self.title)

        if len(self.values) < 2:
            return

        w = self.width()
        h = self.height() - 20
        max_val = max(self.values) or 1
        min_val = min(self.values)
        span = max_val - min_val or 1

        scale_x = w / (self.max_points - 1)
        pen = QPen(QColor(0, 255, 0))
        pen.setWidth(2)
        painter.setPen(pen)
        for i in range(1, len(self.values)):
            x1 = (i - 1) * scale_x
            x2 = i * scale_x
            y1 = h - ((self.values[i - 1] - min_val) / span) * h + 20
            y2 = h - ((self.values[i] - min_val) / span) * h + 20
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))
