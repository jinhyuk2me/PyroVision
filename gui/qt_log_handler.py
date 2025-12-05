import logging
from PyQt6.QtCore import QObject, pyqtSignal


class LogSignaller(QObject):
    message = pyqtSignal(str)


class QtLogHandler(logging.Handler):
    """Qt 텍스트 뷰로 로그를 전달하기 위한 핸들러"""

    def __init__(self):
        super().__init__()
        self.signaller = LogSignaller()

    def emit(self, record):
        msg = self.format(record)
        self.signaller.message.emit(msg)
