from abc import ABC, abstractmethod


class FrameSource(ABC):
    def __init__(self, name):
        self.name = name
        self.thread = None

    @abstractmethod
    def start(self):
        raise NotImplementedError

    def stop(self):
        """옵션: 소스 종료"""
        return False
