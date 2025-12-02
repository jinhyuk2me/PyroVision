import threading


class DoubleBuffer:
    def __init__(self):
        self.buffers = [None, None]
        self.index = 0
        self.lock = threading.Lock()

    def write(self, frame):
        with self.lock:
            self.buffers[self.index] = frame
            self.index = 1 - self.index

    def read(self):
        with self.lock:
            return self.buffers[1 - self.index]
