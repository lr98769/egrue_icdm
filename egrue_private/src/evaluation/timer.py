from time import time

class Timer:
    def __init__(self, event: str):
        self.event = event
        self.start = time()

    def end(self):
        self.end = time()
        self.total = self.end-self.start
        print(f"{self.event} took {self.total:.3f}s")