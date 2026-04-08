from datetime import datetime


class Tracker(object):
    def __init__(self) -> None:
        self.start: datetime = datetime.now()
        print('Tracker started at {}'.format(self.start))
        self.then: datetime = self.start
        self.now: datetime = self.start

    def tick(self, msg: str, end: str = '\n', verbose: bool = True) -> None:
        now = datetime.now()
        if verbose:
            print('[{}] [{}] {}'.format(now - self.start, now - self.then, msg), end=end)
        self.then = now
