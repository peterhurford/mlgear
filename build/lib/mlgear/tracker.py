from datetime import datetime


class Tracker(object):
    def __init__(self):
        self.start = datetime.now()
        print('Tracker started at {}'.format(self.start))
        self.then = self.start
        self.now = self.start

    def tick(self, msg, end='\n', verbose=True):
        now = datetime.now()
        if verbose:
            print('[{}] [{}] {}'.format(now - self.start, now - self.then, msg), end=end)
        self.then = now
