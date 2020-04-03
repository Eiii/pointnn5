""" Class to record & store training stats """
from uuid import uuid4


class Measure:
    def __init__(self, name=None):
        self._training_loss = []
        self._valid_stats = []
        self._lr = []
        self.name = name
        self.uuid = uuid4().hex

    def training_loss(self, epoch, time, loss):
        data = (epoch, time, loss)
        self._training_loss.append(data)
        print(f'{epoch:.2f},{time:.1f},{loss:.5f}')

    def valid_stats(self, epoch, time, loss):
        data = (epoch, time, loss)
        self._valid_stats.append(data)
        fmt = f'VALID: {epoch},{time:.1f},{loss:.5f}'
        print(fmt)
