#!/usr/bin/env python
# encoding: utf-8
from .meter import Meter
class AverageValueMeter(Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()

    def update(self, val, num=1):
        self.val = val
        self.num += num
        self.sum += val * num
        self.avg = self.sum / self.num

    def value(self):
        return self.val, self.avg

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.num = 0
