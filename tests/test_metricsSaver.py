import unittest
import numpy as np

from utils.metrics import Metrics_Saver


class SampleMetricsSave(unittest.TestCase):
    def test_Metrics_Saver(self):
        N, M = 10, 3
        saver = Metrics_Saver('test_dataset')
        algoNames = [f'Algo{i}' for i in range(M)]
        for algoName in algoNames:
            metrics = np.random.rand(N, 5)
            titles = [f'title{i}' for i in range(N)]
            saver.addResult(algoName, metrics, titles)
        saver.removeResult('Algo0')
        saver.dump('data/')
        saver = Metrics_Saver('test_dataset')
        saver.load('data/')
        saver.writeFullResults('data/')
        saver.writeAveResults('data/')
        saver.saveViolinPlot('data/')

