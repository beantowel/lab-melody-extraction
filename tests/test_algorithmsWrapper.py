import unittest
from utils.dataset import MedleyDB_Dataset, Adc2004_Dataset
from utils.algorithmsWrapper import *


class MelodyExtractionTest(unittest.TestCase):
    def setUp(self):
        # x = Adc2004_Dataset()
        x = MedleyDB_Dataset()
        idx = 0
        self.wav = x[idx]['wav']
        data, sr = self.wav
        self.dur = len(data) / sr
        print(f'test on:{x[idx]["title"]} {self.dur:.1f}s')

    def durationCheck(self, times):
        self.assertAlmostEqual(self.dur, times[-1], delta=0.5)

    def test_melodia(self):
        print('algo:melodia')
        times, _ = melodia(self.wav)
        self.durationCheck(times)

    def test_SLS(self):
        print('algo:SLS')
        times, _ = separateLeadStereo(self.wav)
        self.durationCheck(times)

    def test_SFC(self):
        print('algo:SFC')
        times, _ = sourceFilterContoursMelody(self.wav)
        self.durationCheck(times)

    def test_MCDNN(self):
        print('algo:MCDNN')
        times, _ = MCDNN(self.wav)
        self.durationCheck(times)

    def test_deepsalience(self):
        print('algo:deepsalience')
        times, _ = deepsalience(self.wav)
        self.durationCheck(times)

    def test_melodicSegnet(self):
        print('algo:melodicSegnet')
        times, _ = melodicSegnet(self.wav, None)
        self.durationCheck(times)

    def test_SFNMF_CRNN(self):
        print('algo:SFNMF_CRNN')
        times, _ = SFNMF_CRNN(self.wav)
        self.durationCheck(times)

    def test_JDC(self):
        print('algo:JDC')
        times, _ = JDC(self.wav)
        self.durationCheck(times)

    def test_SemanticSeg(self):
        print('algo:SemanticSeg')
        times, _ = SemanticSeg(self.wav)
        self.durationCheck(times)

    def test_algoProposed(self):
        print('algo:algoProposed')
        times, _ = algoProposed(self.wav)
        self.durationCheck(times)
