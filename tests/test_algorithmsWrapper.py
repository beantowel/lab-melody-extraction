import unittest
from utils.dataset import MedleyDB_Dataset, Adc2004_Dataset
from utils.algorithmsWrapper import melodia, separateLeadStereo, sourceFilterContoursMelody, MCDNN, deepsalience, melodicSegnet, SFNMF_CRNN, algoFeng, JDC, algoCao


class MelodyExtractionTest(unittest.TestCase):
    def setUp(self):
        # x = Adc2004_Dataset()
        x = MedleyDB_Dataset()
        idx = 0
        self.wav = x[idx]['wav']
        data, sr = self.wav
        self.dur = len(data) / sr
        print(f'test on:{x[idx]["title"]} {self.dur:.1f}s')

    def test_melodia(self):
        print('algo:melodia')
        times, _ = melodia(self.wav)
        self.assertAlmostEqual(self.dur, times[-1], delta=0.5)

    def test_SLS(self):
        print('algo:SLS')
        times, _ = separateLeadStereo(self.wav)
        self.assertAlmostEqual(self.dur, times[-1], delta=0.5)

    def test_SFC(self):
        print('algo:SFC')
        times, _ = sourceFilterContoursMelody(self.wav)
        self.assertAlmostEqual(self.dur, times[-1], delta=0.5)

    def test_MCDNN(self):
        print('algo:MCDNN')
        times, _ = MCDNN(self.wav)
        self.assertAlmostEqual(self.dur, times[-1], delta=0.5)

    def test_deepsalience(self):
        print('algo:deepsalience')
        times, _ = deepsalience(self.wav)
        self.assertAlmostEqual(self.dur, times[-1], delta=0.5)

    def test_melodicSegnet(self):
        print('algo:melodicSegnet')
        times, _ = melodicSegnet(self.wav, None)
        self.assertAlmostEqual(self.dur, times[-1], delta=0.5)

    def test_SFNMF_CRNN(self):
        print('algo:SFNMF_CRNN')
        times, _ = SFNMF_CRNN(self.wav)
        self.assertAlmostEqual(self.dur, times[-1], delta=0.5)

    def test_JDC(self):
        print('algo:JDC')
        times, _ = JDC(self.wav)
        self.assertAlmostEqual(self.dur, times[-1], delta=0.5)

    def test_algoFeng(self):
        print('algo:algoFeng')
        times, _ = algoFeng(self.wav)
        self.assertAlmostEqual(self.dur, times[-1], delta=0.5)

    def test_algoCao(self):
        print('algo:algoCao')
        times, _ = algoCao(self.wav)
        self.assertAlmostEqual(self.dur, times[-1], delta=0.5)
