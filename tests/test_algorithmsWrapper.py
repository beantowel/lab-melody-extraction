import unittest
from utils.dataLoader import MedleyDB_Loader, Adc2004_Loader
from utils.algorithmsWrapper import melodia, seprateLeadStereo, sourceFilterContoursMelody, MCDNN, deepsalience


class MelodyExtractionTest(unittest.TestCase):
    def setUp(self):
        x = Adc2004_Loader()
        # x = MedleyDB_Loader()
        idx = 0
        self.wav = x[idx]['wav']
        data, sr = self.wav
        self.dur = len(data) / sr
        print(f'test on:{x[idx]["title"]} {self.dur:.1f}s')

    def test_melodia(self):
        times, pitchs = melodia(self.wav)
        self.assertAlmostEqual(self.dur, times[-1], delta=0.5)

    def test_SLS(self):
        times, pitchs = seprateLeadStereo(self.wav)
        self.assertAlmostEqual(self.dur, times[-1], delta=0.5)

    def test_SFC(self):
        times, pitchs = sourceFilterContoursMelody(self.wav)
        self.assertAlmostEqual(self.dur, times[-1], delta=0.5)

    def test_MCDNN(self):
        times, pitchs = MCDNN(self.wav)
        self.assertAlmostEqual(self.dur, times[-1], delta=0.5)

    def test_deepsalience(self):
        times, pitchs = deepsalience(self.wav)
        self.assertAlmostEqual(self.dur, times[-1], delta=0.5)
