import unittest
from utils.dataset import MedleyDB_Dataset, Adc2004_Dataset, RWC_PR_Dataset, Orchset_Dataset
from utils.dataset import MedleyDB_vocal_Dataset, MedleyDB_instrumental_Dataset, Adc2004_vocal_Dataset, Adc2004_instrumental_Dataset
from utils.dataset import Segments_Dataset


class EnumerateAllinDataset(unittest.TestCase):
    def enumerateTest(self, x):
        print(f'{x.__class__.__name__}: {len(x)} samples')
        for i in range(len(x)):
            try:
                _ = x[i]
            except Exception:
                self.fail()

    def test_MedleyDB_Dataset(self):
        for x in (MedleyDB_Dataset(), MedleyDB_vocal_Dataset(), MedleyDB_instrumental_Dataset()):
            self.enumerateTest(x)

    def test_Adc2004_Dataset(self):
        for x in (Adc2004_Dataset(), Adc2004_vocal_Dataset(), Adc2004_instrumental_Dataset()):
            self.enumerateTest(x)

    def test_RWC_PR_Dataset(self):
        x = RWC_PR_Dataset()
        self.enumerateTest(x)

    def test_Orchset_Dataset(self):
        x = Orchset_Dataset()
        self.enumerateTest(x)

    def test_Segments_Dataset(self):
        baseDataset = MedleyDB_Dataset()
        x = Segments_Dataset(baseDataset, 44100, 2048, 512)
        self.enumerateTest(x)
        x = Segments_Dataset(baseDataset, 44100, 34304, 8576)
        self.enumerateTest(x)
