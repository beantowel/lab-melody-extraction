import unittest
from utils.dataLoader import MedleyDB_Loader, Adc2004_Loader, RWC_Loader, Orchset_Loader


class EnumerateAllinDataset(unittest.TestCase):
    def enumerateTest(self, x):
        print(f'{len(x)} files')
        for i in range(len(x)):
            try:
                _ = x[i]
            except Exception:
                self.fail()

    def test_MedleyDB_Loader(self):
        x = MedleyDB_Loader()
        print('MedleyDB:')
        self.enumerateTest(x)

    def test_adc2004_Loader(self):
        x = Adc2004_Loader()
        print('adc2004:')
        self.enumerateTest(x)

    def test_RWC_Loader(self):
        x = RWC_Loader()
        print('RWC:')
        self.enumerateTest(x)

    def test_Orchset_Loader(self):
        x = Orchset_Loader()
        print('Orchset:')
        self.enumerateTest(x)