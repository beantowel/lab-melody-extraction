import unittest
import librosa
import numpy as np
from copy import copy

from utils.dataset import MedleyDB_Dataset
from utils.preprocessor import FreqToPitchClass, STFT, SelectFreqs


class TransformTest(unittest.TestCase):
    def test_FreqToPitchClass(self, low='A1', high='A6', res=120):
        def check(freqs, labels, ifreqs):
            self.assertIsInstance(labels, np.ndarray)
            self.assertEqual(labels.dtype, np.int)
            for freq, label, ifreq in zip(freqs, labels, ifreqs):
                if freq >= fLow and freq < fHigh:
                    self.assertGreaterEqual(label, 0)
                    self.assertLess(label, res)
                    self.assertGreater(freq, ifreq)
                    self.assertLess(freq, ifreq * rDelta)
                elif freq >= fHigh:
                    self.assertEqual(label, res-1)
                elif freq < fLow and freq > 0:
                    self.assertEqual(label, 0)
                else:
                    self.assertEqual(label, -1)

        fLow = librosa.note_to_hz(low)
        fHigh = librosa.note_to_hz(high)
        rDelta = np.power(fHigh / fLow, 1. / res)
        self.assertGreater(rDelta, 1.)
        transform = FreqToPitchClass(low=low, high=high, resolution=res)
        labels = np.arange(-10, 128 + 10)
        rfreqs = np.append(librosa.midi_to_hz(labels), 0)
        rfreqs = np.repeat(rfreqs, 30)
        freqs = rfreqs * np.random.uniform(
            low=1./rDelta, high=rDelta, size=rfreqs.shape)  # add some deviation

        sample = {'freqs': freqs}
        t_sample = transform(sample)
        it_sample = transform.inv(copy(t_sample))
        check(freqs, t_sample['labels'], it_sample['freqs'])

    def test_STFT(self):
        transform = STFT(2048)
        dataset = MedleyDB_Dataset(transform=transform)
        data = dataset[0]
        features = data['features']
        self.assertEqual(features.shape[1], 1025)
