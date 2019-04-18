import torch
import librosa
import numpy as np
from numpy import errstate, isneginf
from collections import Iterable

from utils.melProcess import melToOnsetOffsetMidi


class Compose(object):
    '''Composes several transforms together.'''

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ExtractNotes(object):
    '''extract onset, offset, notes from melody(MIREX format) in piano-roll representation'''

    def __call__(self, sample):
        times, freqs = sample['gt']
        onsets, offsets, midiNums = melToOnsetOffsetMidi(times, freqs)
        sample['notes'] = (onsets, offsets, midiNums)
        return sample

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class SelectFreqs(object):
    '''select freq from gt for label generation'''

    def __init__(self, sr, hop_length, frame_length, center=False):
        self.sr = sr
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.center = center

    def __call__(self, sample):
        start, end = sample.pop('interval')
        times, freqs = sample.pop('gt')
        nFrames = 1 + int((end - start - self.frame_length) /
                          float(self.hop_length))
        # select sample in the frame
        samples = start + self.hop_length * np.arange(nFrames)
        samples += self.frame_length // 2 if not self.center else 0
        timestamps = librosa.samples_to_time(samples, self.sr)
        indices = np.searchsorted(times, timestamps)
        # out of range index: freq=0
        s_freqs = np.insert(freqs, len(freqs), 0.)[indices]
        sample['freqs'] = s_freqs
        return sample

    def __repr__(self):
        return f'{self.__class__.__name__}(sr={self.sr}, hop_length={self.hop_length}, frame_length={self.frame_length})'


class FreqToPitchClass(object):
    '''convert frequency in hz to logscale classes'''

    def __init__(self, low='A1', high='A6', resolution=120):
        self.low = low
        self.high = high
        self.resolution = resolution

        self.fLow = librosa.note_to_hz(low)
        self.fHigh = librosa.note_to_hz(high)
        # [fLow, fHig) <--> [0, resolution)
        # clz(f) = a * log_2(f / fLow)
        # clz(fHigh) - clz(fLow) = a * log_2(fHigh / fLow) - 0 = resolution
        self.a = resolution / np.log2(self.fHigh / self.fLow)

    def __call__(self, sample):
        freqs = sample.pop('freqs')
        assert isinstance(freqs, np.ndarray)
        with errstate(divide='ignore'):
            clz = self.a * np.log2(freqs / self.fLow)

        m_unvoice = isneginf(clz)
        m_out = (~m_unvoice) & ((clz < 0) | (
            clz >= self.resolution))  # out of range mask

        clz[m_unvoice | m_out] = -1  # out of range mark as unvoice
        clz = clz.astype(int)

        sample['labels'] = clz
        return sample

    def inv(self, sample):
        labels = sample.pop('labels')
        assert isinstance(labels, Iterable)
        labels = np.array(labels)
        m_unvoice = (labels == -1)
        freqs = self.fLow * np.power(2, labels / self.a)
        freqs[m_unvoice] = 0.

        sample['freqs'] = freqs
        return sample

    def __repr__(self):
        return f'{self.__class__.__name__}(low={self.low}, high={self.high}, resolution={self.resolution})'


class STFT(object):
    '''Short-time fourier transform for one frame'''

    def __init__(self, n_fft=2048, hop_length=512, win_length=2048, window='hann', center=False):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center

    def __call__(self, sample):
        if 'signal' in sample:
            signal = sample.pop('signal')
        elif 'wav' in sample:
            signal, sr = sample.pop('wav')
        else:
            raise KeyError

        spectrogram = librosa.stft(
            signal, self.n_fft, self.hop_length, self.win_length, self.window, self.center)
        # shape (N_FFT, N_HOP) -> (N_HOP, N_FFT)
        features = np.abs(spectrogram).transpose(1, 0)
        sample['features'] = features
        return sample

    def __repr__(self):
        return f'{self.__class__.__name__}(n_fft={self.n_fft}, hop_length={self.hop_length}, win_length={self.win_length}, window={self.window})'


class PadSequence(object):
    def __call__(self, batch):
        sequences = [x['features'] for x in batch]
        paddedSeq = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True)
        seqLens = torch.LongTensor([len(x) for x in sequences])
        labels = torch.LongTensor(batch['labels'])
        return {'features': paddedSeq, 'lengths': seqLens, 'labels': labels}
