import logging
import torch
import librosa
import numpy as np
from numpy import errstate, isneginf
from collections import Iterable
from scipy.interpolate import interp1d


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


class SelectFreqs(object):
    '''select freq from gt in given interval for frame-wise label generation'''

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
        # f0, f1, f2, ..., fn
        # t0, t1, t2, ..., tn
        #      ts(sample0), ts(sample1), ..., ts(stamplen)
        samples = start + self.hop_length * np.arange(nFrames)
        samples += self.frame_length // 2 if not self.center else 0
        timestamps = librosa.samples_to_time(samples, self.sr)
        indices = np.searchsorted(times, timestamps)
        # out of range index: timestamp(idx)>tn --> freq=0 e.g. unvoice
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
        m_greater = (~m_unvoice) & (clz >= self.resolution)  # >= range mask
        m_lesser = (~m_unvoice) & (clz < 0)  # < range mask

        clz[m_unvoice] = -1  # unvoice mark as -1
        clz[m_greater] = self.resolution - 1  # rectify pitch
        clz[m_lesser] = 0  # rectify pitch
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

    def __init__(self, n_fft=2048, hop_length=512, win_length=2048, window='hann', center=False, phase=None):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.phase = phase

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
        if self.phase:
            A = np.log(np.abs(np.real(spectrogram)) + 0.0001)
            B = np.log(np.abs(np.imag(spectrogram)) + 0.0001)
            features = np.concatenate(
                [A.transpose(1, 0), B.transpose(1, 0)], axis=1)
        else:
            S = np.abs(spectrogram)
            features = S.transpose(1, 0)
        sample['features'] = features
        return sample

    def __repr__(self):
        return f'{self.__class__.__name__}(n_fft={self.n_fft}, hop_length={self.hop_length}, win_length={self.win_length}, window={self.window}, phase={self.phase})'


class EmphasisF(object):
    def __init__(self, n_fft=2048*2, hop_length=512, n_harmonics=1, factor=50, new_key=None):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_harmonics = n_harmonics
        self.factor = factor
        self.new_key = new_key

    def __call__(self, sample):
        y, sr = sample['wav']
        y, sr = librosa.resample(y, sr, 22050), 22050
        ref_times, ref_freqs = sample['gt']

        fft_f = librosa.fft_frequencies(sr, self.n_fft)
        f_interp = interp1d(
            librosa.time_to_frames(ref_times, sr, self.hop_length, self.n_fft),
            ref_freqs,
            fill_value=0.0,
            bounds_error=False)
        fft = librosa.stft(y, self.n_fft, self.hop_length)
        n_fft = np.zeros(fft.shape, dtype=fft.dtype)
        for frame in range(fft.shape[1]):
            freq = f_interp(frame)
            for i in range(self.n_harmonics):
                idx = np.argmin(np.abs(fft_f - freq*(i+1)))
                if np.abs(fft_f[idx] - freq * (i+1)) < fft_f[idx] * (2**(1/24) - 1):
                    n_fft[idx, frame] = fft[idx, frame]
                else:
                    fft[:, frame] = 0

        y = librosa.istft(n_fft, self.hop_length)
        y = y / max(y)
        if self.new_key is None:
            sample['wav'] = y, sr
        else:
            sample[self.new_key] = y, sr
        return sample


class LogSignal(object):
    def __init__(self, frame_length=2048, hop_length=512):
        self.frame_length = frame_length
        self.hop_length = hop_length

    def __call__(self, sample):
        if 'signal' in sample:
            signal = sample.pop('signal')
        elif 'wav' in sample:
            signal, sr = sample.pop('wav')
        else:
            raise KeyError

        sign = np.sign(signal)
        signal = np.abs(signal)
        signal = np.log(signal + 1) * sign
        signal = librosa.util.frame(
            signal,
            frame_length=self.frame_length,
            hop_length=self.hop_length)
        sample['features'] = signal.reshape(-1, self.frame_length)
        return sample


class PadSequence(object):
    def __call__(self, batch):
        sequences = [x['features'] for x in batch]
        paddedSeq = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True)
        seqLens = torch.LongTensor([len(x) for x in sequences])
        labels = torch.LongTensor(batch['labels'])
        return {'features': paddedSeq, 'lengths': seqLens, 'labels': labels}
