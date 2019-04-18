import numpy as np
import matplotlib.pyplot as plt
from subprocess import call
from librosa import hz_to_midi, midi_to_hz


def melToOnsetOffsetMidi(times, freqs, melFile=None):
    '''extract onset, offset, midiNums from a melody(MIREX format) in piano-roll representation'''
    def melCompress(times, midiNums):
        '''compressed midiNums:=[n0,n0,n1,n1,...,nn,nn]'''
        # preprocess data for algorithm simplicity
        times = np.pad(times, (1, 1), 'constant', constant_values=(-1, -1))
        midiNums = np.pad(midiNums, (1, 1), 'constant', constant_values=(-1, -1))
        ct, cn = [], []
        for i, f in enumerate(midiNums[1:-1]):
            idx = i + 1
            if (midiNums[idx-1] == f) ^ (midiNums[idx+1] == f):
                ct.append(times[idx])
                cn.append(midiNums[idx])
        ct, cn = np.array(ct, dtype=float), np.array(cn, dtype=int)
        return ct, cn

    if melFile is not None:
        try:
            times, freqs = np.loadtxt(melFile, unpack='True')
        except:
            times, freqs = np.loadtxt(melFile, unpack='True', delimiter=',')
    times = times[freqs > 0]
    freqs = freqs[freqs > 0]
    midis = hz_to_midi(freqs).astype(int)
    ct, cn = melCompress(times, midis)
    assert len(ct) % 2 == 0, 'midiNums length should be even'
    onsets = ct[::2]
    offsets = ct[1::2]
    midiNums = cn[::2]
    return onsets, offsets, midiNums


def midiSpectrogram(times, freqs, values, title='midi spectrum'):
    midiNums = hz_to_midi(freqs).astype(int)
    plt.pcolormesh(times, midiNums, values)
    plt.xlabel('time')
    plt.ylabel('midi note number')
    plt.title(title)
    plt.show()
    return plt


def wavSpectrogram(rate, data, title='wave spectrum'):
    NFFT = 2048
    spectrum, freqs, t, im = plt.specgram(
        data, Fs=rate, Fc=440, NFFT=NFFT, mode='magnitude')
    plt.yscale('log')
    plt.xlabel('time')
    plt.ylabel('freq')
    plt.title(title)
    return plt


def intHist(midiNums, xlabel='midi note number', title='note histogram'):
    l, r = midiNums.min(), midiNums.max()
    plt.hist(midiNums, bins=np.arange(l, r)-0.5)
    plt.xlabel(xlabel)
    plt.ylabel('count')
    plt.title(title)
    return plt


def writeMel(pathFile, midiNums, onsets, offsets, till=0, dt=0.01):
    times = np.arange(0, max(offsets[-1], till), dt)
    freqs = []
    for t in times:
        selector = (onsets <= t) & (offsets > t)
        if selector.any():
            assert sum(selector) <= 1, 'multiple freqs at t'
            freqs.append(midi_to_hz(midiNums[selector]))
        else:
            freqs.append(0)

    np.savetxt(pathFile, np.array([times, freqs]).T,
               delimiter='\t', fmt=['%.2f', '%.3f'])
    return times, freqs


def melToWave(outMelName, outDir='./data/'):
    call(["python3", "./scripts/melToWave.py", outMelName, outDir])
