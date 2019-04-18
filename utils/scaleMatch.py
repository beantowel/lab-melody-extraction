import itertools
import numpy as np
from collections import Iterable, deque
from heapq import nlargest
from operator import itemgetter
from librosa import hz_to_midi, midi_to_note, note_to_midi


NOTE_NAMES = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
NOTE_NAMES_DIC = {name: idx for idx, name in enumerate(NOTE_NAMES)}
SCALE_INTERVALS = {
    'major':  (2, 2, 1, 2, 2, 2, 1),
    'harmonic':  (2, 1, 2, 2, 1, 3, 1),
    'melodic':  (2, 1, 2, 2, 2, 2, 1),
}


def selectNotes(onsets, times, freqs, salience, k=4, dt=0.04):
    def metric(candidates):
        grad = np.max(np.diff(candidates, axis=1), axis=1)
        mean = np.mean(candidates, axis=1)
        return grad

    midis = hz_to_midi(freqs).astype(int)
    notes, metrics = [], []
    for t in onsets:
        candidates = salience[:, (times < t + dt) & (times > t - dt)]
        met = metric(candidates)
        nlar = nlargest(k, enumerate(met), itemgetter(1))
        notes.append([midis[enum[0]] for enum in nlar])
        metrics.append([enum[1] for enum in nlar])
    return np.array(notes), np.array(metrics)


class SCALE():
    def __init__(self, midi, idx, atScaleType):
        self._scaleType = atScaleType
        self._itvs = SCALE_INTERVALS[atScaleType]
        # np.sum([]) = 0.0 not 0
        self._tonic = midi - int(np.sum(self._itvs[0: idx]))
        stripChars = '-0123456789'  # remove digits from note name
        name = midi_to_note(self._tonic).rstrip(stripChars)
        self._name = f'{name} {atScaleType}'

        self._midis = [self._tonic]  # range[0, 128)
        # add midi in (tonic, 128)
        m = self._tonic
        for itv in itertools.cycle(self._itvs):
            m += itv
            if m < 128:
                self._midis.append(m)
            else:
                break
        # add midi in [0, tonic)
        m = self._tonic
        for itv in itertools.cycle(self._itvs[::-1]):
            m -= itv
            if m >= 0:
                self._midis.insert(0, m)
            else:
                break
        self._midis = np.array(self._midis).astype(int)

    def __repr__(self):
        return self._name

    def midis(self):
        return self._midis


def getScale(notes):
    def scaleMatch(intervals, sitvs):
        '''match observed interval to given scale-interval,
        return index shift so that intervals[i]==sitvs[idx + i]'''
        indices = []
        Y = deque(sitvs)
        for idx, _ in enumerate(sitvs):
            X = deque(intervals)
            for y in itertools.cycle(Y):
                if X:  # X is not empty
                    X[0] -= y
                    if X[0] < 0:
                        break
                    elif X[0] == 0:
                        X.popleft()
                else:
                    indices.append(idx)
                    break
            Y.rotate(-1)
        return indices

    intervals = np.diff(notes)
    tonic = notes[0]  # assume the lowest note is the tonic
    scales = []
    for scaleType, sitvs in SCALE_INTERVALS.items():
        indices = scaleMatch(intervals, sitvs)
        for idx in indices:
            scales.append(SCALE(tonic, idx, scaleType))
    return scales
