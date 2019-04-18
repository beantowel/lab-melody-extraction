import itertools
import numpy as np
from collections import Iterable


def insertRest(onsets, offsets, thresh=0.0001):
    onsetsWithR, offsetsWithR, isRest = [onsets[0]], [offsets[0]], [False]
    for onset, offset, pOffset in zip(onsets[1:], offsets[1:], offsets[:-1]):
        if onset - pOffset > thresh:
            onsetsWithR.append(pOffset)
            offsetsWithR.append(onset)
            isRest.append(True)
        onsetsWithR.append(onset)
        offsetsWithR.append(offset)
        isRest.append(False)
    return np.array(onsetsWithR), np.array(offsetsWithR), np.array(isRest)


def tuneOnsetOffset(onsets, offsets, tBase16, tStart=None):
    onsetsR, offsetsR, isRest = insertRest(onsets, offsets)
    noteValsR = offsetsR - onsetsR
    tunnedNoteValsR = tuneValBase16(noteValsR / tBase16) * tBase16

    tStart = onsets[0] if tStart is None else tStart
    acc = np.add.accumulate(tunnedNoteValsR)
    t = np.insert(acc + tStart, 0, tStart)
    tOnsets = (t[:-1])[isRest == False]
    tOffsets = (t[1:])[isRest == False]
    return tOnsets, tOffsets


NOTE_VALUE_BASE16 = [0., 1., 2., 4., 8., 16., 3., 6., 12., 24.]
NOTE_VALUE_3BASE16 = [2/3, 4/3, 8/3, 16/3]
NOTE_VALUE_ALL_BASE16 = np.array(tuple(itertools.chain(
    NOTE_VALUE_BASE16,
    NOTE_VALUE_3BASE16)))
BASIC_BEAT_BASE16 = {
    '2': 8,  # half note
    '4': 4,  # quater note
    '8': 2,  # 8th note
}


def tuneValBase16(valBase16, NVB=NOTE_VALUE_ALL_BASE16):
    if isinstance(valBase16, Iterable):
        tunned = np.array([tuneValBase16(v) for v in valBase16])
        ret = np.copy(tunned)
#         # allow 1/3 only in sequence mode
#         tunned3 = np.array([tuneValBase16(v, NVB=NOTE_VALUE_3BASE16)
#                             for v in valBase16])
#         for i in range(len(valBase16) - 2):
#             X = valBase16[i: i + 3]
#             Y, Y3 = tunned[i: i + 3], tunned3[i: i + 3]
#             if Y3[0] == Y3[1] == Y3[2]:
#                 e, e3 = sum(abs(X - Y)), sum(abs(X - Y3))
#                 if e3 < e:
#                     ret[i: i + 3] = tunned3[i: i + 3]
        return ret
    else:
        # match to closest base value
        idx = np.argmin(abs(NVB - valBase16))
        return NVB[idx]


def getBPM(noteVals, timeSignature='4/4'):
    def metric(valsBase16, tBase, coeffDt=0.1):
        tVals = tuneValBase16(valsBase16)
        errors = abs(tVals - valsBase16)
        detT = sum(tVals) - sum(valsBase16)
        met = tBase * np.average(errors) + coeffDt * \
            abs(detT) / sum(valsBase16)
        return met

    def getTBase16(vals, precision=0.01):
        vm = vals.min()
        rng = np.arange(vm / 24, vm * 24, vm * precision)
        met = [metric(vals / t, t) for t in rng]
        idx = np.argmin(met)
        return rng[idx]

    tBase16 = getTBase16(noteVals)
    bpMeasure, basicBeat = timeSignature.split('/')
    bpm = 60. / (tBase16 * BASIC_BEAT_BASE16[basicBeat])
    return bpm, tBase16
