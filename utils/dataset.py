import os
import time
import librosa
import torch
import numpy as np
from copy import deepcopy
from itertools import chain
from multiprocessing import Pool
from collections import namedtuple, deque
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from mir_eval.io import load_delimited, load_time_series

from configs.configs import DATASET_BASE_DIRS, MEDLEYDB_INIT

MEDLEYDB_INIT()
import medleydb as mdb

MelDataPathPair = namedtuple('MelDataPathPair', 'title wav GT')


class BaseMelodyDataset(Dataset):
    '''Melody Dataset Base Loader,
    pathPairs are recommended to be sorted for consistency '''

    def __init__(self, baseDir, sr, transform):
        self.baseDir = baseDir
        self.sr = sr
        self.transform = transform
        self.pathPairs = []

    def __len__(self):
        return len(self.pathPairs)

    def __getitem__(self, idx):
        return self.getSample(self.pathPairs[idx])

    def getSample(self, pathPair):
        '''load wav, melody, title from pathPair'''
        wavPath = pathPair.wav
        GTPath = pathPair.GT
        title = pathPair.title

        wav = librosa.load(wavPath, sr=self.sr)
        gt = self.loadGT(GTPath)
        sample = {'wav': wav, 'gt': gt, 'title': title}

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def loadGT(self, GTPath):
        '''load melody ground truth'''
        raise NotImplementedError

    def __add__(self, other):
        '''do not provide closure for BaseMelodyDataset class
           A + B will no longer ∈ BaseMelodyDataset
           because it has a virtual loadGT() function'''
        x = ConcatDataset([self, other])
        x.pathPairs = self.pathPairs + other.pathPairs
        return x

    def randomSplit(self, splitRatio, seed=42):
        np.random.seed(seed)
        indices = np.random.permutation(range(len(self)))
        newLen = int(len(self) * splitRatio)
        a_pathPairs = [self.pathPairs[i] for i in indices[:newLen]]
        b_pathPairs = [self.pathPairs[i] for i in indices[newLen:]]

        a, b = self, deepcopy(self)
        a.pathPairs = a_pathPairs
        b.pathPairs = b_pathPairs
        return a, b


class MedleyDB_Dataset(BaseMelodyDataset):
    '''<MedleyDB: A Multitrack Dataset for Annotation-Intensive MIR Research>'''

    def __init__(self, baseDir=DATASET_BASE_DIRS['MedleyDB'], sr=44100, transform=None):
        super(MedleyDB_Dataset, self).__init__(baseDir, sr, transform)
        # Audio/<title>/<title>_MIX.wav
        # MELODY2/<title>_MELODY2.csv
        for GTname in sorted(os.listdir(os.path.join(baseDir, 'MELODY2'))):
            assert GTname[-12:] == '_MELODY2.csv'
            title = GTname[:-12]
            wavPath = os.path.join(baseDir, 'Audio', title, title+'_MIX.wav')
            GTPath = os.path.join(baseDir, 'MELODY2', GTname)
            self.pathPairs.append(MelDataPathPair(title, wavPath, GTPath))

    def loadGT(self, GTPath):
        gt = load_time_series(GTPath, delimiter=',')
        return gt


class MedleyDB_vocal_Dataset(MedleyDB_Dataset):

    def __init__(self, baseDir=DATASET_BASE_DIRS['MedleyDB'], sr=44100, transform=None):
        super(MedleyDB_vocal_Dataset, self).__init__(baseDir, sr, transform)
        pathPairs = []
        for pathPair in self.pathPairs:
            title = pathPair.title
            if not mdb.MultiTrack(title).is_instrumental:
                pathPairs.append(pathPair)
        self.pathPairs = pathPairs


class MedleyDB_instrumental_Dataset(MedleyDB_Dataset):

    def __init__(self, baseDir=DATASET_BASE_DIRS['MedleyDB'], sr=44100, transform=None):
        super(MedleyDB_instrumental_Dataset, self).__init__(
            baseDir, sr, transform)
        pathPairs = []
        for pathPair in self.pathPairs:
            title = pathPair.title
            if mdb.MultiTrack(title).is_instrumental:
                pathPairs.append(pathPair)
        self.pathPairs = pathPairs


class Adc2004_Dataset(BaseMelodyDataset):
    '''ISMIR 2004'''

    def __init__(self, baseDir=DATASET_BASE_DIRS['adc2004'], sr=44100, transform=None):
        super(Adc2004_Dataset, self).__init__(baseDir, sr, transform)
        # Audio/<title>.wav
        # GT/<title>REF.txt
        for GTname in sorted(os.listdir(os.path.join(baseDir, 'Audio'))):
            title = GTname[:-4]
            wavPath = os.path.join(baseDir, 'Audio', title+'.wav')
            GTPath = os.path.join(baseDir, 'GT', title+'REF.txt')
            self.pathPairs.append(MelDataPathPair(title, wavPath, GTPath))
        self.instrumentalList = ('midi1','midi2','midi3','midi4','jazz1','jazz2','jazz3','jazz4',)

    def loadGT(self, GTPath):
        gt = load_time_series(GTPath)
        return gt


class Adc2004_vocal_Dataset(Adc2004_Dataset):

    def __init__(self, baseDir=DATASET_BASE_DIRS['adc2004'], sr=44100, transform=None):
        super(Adc2004_vocal_Dataset, self).__init__(baseDir, sr, transform)
        
        pathPairs = []
        for pathPair in self.pathPairs:
            title = pathPair.title
            if title not in self.instrumentalList:
                pathPairs.append(pathPair)
        self.pathPairs = pathPairs


class Adc2004_instrumental_Dataset(Adc2004_Dataset):

    def __init__(self, baseDir=DATASET_BASE_DIRS['adc2004'], sr=44100, transform=None):
        super(Adc2004_instrumental_Dataset, self).__init__(baseDir, sr, transform)
        pathPairs = []
        for pathPair in self.pathPairs:
            title = pathPair.title
            if title in self.instrumentalList:
                pathPairs.append(pathPair)
        self.pathPairs = pathPairs


class Orchset_Dataset(BaseMelodyDataset):
    '''<Evaluation and Combination of Pitch Estimation Methods for Melody Extraction in Symphonic Classical Music>'''

    def __init__(self, baseDir=DATASET_BASE_DIRS['Orchset'], sr=44100, transform=None):
        super(Orchset_Dataset, self).__init__(baseDir, sr, transform)
        # GT/<title>.mel
        # audio/stereo/<title>.wav
        for GTname in sorted(os.listdir(os.path.join(baseDir, 'GT'))):
            title = GTname[:-4]
            wavPath = os.path.join(baseDir, 'audio/stereo', title+'.wav')
            GTPath = os.path.join(baseDir, 'GT', title+'.mel')
            self.pathPairs.append(MelDataPathPair(title, wavPath, GTPath))

    def loadGT(self, GTPath):
        gt = load_time_series(GTPath)
        return gt


class RWC_Base_Dataset(BaseMelodyDataset):
    '''<RWC Music Database: Popular, Classical, and Jazz Music Databases>'''

    def __init__(self, baseDir=DATASET_BASE_DIRS['RWC'], sr=44100, transform=None):
        super(RWC_Base_Dataset, self).__init__(baseDir, sr, transform)

    def loadGT(self, GTPath):
        cols = load_delimited(GTPath, [int, int, str, float, float])
        gt = np.array(cols[0]) / 100., np.array(cols[3])
        return gt

    def addPairFromPaths(self, discPaths, X):
        def listDisc(discPath):
            # ensure the wav files are well-ordered, or GTfile will mismatch
            names = sorted(os.listdir(discPath))
            return [os.path.join(discPath, n) for n in names]

        for num, wavPath in enumerate(chain(*map(listDisc, discPaths))):
            _, tail = os.path.split(wavPath)
            assert tail[-4:] == '.wav', 'has non-wave file in the RWC disc folder'
            title = tail[3:-4]
            GTPath = os.path.join(
                self.baseDir, f'RWC-MDB-{X}-2001', f'AIST.RWC-MDB-{X}-2001.MELODY', f'RM-{X}{num+1:03d}.MELODY.TXT')
            self.pathPairs.append(MelDataPathPair(title, wavPath, GTPath))


class RWC_Popular_Dataset(RWC_Base_Dataset):

    def __init__(self, baseDir=DATASET_BASE_DIRS['RWC'], sr=44100, transform=None):
        super(RWC_Popular_Dataset, self).__init__(baseDir, sr, transform)
        # RWC-MDB-P-2001/AIST.RWC-MDB-P-2001.MELODY/RM-P<Num:03d>.MELODY.TXT
        # RWC-MDB-P-2001/RWC研究用音楽データベース[| Disc <Id>]/<Num':02d> <title>.wav
        discPaths = [os.path.join(
            baseDir, 'RWC-MDB-P-2001', 'RWC研究用音楽データベース')]
        for Id in range(2, 8):
            dp = os.path.join(baseDir, 'RWC-MDB-P-2001',
                              f'RWC研究用音楽データベース Disc {Id}')
            discPaths.append(dp)
        self.addPairFromPaths(discPaths, 'P')


class RWC_Royalty_Free_Dataset(RWC_Base_Dataset):

    def __init__(self, baseDir=DATASET_BASE_DIRS['RWC'], sr=44100, transform=None):
        super(RWC_Royalty_Free_Dataset, self).__init__(baseDir, sr, transform)
        # RWC-MDB-R-2001/AIST.RWC-MDB-R-2001.MELODY/RM-R<Num:03d>.MELODY.TXT
        # RWC-MDB-R-2001/RWC研究用音楽データベース 著作権切れ音楽/<Num':02d> <title>.wav
        discPaths = [os.path.join(
            baseDir, 'RWC-MDB-R-2001', 'RWC研究用音楽データベース 著作権切れ音楽')]
        self.addPairFromPaths(discPaths, 'R')


class MIREX_05_Dataset(BaseMelodyDataset):
    '''<Music Information Retrieval Evaluation eXchange 2005>'''

    def __init__(self, baseDir=DATASET_BASE_DIRS['MIREX-05'], sr=44100, transform=None):
        super(MIREX_05_Dataset, self).__init__(baseDir, sr, transform)
        # Audio/<title>.wav
        # GT/<title>[:7]REF.txt ([:7] for train13MIDI exception)
        for GTname in sorted(os.listdir(os.path.join(baseDir, 'Audio'))):
            title = GTname[:-4]
            wavPath = os.path.join(baseDir, 'Audio', title+'.wav')
            GTPath = os.path.join(baseDir, 'GT', title[:7]+'REF.txt')
            self.pathPairs.append(MelDataPathPair(title, wavPath, GTPath))
        self.instrumentalList = ('train10','train11','train12','train13MIDI',)

    def loadGT(self, GTPath):
        gt = load_time_series(GTPath)
        return gt


class MIREX_05_vocal_Dataset(MIREX_05_Dataset):

    def __init__(self, baseDir=DATASET_BASE_DIRS['MIREX-05'], sr=44100, transform=None):
        super(MIREX_05_vocal_Dataset, self).__init__(baseDir, sr, transform)
        
        pathPairs = []
        for pathPair in self.pathPairs:
            title = pathPair.title
            if title not in self.instrumentalList:
                pathPairs.append(pathPair)
        self.pathPairs = pathPairs


class MIREX_05_instrumental_Dataset(MIREX_05_Dataset):

    def __init__(self, baseDir=DATASET_BASE_DIRS['MIREX-05'], sr=44100, transform=None):
        super(MIREX_05_instrumental_Dataset, self).__init__(baseDir, sr, transform)
        pathPairs = []
        for pathPair in self.pathPairs:
            title = pathPair.title
            if title in self.instrumentalList:
                pathPairs.append(pathPair)
        self.pathPairs = pathPairs


class IKala_Dataset(BaseMelodyDataset):
    '''<Vocal activity informed singing voice separation with the iKala dataset>'''

    def __init__(self, baseDir=DATASET_BASE_DIRS['iKala'], sr=44100, transform=None):
        super(IKala_Dataset, self).__init__(baseDir, sr, transform)
        # Wavfile/<title>.wav
        # PitchLabel/<title>.pv)
        for GTname in sorted(os.listdir(os.path.join(baseDir, 'Wavfile'))):
            title = GTname[:-4]
            wavPath = os.path.join(baseDir, 'Wavfile', title+'.wav')
            GTPath = os.path.join(baseDir, 'PitchLabel', title+'.pv')
            self.pathPairs.append(MelDataPathPair(title, wavPath, GTPath))

    def loadGT(self, GTPath):
        freqs = load_delimited(GTPath, [float])
        gt = np.arange(0.5, len(freqs)) / 31.25, np.array(freqs)
        return gt

class Segments_Dataset(Dataset):
    '''make a equal-length segmented wav dataset from an existed dataset,
    sequential queries will be much faster with sample cache'''

    def __init__(self, dataset, sr=22050, frameSize=2048, hopSize=512, transform=None, cacheSize=4):
        self.dataset = dataset
        self.sr = sr
        self.frameSize = frameSize
        self.hopSize = hopSize
        self.transform = transform
        self.cacheSize = cacheSize
        self.frameDt = float(frameSize) / sr

        # count frames in dataset
        nFramesList = []
        for pathPair in dataset.pathPairs:
            wavPath = pathPair.wav
            duration = librosa.get_duration(filename=wavPath)
            nSamples = librosa.time_to_samples(duration, sr=self.sr)
            nFrames = 1 + int((nSamples - self.frameSize) /
                              float(self.hopSize))
            nFramesList.append(nFrames)
            # check validation
            sStart = librosa.frames_to_samples(
                nFrames-1, hop_length=self.hopSize)
            sEnd = sStart + self.frameSize
            assert (nSamples > 0) and (sEnd <= nSamples), f'{nFrames}:{sStart}_{sEnd}, {nSamples}'
        self.frameCumsum = np.cumsum(nFramesList)

        # FIFO cache
        self._sampleCache = deque(maxlen=cacheSize)
        self._sampleIdxCache = deque(maxlen=cacheSize)

    def __len__(self):
        return self.frameCumsum[-1]

    def __getitem__(self, idx):
        '''wavIdx: wav0   wav1    wav2 ...
           idx:   [0-f1) [f1-f2) [f2-f3) ...
           cumsum:  f1     f2      f3 ...'''
        sampleIdx = np.searchsorted(self.frameCumsum, idx, side='right')
        frame = idx - (self.frameCumsum[sampleIdx - 1] if sampleIdx > 0 else 0)
        sample = self._getSampleFromCache(sampleIdx)

        sStart = librosa.frames_to_samples(frame, hop_length=self.hopSize)
        tStart = librosa.frames_to_time(
            frame, sr=self.sr, hop_length=self.hopSize)
        segSample = self.getSegSample(sample, sStart, tStart)
        return segSample

    def getSegSample(self, sample, sStart, tStart):
        y, _ = sample['wav']
        ref_times, ref_freqs = sample['gt']
        sEnd = sStart + self.frameSize
        tEnd = tStart + self.frameDt

        signal = y[sStart: sEnd]
        assert len(signal) == sEnd - \
            sStart, f'len(y[{sStart}:{sEnd}]) != len(signal):{len(signal)}'

        seg_mask = (ref_times >= tStart) & (ref_times < tEnd)
        seg_freqs = ref_freqs[seg_mask]
        seg_times = ref_times[seg_mask]
        segSample = {
            'signal': signal,
            'gt': (seg_times, seg_freqs),
            'interval': (sStart, sEnd),
        }

        if self.transform is not None:
            segSample = self.transform(segSample)
        return segSample

    def _loadSample(self, sampleIdx):
        sample = self.dataset[sampleIdx]
        y, sr = sample['wav']
        if sr != self.sr:
            y = librosa.resample(y, sr, self.sr)
            sample['wav'] = y, self.sr
        return sample

    def _getSampleFromCache(self, sampleIdx):
        try:
            # if sampleIdx was cacheed
            pos = self._sampleIdxCache.index(sampleIdx)
        except ValueError:
            # load to cache and resample
            sample = self._loadSample(sampleIdx)
            # update FIFO deque
            self._sampleIdxCache.append(sampleIdx)
            self._sampleCache.append(sample)
            pos = self._sampleIdxCache.index(sampleIdx)
        return self._sampleCache[pos]

    def warmUp(self, num_workers=12):
        size = min(self.cacheSize, len(self.frameCumsum))
        print(f'{self.__class__.__name__}({len(self.dataset)}) warming up cache({size})')
        start = time.time()

        with Pool(processes=num_workers) as pool:
            indices = list(range(size))
            samples = pool.map(self._loadSample, indices)
            self._sampleCache.extend(samples)
            self._sampleIdxCache.extend(indices)

        elapse = (time.time() - start) / 60
        print(f'warm up elapse:{elapse:4.3f}min')


