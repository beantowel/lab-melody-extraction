import os
from collections import namedtuple
from itertools import chain
import numpy as np
import librosa
from mir_eval.io import load_delimited, load_time_series

from configs.configs import BaseDirs


MelDataPathPair = namedtuple('MelDataPathPair', 'title wav GT')


class BaseMelodyDataLoader():
    '''Melody Dataset Loader'''

    def __init__(self, baseDir):
        self.baseDir = baseDir
        self.pathPairs = []

    def __len__(self):
        return len(self.pathPairs)

    def __getitem__(self, idx):
        wav, gt = self.loadData(self.pathPairs[idx])
        title = self.pathPairs[idx].title
        sample = {'wav': wav, 'gt': gt, 'title': title}
        return sample

    def loadData(self, pathPair):
        wavPath = pathPair.wav
        GTPath = pathPair.GT
        wav = librosa.load(wavPath)
        gt = self.loadGT(GTPath)
        return wav, gt

    def loadGT(self, GTPath):
        raise NotImplementedError


class MedleyDB_Loader(BaseMelodyDataLoader):

    def __init__(self, baseDir=BaseDirs['MedleyDB']):
        super(MedleyDB_Loader, self).__init__(baseDir)
        # Audio/<Title>/<Title>_MIX.wav
        # MELODY2/<Title>_MELODY2.csv
        for GTname in os.listdir(os.path.join(baseDir, 'MELODY2')):
            assert GTname[-12:] == '_MELODY2.csv'
            title = GTname[:-12]
            wavPath = os.path.join(baseDir, 'Audio', title, title+'_MIX.wav')
            GTPath = os.path.join(baseDir, 'MELODY2', GTname)
            self.pathPairs.append(MelDataPathPair(title, wavPath, GTPath))

    def loadGT(self, GTPath):
        gt = load_time_series(GTPath, delimiter=',')
        return gt


class Adc2004_Loader(BaseMelodyDataLoader):

    def __init__(self, baseDir=BaseDirs['adc2004']):
        super(Adc2004_Loader, self).__init__(baseDir)
        # Audio/<title>.wav
        # GT/<title>REF.txt
        for GTname in os.listdir(os.path.join(baseDir, 'Audio')):
            title = GTname[:-4]
            wavPath = os.path.join(baseDir, 'Audio', title+'.wav')
            GTPath = os.path.join(baseDir, 'GT', title+'REF.txt')
            self.pathPairs.append(MelDataPathPair(title, wavPath, GTPath))

    def loadGT(self, GTPath):
        gt = load_time_series(GTPath)
        return gt


class RWC_Loader(BaseMelodyDataLoader):

    def __init__(self, baseDir=BaseDirs['RWC']):
        super(RWC_Loader, self).__init__(baseDir)
        # RWC-MDB-P-2001/AIST.RWC-MDB-P-2001.MELODY/RM-P<Num:03d>.MELODY.TXT
        # RWC-MDB-P-2001/RWC研究用音楽データベース[| Disc <Id>]/<Num':02d> <title>.wav

        discPaths = [os.path.join(
            baseDir, 'RWC-MDB-P-2001', 'RWC研究用音楽データベース')]
        for Id in range(2, 8):
            dp = os.path.join(baseDir, 'RWC-MDB-P-2001',
                              f'RWC研究用音楽データベース Disc {Id}')
            discPaths.append(dp)
        self.addPairFromPaths(discPaths, 'P')
        # RWC-MDB-R-2001/AIST.RWC-MDB-R-2001.MELODY/RM-R<Num:03d>.MELODY.TXT
        # RWC-MDB-R-2001/RWC研究用音楽データベース 著作権切れ音楽/<Num':02d>.*.wav
        discPaths = [os.path.join(
            baseDir, 'RWC-MDB-R-2001', 'RWC研究用音楽データベース 著作権切れ音楽')]
        self.addPairFromPaths(discPaths, 'R')

    def loadGT(self, GTPath):
        def fdiv100(x): return float(x) / 100
        cols = load_delimited(GTPath, [fdiv100, fdiv100, str, float, float])
        gt = np.array(cols[0]), np.array(cols[3])
        return gt

    def addPairFromPaths(self, discPaths, X):
        def listDisc(discPath):
            names = sorted(os.listdir(discPath))
            return [os.path.join(discPath, n) for n in names]

        for num, wavPath in enumerate(chain(*map(listDisc, discPaths))):
            _, tail = os.path.split(wavPath)
            assert tail[-4:] == '.wav', 'has non-wave file in the folder'
            title = tail[3:-4]
            GTPath = os.path.join(
                self.baseDir, f'RWC-MDB-{X}-2001', f'AIST.RWC-MDB-{X}-2001.MELODY', f'RM-{X}{num+1:03d}.MELODY.TXT')
            self.pathPairs.append(MelDataPathPair(title, wavPath, GTPath))


class Orchset_Loader(BaseMelodyDataLoader):

    def __init__(self, baseDir=BaseDirs['Orchset']):
        super(Orchset_Loader, self).__init__(baseDir)
        # GT/<title>.mel
        # audio/stereo/<title>.wav
        for GTname in os.listdir(os.path.join(baseDir, 'GT')):
            title = GTname[:-4]
            wavPath = os.path.join(baseDir, 'audio/stereo', title+'.wav')
            GTPath = os.path.join(baseDir, 'GT', title+'.mel')
            self.pathPairs.append(MelDataPathPair(title, wavPath, GTPath))

    def loadGT(self, GTPath):
        gt = load_time_series(GTPath)
        return gt