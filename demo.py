import click
import librosa
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from librosa import midi_to_note

# from utils.melProcess import melToOnsetOffsetMidi, hz2Midi
# from modelCao.melExtractor import MelExtractor
# from modelCao.pitchEstimator import KlapuriEstimator, MelodiaEstimator
# from modelCao.pitchTracker import GrammarTracker

from utils.dataset import Adc2004_Dataset, MedleyDB_vocal_Dataset, Segments_Dataset
from utils.preprocessor import Compose, FreqToPitchClass, STFT, SelectFreqs
from modelFeng.model import MultiDNN
from configs.modelFengConfigs import *
from trainFeng import *

# load model
model, train_loader, valid_loader, epoch_r = resumeFrom(SAVE_DATA, 'cpu')
print(f'resume state from {SAVE_DATA} to {DEVICE}, epoch:{epoch_r}')
lossFunction = MultiTaskLoss(AUXILIARY_WEIGHT)

# custom valid_loader
dataset, _ = Adc2004_Dataset().randomSplit(1.0)
transform = Compose([
    SelectFreqs(SR, HOP_SIZE, FRAME_SIZE),
    FreqToPitchClass(NOTE_LOW, NOTE_HIGH, BIN_RESOLUTION),
    STFT(N_FFT, HOP_SIZE, FRAME_SIZE),
])
segset = Segments_Dataset(dataset, transform=transform)
valid_loader = DataLoader(segset, batch_size=7)

valid_loss, valid_accu = eval_epoch(model, valid_loader, lossFunction, 'cpu')
print(f'loss:{valid_loss} accu:{valid_accu}')
