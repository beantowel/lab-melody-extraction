import os

VERBOSE_OUT = False
VERBOSE_ERR = False

BaseDirs = {
    'MedleyDB': '/home/beantowel/FDU/MIR/dataset/melodyExtraction/MedleyDB',
    'RWC': '/home/beantowel/FDU/MIR/dataset/melodyExtraction/RWC',
    'adc2004': '/home/beantowel/FDU/MIR/dataset/melodyExtraction/adc2004_full_set',
    'Orchset': '/home/beantowel/FDU/MIR/dataset/melodyExtraction/Orchset',
}

TmpDir = '/tmp/Lab_MelExt'
SLS_DIR = '/home/beantowel/FDU/MIR/Projects/separateLeadStereo'
MCDNN_DIR = '/home/beantowel/FDU/MIR/Projects/MelodyExtraction_MCDNN'
SFC_DIR = '/home/beantowel/FDU/MIR/Projects/SourceFilterContoursMelody/src'
DEEPSALIENCE_DIR = '/home/beantowel/FDU/MIR/Projects/ismir2017-deepsalience/predict/'

try:
    os.mkdir(TmpDir)
except FileExistsError:
    pass
