import numpy as np
import vamp
import subprocess
import os
from librosa.output import write_wav
from mir_eval.io import load_time_series


from configs.configs import ALGO_BASE_DIRS, ALGO_WRAPPER_INIT
from configs.configs import VERBOSE_OUT, VERBOSE_ERR


def melodia(wav):
    '''<Melody Extraction from Polyphonic Music Signals using Pitch Contour Characteristics>'''
    data, sr = wav
    # extract melody using melodia vamp plugin
    melody = vamp.collect(data, sr, "mtg-melodia:melodia",
                          parameters={"voicing": 0.2})
    hop = melody['vector'][0]
    pitchs = melody['vector'][1]
    # impute missing 0's to compensate for starting timestamp
    pitchs = np.insert(pitchs, 0, [0]*8)
    times = np.arange(0, len(pitchs)) * float(hop)
    return times, pitchs


def callOnFile(wav, commands, wavPath, melPath, scKwargs={}):
    '''call <commands> with <wav> write to <wavPath>, then collect results from <melPath>'''
    if not VERBOSE_OUT:
        scKwargs['stdout'] = subprocess.DEVNULL
    if not VERBOSE_ERR:
        scKwargs['stderr'] = subprocess.DEVNULL
    write_wav(wavPath, wav[0], wav[1])
    ret = subprocess.call(commands, **scKwargs)
    assert ret == 0, f'return value: {ret} != 0'
    times, pitches = load_time_series(melPath, delimiter=r'\s+|,')
    return times, pitches


def separateLeadStereo(wav):
    '''<A musically motivated mid-level representation for pitch estimation and musical audio source separation>'''
    wavPath = os.path.join(TmpDir, 'melInput.wav')
    melPath = os.path.join(TmpDir, 'melInput_pitches.txt')
    excPath = os.path.join(ALGO_BASE_DIRS['SLS'], 'separateLeadStereoParam.py')
    commands = ('python', excPath, wavPath, '-n')
    return callOnFile(wav, commands, wavPath, melPath, {'cwd': ALGO_BASE_DIRS['SLS']})


def sourceFilterContoursMelody(wav):
    '''<A Comparison of Melody Extraction Methods Based on Source-Filter Modelling>'''
    wavPath = os.path.join(TmpDir, 'melInput.wav')
    melPath = os.path.join(TmpDir, 'melInput_pitches.txt')
    excPath = os.path.join(
        ALGO_BASE_DIRS['SFC'], 'MelodyExtractionFromSingleWav.py')
    commands = ('python2', excPath, wavPath, melPath,
                '--extractionMethod=CBM', '--hopsize=0.0029025', '--nb-iterations=30')
    return callOnFile(wav, commands, wavPath, melPath, {'cwd': ALGO_BASE_DIRS['SFC']})


def MCDNN(wav):
    '''<Melody extraction on vocal segments using multi-column deep neural networks>'''
    wavPath = os.path.join(TmpDir, 'melInput.wav')
    melPath = os.path.join(TmpDir, 'melInput_pitches.txt')
    excPath = os.path.join(ALGO_BASE_DIRS['MCDNN'], 'main.py')
    commands = ('python2', excPath, '0.2', wavPath, melPath)
    return callOnFile(wav, commands, wavPath, melPath, {'cwd': ALGO_BASE_DIRS['MCDNN']})


def deepsalience(wav, task='melody2'):
    '''<Deep salience representations for f0 estimation in polyphonic music>'''
    wavPath = os.path.join(TmpDir, 'melInput.wav')
    melPath = os.path.join(TmpDir, f'melInput_{task}_singlef0.csv')
    saveDir = TmpDir
    excPath = os.path.join(
        ALGO_BASE_DIRS['deepSalience'], 'predict_on_audio.py')
    commands = ('python', excPath, wavPath, task, saveDir, '-f', 'singlef0')
    return callOnFile(wav, commands, wavPath, melPath, {'cwd': ALGO_BASE_DIRS['deepSalience']})


def melodicSegnet(wav, gpu_index='0', mode='fast'):
    '''<A Streamlined Encoder/Decoder Architecture for Melody Extraction>'''
    wavPath = os.path.join(TmpDir, 'melInput.wav')
    melPath = os.path.join(TmpDir, 'melInput.txt')
    saveDir = TmpDir
    excPath = os.path.join(
        ALGO_BASE_DIRS['melodicSegnet'], 'predict_on_audio.py')
    commands = ('python', excPath, '-fp', wavPath, '-o', saveDir, '-m', mode)
    if gpu_index is not None:
        commands = commands + ('-gpu', gpu_index)
    return callOnFile(wav, commands, wavPath, melPath, {'cwd': ALGO_BASE_DIRS['melodicSegnet']})


def SFNMF_CRNN(wav):
    '''<Main Melody Extraction with Source-Filter NMF and CRNN>'''
    wavPath = os.path.join(TmpDir, 'melInput.wav')
    melPath = os.path.join(TmpDir, 'SFNMF_CRNN_out.csv')
    excPath = os.path.join(
        ALGO_BASE_DIRS['SFNMF-CRNN'], 'predict_on_single_audio_CRNN.py')
    commands = ('python', excPath, wavPath, melPath)
    return callOnFile(wav, commands, wavPath, melPath, {'cwd': ALGO_BASE_DIRS['SFNMF-CRNN']})


def JDC(wav):
    '''<Joint Detection and Classification of Singing Voice Melody Using Convolutional Recurrent Neural Networks>'''
    wavPath = os.path.join(TmpDir, 'melInput.wav')
    melPath = os.path.join(TmpDir, 'JDC_out.csv')
    excPath = os.path.join(
        ALGO_BASE_DIRS['JDC'], 'melodyExtraction_JDC.py')
    commands = ('python', excPath, wavPath, melPath)
    return callOnFile(wav, commands, wavPath, melPath, {'cwd': ALGO_BASE_DIRS['JDC']})
# TO DO:
# <Vocal melody extraction with semantic segmentation and audio-symbolic domain transfer learning> @ Vocal-Melody-Extraction
# proposed model:


def algoFeng(wav):
    wavPath = os.path.join(TmpDir, 'melInput.wav')
    melPath = os.path.join(TmpDir, 'mel.csv')
    excPath = os.path.join(ALGO_BASE_DIRS['algoFeng'], 'predictFeng.py')
    commands = ('python', excPath, wavPath, melPath)
    return callOnFile(wav, commands, wavPath, melPath, {'cwd': ALGO_BASE_DIRS['algoFeng']})


def algoCao(wav):
    wavPath = os.path.join(TmpDir, 'melInput.wav')
    melPath = os.path.join(TmpDir, 'mel.csv')
    excPath = os.path.join(ALGO_BASE_DIRS['algoCao'], 'predictCao.py')
    commands = ('python', excPath, wavPath, melPath)
    return callOnFile(wav, commands, wavPath, melPath, {'cwd': ALGO_BASE_DIRS['algoCao']})


# initialize
ALGO_WRAPPER_INIT()
TmpDir = ALGO_BASE_DIRS['TmpDir']
