import numpy as np
import vamp
import subprocess
import os
from librosa.output import write_wav
from mir_eval.io import load_time_series


from configs.configs import TmpDir, SLS_DIR, SFC_DIR, MCDNN_DIR, DEEPSALIENCE_DIR
from configs.configs import VERBOSE_OUT, VERBOSE_ERR


def melodia(wav):
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
    if not VERBOSE_OUT:
        scKwargs['stdout'] = subprocess.DEVNULL
    if not VERBOSE_ERR:
        scKwargs['stderr'] = subprocess.DEVNULL
    write_wav(wavPath, wav[0], wav[1])
    ret = subprocess.call(commands, **scKwargs)
    assert ret == 0
    times, pitches = load_time_series(melPath)
    return times, pitches


def seprateLeadStereo(wav):
    wavPath = os.path.join(TmpDir, 'melInput.wav')
    melPath = os.path.join(TmpDir, 'melInput_pitches.txt')
    excPath = os.path.join(SLS_DIR, 'separateLeadStereoParam.py')
    commands = ['python', excPath, wavPath, '-n']
    return callOnFile(wav, commands, wavPath, melPath, {'cwd': SLS_DIR})


def sourceFilterContoursMelody(wav):
    wavPath = os.path.join(TmpDir, 'melInput.wav')
    melPath = os.path.join(TmpDir, 'melInput_pitches.txt')
    excPath = os.path.join(SFC_DIR, 'MelodyExtractionFromSingleWav.py')
    commands = ['python2', excPath, wavPath, melPath,
                '--extractionMethod=CBM', '--hopsize=0.0029025', '--nb-iterations=30']
    return callOnFile(wav, commands, wavPath, melPath, {'cwd': SFC_DIR})


def MCDNN(wav):
    wavPath = os.path.join(TmpDir, 'melInput.wav')
    melPath = os.path.join(TmpDir, 'melInput_pitches.txt')
    excPath = os.path.join(MCDNN_DIR, 'main.py')
    commands = ['python2', excPath, '0.2', wavPath, melPath]
    return callOnFile(wav, commands, wavPath, melPath, {'cwd': MCDNN_DIR})


def deepsalience(wav, task='melody2'):
    wavPath = os.path.join(TmpDir, 'melInput.wav')
    melPath = os.path.join(TmpDir, f'melInput_{task}_singlef0.csv')
    saveDir = TmpDir
    excPath = os.path.join(DEEPSALIENCE_DIR, 'predict_on_audio.py')
    commands = ['python', excPath, wavPath, task, saveDir, '-f', 'singlef0']
    return callOnFile(wav, commands, wavPath, melPath, {'cwd': DEEPSALIENCE_DIR})
