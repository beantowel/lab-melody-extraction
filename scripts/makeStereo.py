#!~/anaconda3/bin python

import click
import mir_eval
import numpy as np
import librosa

from os import path
from scipy.io import wavfile


def toMono(data):
    if len(data.shape) > 1:
        data = data[:, 0]
    return data


def makeStereo(dataL, dataR, rate):
    dataL = toMono(dataL)
    dataR = toMono(dataR)
    dlen = abs(len(dataR) - len(dataL))
    print('pad {} samples, {:.3f}s'.format(dlen, float(dlen) / rate))
    if len(dataL) > len(dataR):
        dataR = np.pad(dataR, (0, dlen), 'constant', constant_values=(0, 0))
    else:
        dataL = np.pad(dataL, (0, dlen), 'constant', constant_values=(0, 0))
    stereo = np.vstack([dataL, dataR]).T
    return stereo


@click.command()
@click.argument('leftChannel', nargs=1, type=click.Path(exists=True))
@click.argument('rightChannel', nargs=1, type=click.Path(exists=True))
@click.argument('dst', nargs=1, type=click.Path())
def writeStereo(leftchannel, rightchannel, dst):
    dataL, rateL = librosa.load(leftchannel, sr=44100)
    dataR, rateR = librosa.load(rightchannel, sr=44100)
    lName, _ = path.splitext(path.basename(leftchannel))
    rName, _ = path.splitext(path.basename(rightchannel))
    outName = 'stereo_{}_{}_.wav'.format(lName, rName)
    outPath = path.join(dst, outName)
    print('l:{}\nr:{}'.format(leftchannel, rightchannel))
    print('rate:{}, {}'.format(rateL, rateR))
    print('shape:{}, {}'.format(dataL.shape, dataR.shape))
    print('write to:', outPath)
    if rateR == rateL:
        stereo = makeStereo(dataL, dataR, rateR)
        wavfile.write(outPath, rateL, stereo)


script_dir = path.dirname(__file__)
if __name__ == "__main__":
    writeStereo()
