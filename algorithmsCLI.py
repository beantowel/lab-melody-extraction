import torch
import librosa
import numpy as np
import click

from utils.algorithmsWrapper import *
from utils.common import writeMel

algos = {
    'Salamon': melodia,
    'Bittner': deepsalience,
    'Hsieh': melodicSegnet,
    'Kum': JDC,
    'Proposed': algoProposed,
    'PeakSel': algoPeakSel,
    'Durrieu': separateLeadStereo,
    'Basaran': SFNMF_CRNN,
    'Bosch': sourceFilterContoursMelody,
}


@click.command()
@click.argument('algo', nargs=1, type=click.STRING)
@click.argument('audiofile', nargs=1, type=click.Path(exists=True))
@click.argument('melfile', nargs=1, default='./data/mel.csv', type=click.Path())
def main(algo, audiofile, melfile):
    print(f'melody extraction with {algo}')
    f = algos[algo]
    wav = librosa.load(audiofile, sr=44100)
    times, pitchs = f(wav)
    writeMel(melfile, times, pitchs)
    print(f'melody write to {melfile}')


if __name__ == '__main__':
    main()
