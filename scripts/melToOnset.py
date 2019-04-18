#!/home/beantowel/anaconda3/bin python

import os
import click
import mir_eval
import numpy as np


def onsetDetect(times, freqs):
    df = np.diff(np.insert(freqs, 0, 0))
    return times[(df != 0) & (freqs > 0)]


def writeToFile(pathFile, ref_time, ref_freq):
    onset_time = onsetDetect(ref_time, ref_freq)
    np.savetxt(pathFile, onset_time, delimiter='\t', fmt='%.2f')
    return onset_time


def readFromFile(pathRefFile):
    ref_time, ref_freq = mir_eval.io.load_time_series(pathRefFile, delimiter=r'\s+|\t+|,')
    print('freqs:', ref_freq)
    return ref_time, ref_freq


@click.command()
@click.argument('src', nargs=-1, type=click.Path(exists=True))
@click.argument('dst', nargs=1, type=click.Path())
def transFiles(src, dst):
    for filename in src:
        head, tail = os.path.split(filename)
        output = os.path.join(dst, tail + '.onset')
        
        click.echo('transform: ' + filename)
        ref_time, ref_freq = readFromFile(filename)
        click.echo('write to: ' + output)
        writeToFile(output, ref_time, ref_freq)


script_dir = os.path.dirname(__file__)
if __name__ == "__main__":
    transFiles()
