#!~/anaconda3/bin python

import os
import click
import mir_eval
import numpy as np
from scipy.io import wavfile
from mir_eval.io import load_delimited, load_time_series


def freqToWav(times, freqs, sampleRate):
    # copied from https://github.com/justinsalamon/melosynth <GPL v3 license>
    """MeloSynth: synthesize a melody
        Copyright (C) 2014 Justin Salamon.
        MeloSynth is free software: you can redistribute it and/or modify it under the
        terms of the GNU General Public License as published by the Free Software
        Foundation, either version 3 of the License, or (at your option) any later
        version.
        MeloSynth is distributed in the hope that it will be useful, but WITHOUT ANY
        WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
        PARTICULAR PURPOSE.  See the GNU General Public License for more details.
        You should have received a copy of the GNU General Public License along with
        this program. If not, see <http://www.gnu.org/licenses/>.
    """
    # Preprocess pitch sequence
    freqs[freqs < 0] = 0
    # Impute silence if start time > 0
    if times[0] > 0:
        estimated_hop = np.median(np.diff(times))
        prev_time = max(times[0] - estimated_hop, 0)
        times = np.insert(times, 0, prev_time)
        freqs = np.insert(freqs, 0, 0)

    print('Generating wave...')
    signal = []

    transLen = 0.010  # duration (in seconds) for fade in/out and freq interp
    phase_prev = 0.0
    f_prev = 0.0  # previous frequency
    t_prev = 0.0  # previous timestamp
    for t, f in zip(times, freqs):
        nSamples = int(np.round((t - t_prev) * sampleRate))
        if nSamples > 0:
            # calculate transition length (in samples)
            transSamples = float(min(np.round(transLen*sampleRate), nSamples))
            # Initiate frequency series
            freq_series = np.ones(nSamples) * f_prev
            # Interpolate between non-zero frequencies
            freq_series += np.minimum(np.arange(nSamples) /
                                      transSamples, 1) * (f - f_prev)
            # calculate phase increments
            dPhase = 2.0 * np.pi * freq_series / \
                float(sampleRate)  # dfai=2*pi*f*dt
            phases = phase_prev + np.cumsum(dPhase)
            # update phase
            phase_prev = phases[-1]
            samples = np.sin(phases)
            # set unvoiced amplitude to zero
            samples[freq_series == 0] = 0
            signal.extend(samples)
        t_prev = t
        f_prev = f

    signal = np.asarray(signal)
    return signal


def writeToFile(pathFile, ref_time, ref_freq):
    sampleRate = 44100
    signal = freqToWav(ref_time, ref_freq, sampleRate)
    # normalize signal
    scaled = np.int16(signal / np.max(np.abs(signal)) * 0x7fff / 4.0)
    print('Saving wav file...')
    wavfile.write(pathFile, sampleRate, scaled)
    return scaled


def loadMel(melfile):
    try:
        times, freqs = load_time_series(
            melfile, delimiter=r'\s+|\t+|,')
    except ValueError:
        cols = load_delimited(melfile, [int, int, str, float, float])
        times, freqs = np.array(cols[0]) / 100., np.array(cols[3])
    return times, freqs


@click.command()
@click.argument('src', nargs=-1, type=click.Path(exists=True))
@click.argument('dst', nargs=1, type=click.Path())
def transFiles(src, dst):
    for filename in src:
        head, tail = os.path.split(filename)
        output = os.path.join(dst, tail + '.wav')

        click.echo('transform: ' + filename)
        ref_time, ref_freq = loadMel(filename)
        click.echo('write to: ' + output)
        writeToFile(output, ref_time, ref_freq)


script_dir = os.path.dirname(__file__)
if __name__ == "__main__":
    transFiles()
