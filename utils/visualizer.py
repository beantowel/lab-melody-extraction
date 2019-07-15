import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import medfilt, wiener
from mir_eval.io import load_delimited, load_time_series

from configs.configs import PLOTS_DIR
from configs.modelConfigs import SAVE_LOG, MODEL_NAME


def plotLog(mode, logfile, model, savedir):
    # 'epoch,loss,accuracy,mode\n'
    df = pd.read_csv(logfile)
    mask = (df['mode'] == mode)
    # remove duplicate epoch cause of non-latest chekpoint
    # keep epoch in ascending order
    r_epoch = df['epoch'][mask]
    for i, m in enumerate(mask):
        if m and (r_epoch.loc[i] > min(r_epoch.loc[i:])):
            mask[i] = False

    epoch = df['epoch'][mask]
    loss, pa, va, oa = df['loss'][mask], df['PA'][mask], df['VA'][mask], df['OA'][mask]

    fig, ax1 = plt.subplots()
    plt.title(mode)

    color = 'tab:red'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss', color=color)
    ax1.plot(epoch, loss, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    # we already handled the x-label with ax1
    ax2.set_ylabel('accuracy', color=color)
    ax2.plot(epoch, pa, label='PA')
    ax2.plot(epoch, va, label='VA')
    ax2.plot(epoch, oa, label='OA')
    ax2.legend()
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    savefile = os.path.join(savedir, f'{model}_{mode}.svg')
    plt.savefig(savefile)
    return plt


def showLog(logfile=SAVE_LOG, model=MODEL_NAME, savedir=PLOTS_DIR):
    plt = plotLog('training', logfile, model, savedir)
    plt.show()
    plt = plotLog('validation', logfile, model, savedir)
    plt.show()


def loadMel(melfile):
    try:
        times, freqs = load_time_series(
            melfile, delimiter=r'\s+|\t+|,')
    except ValueError:
        cols = load_delimited(melfile, [int, int, str, float, float])
        times, freqs = np.array(cols[0]) / 100., np.array(cols[3])
    return times, freqs


def showDuration(melfile, srcDir=None):
    if srcDir is not None:
        files = os.listdir(srcDir)
        melFiles = list(map(lambda x: os.path.join(srcDir, x), files))
    else:
        melFiles = [melfile]

    durations = np.array([])
    for melfile in melFiles:
        times, freqs = loadMel(melfile)
        midis = librosa.hz_to_midi(freqs).astype(int)
        midis[freqs <= 0] = 0
        dMidi = np.diff(np.insert(midis, 0, 0))
        onsets = times[:-1][(freqs[:-1] > 0) &
                            (dMidi[:-1] != 0) & (dMidi[1:] == 0)]
        durations = np.concatenate([durations, np.diff(onsets)])

    plt.hist(durations, bins=25)
    plt.xlabel('duration')
    plt.show()
    return durations


def showMidi(melfile, srcDir=None):
    if srcDir is not None:
        files = os.listdir(srcDir)
        melFiles = list(map(lambda x: os.path.join(srcDir, x), files))
    else:
        melFiles = [melfile]

    midis = np.array([])
    for melfile in melFiles:
        times, freqs = loadMel(melfile)
        x = librosa.hz_to_midi(freqs[freqs > 0])
        midis = np.concatenate([midis, x])

    plt.hist(midis, bins=25)
    plt.xlabel('midi')
    plt.show()
    return midis


def showMel(melfiles, grid=(44100, 2048, 512)):
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    timesList, freqsList = [], []
    for i, melfile in enumerate(melfiles):
        times, freqs = loadMel(melfile)
        times, freqs = times[times < 4], freqs[times < 4]
        timesList.append(times)
        freqsList.append(freqs)
    plotMel(timesList, freqsList, grid=grid)


def plotMel(timesList, freqsList, title='', grid=(44100, 2048, 512)):
    for i, (times, freqs) in enumerate(zip(timesList, freqsList)):
        midis = np.zeros([len(freqs)])
        midis[freqs > 0] = librosa.hz_to_midi(freqs[freqs > 0])
        seminotes = midis.astype(int)+0.5

        plt.plot(times[midis > 0], midis[midis > 0], label=f'midi {i}')
        # plt.plot(times, seminotes, label=f'seminote {i}')

    sr, n_fft, hop_length = grid
    frames = librosa.time_to_frames(times[-1], sr, hop_length, n_fft)
    for t in librosa.frames_to_time(range(frames), sr, hop_length, n_fft):
        plt.axvline(x=t, color='g')
    for f in librosa.fft_frequencies(sr, n_fft):
        if f >= np.min(freqs[freqs > 0])*0.9 and f <= np.max(freqs)*1.1:
            plt.axhline(y=librosa.hz_to_midi(f), color='g')

    plt.xlabel('time')
    plt.ylabel('midi note')
    plt.legend()
    plt.title(f'melody:{title}')
    plt.show()
