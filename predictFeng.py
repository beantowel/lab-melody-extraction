import torch
import librosa
import numpy as np
import click

from configs.modelFengConfigs import *
from modelFeng.model import MultiDNN
from utils.preprocessor import Compose, STFT, FreqToPitchClass, SelectFreqs


def loadModel(save_data=SAVE_DATA):
    # load model
    device = torch.device('cpu')
    checkpoint = torch.load(SAVE_DATA, map_location=device)
    model_stat_dict = checkpoint['model']
    params = checkpoint['params']
    model = MultiDNN(*params).to(device)
    model.load_state_dict(model_stat_dict)
    return model


def writeMel(output, times, freqs):
    # write melody
    np.savetxt(output, np.array([times, freqs]).T,
               fmt=['%.3f', '%.4f'], delimiter='\t')


def predict_on_audio(model, audiofile):
    # load audio -> features
    y, sr = librosa.load(audiofile, sr=SR)
    transform = Compose([
        STFT(N_FFT, HOP_SIZE, FRAME_SIZE),
    ])
    features = transform({'signal': y})['features']
    features = torch.tensor(features)
    # label timestamps
    sampleStamps = np.arange(0, len(y) - FRAME_SIZE, HOP_SIZE)
    sampleStamps += FRAME_SIZE // 2  # for non-centered STFT
    times = sampleStamps / sr
    # inference
    model.eval()
    with torch.no_grad():
        pitchLogit, voiceLogit = model(features)
        pitch = pitchLogit.argmax(1)
        voice = voiceLogit.argmax(1).byte()  # 0-unvoice 1-voice
        pitch.masked_fill_(~voice, -1)
    # construct melody annotation
    pitch = pitch.numpy()
    freqs = FreqToPitchClass().inv({'labels': pitch})['freqs']
    assert len(times) <= len(freqs), f'{len(times)} mismatch {len(freqs)}'
    return times, freqs[:len(times)]


@click.command()
@click.argument('audiofile', nargs=1, type=click.Path(exists=True))
@click.argument('dst', nargs=1, default='./data/mel.csv', type=click.Path())
def main(audiofile, dst):
    model = loadModel()
    times, freqs = predict_on_audio(model, audiofile)
    writeMel(dst, times, freqs)
    print(f'melody write to {dst}')


if __name__ == '__main__':
    main()
