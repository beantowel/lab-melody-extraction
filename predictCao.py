import torch
import librosa
import numpy as np
import click

from configs.modelCaoConfigs import *
from modelCao.model import ScopeMultiDNN
from utils.preprocessor import Compose, STFT, FreqToPitchClass, SelectFreqs


def loadModel(save_data):
    # load model
    print(f'load model from {save_data}')
    device = torch.device('cpu')
    checkpoint = torch.load(SAVE_DATA, map_location=device)
    model_stat_dict = checkpoint['model']
    params = list(checkpoint['params'])
    assert isinstance(params[-1], torch.device), f'params:{params}'
    params[-1] = device
    model = ScopeMultiDNN(*params)
    model.load_state_dict(model_stat_dict)
    return model


def writeMel(output, times, freqs):
    # write melody
    np.savetxt(output, np.array([times, freqs]).T,
               fmt=['%.3f', '%.4f'], delimiter='\t')


def evalModel(model, features, hop=None):
    model.eval()
    with torch.no_grad():
        scope = model.scope
        hop = scope // 4 if hop is None else hop
        segStarts = np.arange(0, features.shape[0] - scope, hop)
        segStarts = np.insert(segStarts, 0, features.shape[0] - scope)
        segPitches = []
        # eval on scope segments
        for start in segStarts:
            in_features = features[start: start+scope].view(1, scope, -1)
            pitchLogit, voiceLogit = model(in_features)
            pitch = pitchLogit.argmax(1)
            voice = voiceLogit.argmax(1).byte()  # 0-unvoice 1-voice
            pitch.masked_fill_(~voice, -1)
            segPitches.append(pitch.view(-1).numpy())
        # collect pitches
        pitchCandidates = [[] for i in range(features.shape[0])]
        for start, segPitch in zip(segStarts, segPitches):
            for i in range(scope):
                pitchCandidates[start + i].append(segPitch[i])
        pitch = []
        for candidate in pitchCandidates:
            if candidate.count(-1) >= len(candidate) / UNVOICE_THRESH:
                pitch.append(-1)  # unvoice
            else:
                pitch.append(np.median(candidate))  # median pitch
        pitch = np.array(pitch)
    print('eval done')
    return pitch


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
    pitch = evalModel(model, features)
    # construct melody annotation
    freqs = FreqToPitchClass().inv({'labels': pitch})['freqs']
    assert len(times) <= len(freqs), f'{len(times)} mismatch {len(freqs)}'
    return times, freqs[:len(times)]


@click.command()
@click.argument('audiofile', nargs=1, type=click.Path(exists=True))
@click.argument('melfile', nargs=1, default='./data/mel.csv', type=click.Path())
@click.option('--modelparams', default=SAVE_DATA, type=click.STRING)
def main(audiofile, melfile, modelparams):
    model = loadModel(modelparams)
    times, freqs = predict_on_audio(model, audiofile)
    writeMel(melfile, times, freqs)
    print(f'melody write to {melfile}')


if __name__ == '__main__':
    main()
