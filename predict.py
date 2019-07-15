import torch
import librosa
import numpy as np
import click
from tqdm import tqdm
from scipy.signal import medfilt

from configs.modelConfigs import *
from model.model import MODEL_CLASS
from utils.common import writeMel
from utils.preprocessor import Compose, STFT, FreqToPitchClass, SelectFreqs


def loadModel(save_data, device=DEVICE):
    # load model
    print(f'load model from:{save_data}')
    checkpoint = torch.load(SAVE_DATA, map_location=device)
    model_stat_dict = checkpoint['model']
    params = list(checkpoint['params'])
    assert isinstance(params[-1], torch.device), f'params:{params}'
    params[-1] = device
    model = MODEL_CLASS(*params)
    model.load_state_dict(model_stat_dict)
    return model


def evalSeq2SeqModel(model, features, hop=None):
    # seq2seq model for overlapping segments
    model.eval()
    with torch.no_grad():
        l_seq = LEN_SEQ
        hop = l_seq // 4 if hop is None else hop
        # segments start timestamp indices tuple
        segStarts = np.arange(0, features.shape[0] - l_seq, hop)
        # add last segment timestamp index
        segStarts = np.concatenate([segStarts, [features.shape[0] - l_seq]])
        segPitches = []
        # eval on l_seq segments
        for start in tqdm(segStarts):
            in_features = features[start: start+l_seq].view(1, l_seq, -1)
            pitchLogit, voiceLogit = model(in_features)
            pitchLogit = pitchLogit.view(-1, NUM_PITCH)
            voiceLogit = voiceLogit.view(-1, NUM_VOICE)

            # # class inference by probability
            # pitch = torch.matmul(pitchLogit, torch.arange(
            #     NUM_PITCH, dtype=torch.float)) / torch.sum(LABEL_BLUR_MATRIX[seminoteResolution])
            pitch = pitchLogit.argmax(1)
            voice = voiceLogit.argmax(1).byte()  # 0-unvoice 1-voice

            pitch.masked_fill_(~voice, -1)
            segPitches.append(pitch.view(-1).cpu().numpy())
        # collect pitches for each timestamp
        pitchCandidates = [[] for i in range(features.shape[0])]
        for start, segPitch in zip(segStarts, segPitches):
            for i in range(l_seq):
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


def evalSeq2ClsModel(model, features, hop=None):
    # sequence to single class model for overlapping segments
    model.eval()
    with torch.no_grad():
        l_seq = LEN_SEQ
        hop = 1
        # pad feautre with (l_seq - 1) zero-vectors at beginning
        features = torch.cat(
            [torch.zeros((l_seq-1, features.shape[1])), features])
        # segments start timestamp indices tuple (order not required)
        segStarts = np.arange(0, features.shape[0] - l_seq, hop)
        # add last segment timestamp index
        segStarts = np.concatenate([segStarts, [features.shape[0] - l_seq]])
        segPitches = []
        # eval on l_seq segments
        for start in tqdm(segStarts):
            in_features = features[start: start+l_seq].view(1, l_seq, -1)
            pitchLogit, voiceLogit = model(in_features)
            pitch = pitchLogit.argmax(1)
            voice = voiceLogit.argmax(1).byte()  # 0-unvoice 1-voice
            pitch.masked_fill_(~voice, -1)
            segPitches.append(pitch.view(-1).numpy())
        # collect pitches for each timestamp

        pitch = []
        for start, segPitch in zip(segStarts, segPitches):
            pitch.append(segPitch[0])
        pitch = np.array(pitch)
    print('eval done')
    return pitch


def predict_on_audio(model, audiofile, device=DEVICE):
    # load audio -> features
    y, sr = librosa.load(audiofile, sr=SR)
    transform = Compose([
        STFT(N_FFT, HOP_SIZE, FRAME_SIZE, phase=STFT_PHASE),
    ])
    features = transform({'signal': y})['features']
    features = torch.tensor(features).to(device)
    # label timestamps
    sampleStamps = np.arange(0, len(y) - FRAME_SIZE, HOP_SIZE)
    sampleStamps += FRAME_SIZE // 2  # for non-centered STFT
    times = sampleStamps / sr
    # inference
    print('inferencing')
    # evalSeq2ClsModel(model, features)
    pitch = evalSeq2SeqModel(model, features)
    # construct melody annotation
    freqs = FreqToPitchClass(NOTE_LOW, NOTE_HIGH, BIN_RESOLUTION).inv(
        {'labels': pitch})['freqs']
    assert len(times) <= len(freqs), f'{len(times)} mismatch {len(freqs)}'
    return times, freqs[:len(times)]


@click.command()
@click.argument('audiofile', nargs=1, type=click.Path(exists=True))
@click.argument('melfile', nargs=1, default='./data/mel.csv', type=click.Path())
@click.option('--model', default=SAVE_DATA, type=click.STRING, help='pretrained model path')
@click.option('--cpu', default=False, type=click.BOOL, help='use when have no cuda support')
def main(audiofile, melfile, model, cpu):
    device = torch.device('cuda') if (not cpu) and (
        torch.cuda.device_count() > 0) else torch.device('cpu')
    model = loadModel(model, device)
    times, freqs = predict_on_audio(model, audiofile, device)
    writeMel(melfile, times, freqs)
    print(f'melody write to {melfile}')


if __name__ == '__main__':
    main()
