import os
import torch
import numpy as np

SEMINOTE_RESOLUTION = 9
STFT_PHASE = False

# model params
DIM_FEATURE = (1025 * 2) if STFT_PHASE else 1025
_D_HIDDEN = 1024
DIM_HIDDEN = (_D_HIDDEN*2, _D_HIDDEN*2, _D_HIDDEN)
LEN_SEQ = 64
NUM_RNN_LAYER = 2
# C2-C6, 12 seminote per octave, 7 Pitch class per seminote
NUM_PITCH = (6 - 1) * 12 * SEMINOTE_RESOLUTION
NUM_VOICE = 2  # Voice or not
DROPOUT = 0.7
# use gpu
DEVICE = torch.device('cuda')
# merge params to list for checkpoint
USE_MODEL = os.environ.get('MEL_EXT_USE_MODEL', 'MultiDNN_RNN')
MODEL_PARAMS = {
    'MultiDNN': (
        DIM_FEATURE,
        DIM_HIDDEN,
        LEN_SEQ,
        NUM_PITCH,
        NUM_VOICE,
        DROPOUT,
        DEVICE,
    ),
    'MultiDNN_RNN': (
        DIM_FEATURE,
        DIM_HIDDEN,
        LEN_SEQ,
        NUM_RNN_LAYER,
        NUM_PITCH,
        NUM_VOICE,
        DROPOUT,
        DEVICE,
    ),
}[USE_MODEL]

# dataset, dataloader params
TRAIN_RATIO = 0.9
BATCH_SIZE = 512
NUM_WORKERS = 12
CACHE_SIZE = 256
SR = 22050
N_FFT = 2048
FRAME_SIZE = N_FFT
HOP_SIZE = FRAME_SIZE // 4
SEG_FRAME = (FRAME_SIZE - HOP_SIZE) + HOP_SIZE * LEN_SEQ
SEG_HOP = HOP_SIZE
NOTE_LOW = 'A1'
NOTE_HIGH = 'A6'
BIN_RESOLUTION = NUM_PITCH
assert FRAME_SIZE / SR < 0.1  # frame duration < 0.1s
assert LEN_SEQ * HOP_SIZE / SR < 3  # context duration < 3s

# optimizer, loss params
WEIGHT_DECAY = 0.001  # 0.0001
LEARNING_RATE = 0.0001
LOSS_RATIO = 0.5
AUXILIARY_WEIGHT = float(os.environ.get('MEL_EXT_AUX_WEIGHT', 0.5))
LABEL_DEVIATION = SEMINOTE_RESOLUTION / 4
# blur matrix(nPitchClass, nBlurPitchClass)
LABEL_BLUR_MATRIX = torch.diag(torch.ones(NUM_PITCH))
UNVOICE_LABEL_SMOOTH = np.exp(-SEMINOTE_RESOLUTION **
                              2 / (2 * LABEL_DEVIATION**2))
for i in range(1, SEMINOTE_RESOLUTION//2):
    blur_i = torch.diag(torch.ones(NUM_PITCH - i), i) * \
        np.exp(-i**2 / (2 * LABEL_DEVIATION**2))
    LABEL_BLUR_MATRIX += blur_i + blur_i.transpose(0, 1)
if DEVICE == torch.device('cuda') and torch.cuda.device_count() > 0:
    LABEL_BLUR_MATRIX = LABEL_BLUR_MATRIX.to(DEVICE)

# trainer params
MODEL_NAME = USE_MODEL
PRETRAINED_NAME = 'MultiDNN'
SAVE_DIR = './data'
PRETRAINED_MODEL = os.path.join(SAVE_DIR, f'{PRETRAINED_NAME}.pkl')
SAVE_LOG = os.path.join(SAVE_DIR, f'{MODEL_NAME}.log')
SAVE_DATA = os.path.join(SAVE_DIR, f'{MODEL_NAME}.pkl')
NUM_EPOCH = 10
MAX_STEP_NUM = int(1e9)

UNVOICE_THRESH = 0.5
