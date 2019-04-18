import os
import torch

# dataset, dataloader params
TRAIN_RATIO = 0.9
BATCH_SIZE = 6144
NUM_WORKERS = 12
CACHE_SIZE = 61
SR = 22050
N_FFT = 2048
FRAME_SIZE = 2048
HOP_SIZE = 512
NOTE_LOW = 'A1'
NOTE_HIGH = 'A6'
BIN_RESOLUTION = 120

# model params
DIM_FEATURE = 1025
DIM_HIDDEN = (2048, 2048, 1024)
NUM_PITCH = 120  # Pitch
NUM_VOICE = 2  # Voice or not
DROPOUT = 0.5
# merge params to list for checkpoint
MODEL_PARAMS = (
    DIM_FEATURE,
    DIM_HIDDEN,
    NUM_PITCH,
    NUM_VOICE,
    DROPOUT,
)

# optimizer, loss params
WEIGHT_DECAY = 0.001  # 0.0001
LEARNING_RATE = 0.0001
LOSS_RATIO = 0.5
AUXILIARY_WEIGHT = 0.5

# trainer params
MODEL_NAME = 'multiDNN'
DEVICE = torch.device('cuda')
SAVE_DIR = './data'
SAVE_LOG = os.path.join(SAVE_DIR, f'{MODEL_NAME}.log')
SAVE_DATA = os.path.join(SAVE_DIR, f'{MODEL_NAME}.pkl')
NUM_EPOCH = 70
MAX_STEP_NUM = int(1e9)
