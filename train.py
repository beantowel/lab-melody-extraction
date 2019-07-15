import time
import click
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from model.model import MODEL_CLASS, multiTaskLoss, AccuracyCounter
from utils.dataset import MedleyDB_vocal_Dataset, MedleyDB_instrumental_Dataset, Adc2004_vocal_Dataset, Adc2004_instrumental_Dataset, RWC_Popular_Dataset, RWC_Royalty_Free_Dataset, Orchset_Dataset, Segments_Dataset
from utils.preprocessor import Compose, FreqToPitchClass, STFT, SelectFreqs, LogSignal
from configs.modelConfigs import *


def prepareLoaders(DEBUG=True):
    if DEBUG:
        Adc2004Base = Adc2004_vocal_Dataset(sr=SR).randomSplit(TRAIN_RATIO)
        train_set = Adc2004Base[0]  # small dataset for debugging
        valid_set = Adc2004Base[1]
    else:
        MedleyBase = MedleyDB_vocal_Dataset(sr=SR).randomSplit(TRAIN_RATIO)
        # RWCBase = RWC_Popular_Dataset(sr=SR).randomSplit(TRAIN_RATIO)
        train_set = MedleyBase[0]
        valid_set = MedleyBase[1]
        # train_set = RWCBase[0]
        # valid_set = RWCBase[1]
        # OrchsetBase = Orchset_Dataset(sr=SR).randomSplit(TRAIN_RATIO)
        # train_set = OrchsetBase[0]
        # valid_set = OrchsetBase[1]

    transform = Compose([
        SelectFreqs(SR, HOP_SIZE, FRAME_SIZE),
        FreqToPitchClass(NOTE_LOW, NOTE_HIGH, BIN_RESOLUTION),
        STFT(N_FFT, HOP_SIZE, FRAME_SIZE, phase=STFT_PHASE),
        # LogSignal(FRAME_SIZE, HOP_SIZE),
    ])
    # make segments loader
    trainSeg_set = Segments_Dataset(
        train_set, SR, SEG_FRAME, SEG_HOP, transform=transform, cacheSize=CACHE_SIZE)
    validSeg_set = Segments_Dataset(
        valid_set, SR, SEG_FRAME, SEG_HOP, transform=transform, cacheSize=CACHE_SIZE)
    trainSeg_set.warmUp()
    validSeg_set.warmUp()

    train_loader = DataLoader(
        trainSeg_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    valid_loader = DataLoader(
        validSeg_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    return train_loader, valid_loader


def init():
    print(f'[ Info ] model: {MODEL_NAME}({MODEL_PARAMS})')
    model = MODEL_CLASS(*MODEL_PARAMS)
    # create log file
    print(
        f'[ Info ] Training performance will be written to file: {SAVE_LOG}')
    with open(SAVE_LOG, 'w') as logf:
        logf.write('epoch,loss,PA,VA,OA,mode\n')
    return model


def resumeFrom(save_data, device=DEVICE):
    checkpoint = torch.load(save_data, map_location=device)
    model_stat_dict = checkpoint['model']
    params = checkpoint['params']
    epoch_r = checkpoint['epoch']

    model = MODEL_CLASS(*params).to(device)
    model.load_state_dict(model_stat_dict)
    return model, epoch_r


def train_epoch(model, train_loader, optimizer, lossFunction):
    model.train()
    total_loss = 0.
    counter = AccuracyCounter()
    for batch in tqdm(train_loader, mininterval=2, desc='  - (Training)', leave=False):
        # get data
        feature = batch['features'].to(DEVICE)
        label = batch['labels'].to(DEVICE)
        # forward
        optimizer.zero_grad()
        output = model(feature)
        # backward
        loss = lossFunction(output, label)
        loss.backward()
        total_loss += loss.item()
        # update parameters
        optimizer.step()
        # counting
        counter.update(output, label)

    ave_loss = total_loss / len(train_loader)
    return ave_loss, counter.getAccuacy()


def eval_epoch(model, valid_loader, lossFunction):
    model.eval()
    total_loss = 0.
    counter = AccuracyCounter()
    with torch.no_grad():
        for batch in tqdm(valid_loader, mininterval=2, desc='  - (Validation)', leave=False):
            # get data
            feature = batch['features'].to(DEVICE)
            label = batch['labels'].to(DEVICE)
            # forward
            output = model(feature)
            # calculate loss
            loss = lossFunction(output, label)
            total_loss += loss.item()
            # counting
            counter.update(output, label)

    ave_loss = total_loss / len(valid_loader)
    return ave_loss, counter.getAccuacy()


def train(model, train_loader, valid_loader, optimizer, lossFunction, epoch_r=0, debug=False):
    valid_losses, valid_accus = [], []
    for epoch_i in range(1, 1 + NUM_EPOCH):
        print(f'[ Epoch {epoch_r} + {epoch_i} ]')

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, train_loader, optimizer, lossFunction)
        pa, va, oa = train_accu
        elapse = (time.time() - start) / 60
        print(
            f'  - ( Training ) loss:{train_loss:3.5f}, PA:{pa:3.4f}, VA:{va:3.4f}, OA:{oa:3.4f}, elapse:{elapse:4.2f}min')

        start = time.time()
        valid_loss, valid_accu = eval_epoch(
            model, valid_loader, lossFunction)
        pa, va, oa = valid_accu
        elapse = (time.time() - start) / 60
        print(
            f'  - (Validation) loss:{valid_loss:3.5f}, PA:{pa:3.4f}, VA:{va:3.4f}, OA:{oa:3.4f}, elapse:{elapse:4.2f}min')

        valid_losses += [valid_loss]
        valid_accus += [valid_accu[-1]]  # OA
        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'params': MODEL_PARAMS,
            'epoch': epoch_i + epoch_r,
        }
        if (SAVE_DATA is not None):
            if valid_accu[-1] >= max(valid_accus):
                torch.save(checkpoint, SAVE_DATA)
        if (SAVE_LOG is not None):
            with open(SAVE_LOG, 'a') as logf:
                pa, va, oa = train_accu
                logf.write(
                    f'{epoch_i+epoch_r},{train_loss:3.5f},{pa:3.4f},{va:3.4f},{oa:3.4f},training\n')
                pa, va, oa = valid_accu
                logf.write(
                    f'{epoch_i+epoch_r},{valid_loss:3.5f},{pa:3.4f},{va:3.4f},{oa:3.4f},validation\n')


@click.command()
@click.option('--resume', default=False, type=click.BOOL, help='resume training state from .pkl file')
@click.option('--debug', default=False, type=click.BOOL, help='use small dataset to debug faster')
def main(resume, debug):
    if not resume:
        model = init()
        epoch_r = 0
    else:
        print(f'[ Info ] resume state from {SAVE_DATA} to {DEVICE}')
        model, epoch_r = resumeFrom(SAVE_DATA)
    train_loader, valid_loader = prepareLoaders(debug)

    optimizer = torch.optim.Adam(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    lossFunction = multiTaskLoss

    train(model, train_loader, valid_loader, optimizer,
          lossFunction, epoch_r=epoch_r, debug=debug)


if __name__ == '__main__':
    main()
