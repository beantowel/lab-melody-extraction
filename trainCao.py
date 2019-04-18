import time
import click
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from modelCao.model import ScopeMultiDNN, MultiTaskLoss
from utils.dataset import MedleyDB_vocal_Dataset, Adc2004_Dataset, Segments_Dataset
from utils.preprocessor import Compose, FreqToPitchClass, STFT, SelectFreqs
from configs.modelCaoConfigs import *


def init():
    # prepare dataloader
    train_set, valid_set = MedleyDB_vocal_Dataset().randomSplit(TRAIN_RATIO)
    # train_set, valid_set = Adc2004_Dataset().randomSplit(TRAIN_RATIO) # smaller dataset for running test

    transform = Compose([
        SelectFreqs(SR, HOP_SIZE, FRAME_SIZE),
        FreqToPitchClass(NOTE_LOW, NOTE_HIGH, BIN_RESOLUTION),
        STFT(N_FFT, HOP_SIZE, FRAME_SIZE),
    ])
    trainSeg_set = Segments_Dataset(
        train_set, SR, SEG_FRAME, SEG_HOP, transform=transform)
    validSeg_set = Segments_Dataset(
        valid_set, SR, SEG_FRAME, SEG_HOP, transform=transform)
    train_loader = DataLoader(
        trainSeg_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(
        validSeg_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    # prepare model
    model = ScopeMultiDNN(DIM_FEATURE, DIM_HIDDEN, NUM_STAGE,
                          NUM_PITCH, NUM_VOICE, DROPOUT, DEVICE)
    # create log file
    print(
        f'[ Info ] Training performance will be written to file: {SAVE_LOG}')
    with open(SAVE_LOG, 'w') as logf:
        logf.write('epoch,loss,accuracy,mode\n')
    return model, train_loader, valid_loader


def resumeFrom(save_data, device=DEVICE):
    checkpoint = torch.load(save_data, map_location=device)
    model_stat_dict = checkpoint['model']
    params = checkpoint['params']
    train_loader, valid_loader = checkpoint['dataloaders']
    epoch_r = checkpoint['epoch']

    model = ScopeMultiDNN(*params).to(device)
    model.load_state_dict(model_stat_dict)
    return model, train_loader, valid_loader, epoch_r


def correctCount(output, label):
    pitchLogit, voiceLogit = output
    pitchLogit = pitchLogit.detach()
    voiceLogit = voiceLogit.detach()
    label = label.view(-1)

    pitch = pitchLogit.argmax(1)
    voice = voiceLogit.argmax(1).byte()  # 0-unvoice 1-voice
    pitch.masked_fill_(~voice, -1)
    cc = label.eq(pitch).cpu().numpy().astype(int)
    return cc


def train_epoch(model, train_loader, optimizer, lossFunction):
    model.train()
    total_loss = 0
    stats = [0, 0]  # (correct, total), accuracy=correct/total
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
        cc = correctCount(output, label)
        stats[0] += cc.sum()
        stats[1] += cc.size

    ave_loss = total_loss / len(train_loader)
    accuracy = stats[0] / stats[1]
    return ave_loss, accuracy


def eval_epoch(model, valid_loader, lossFunction, device=DEVICE):
    model.eval()
    total_loss = 0
    stats = [0, 0]  # (correct, total), accuracy=correct/total
    with torch.no_grad():
        for batch in tqdm(valid_loader, mininterval=2, desc='  - (Validation)', leave=False):
            # get data
            feature = batch['features'].to(device)
            label = batch['labels'].to(device)
            # forward
            output = model(feature)
            # calculate loss
            total_loss += lossFunction(output, label)
            # counting
            cc = correctCount(output, label)
            stats[0] += cc.sum()
            stats[1] += cc.size

    ave_loss = total_loss / len(valid_loader)
    accuracy = stats[0] / stats[1]
    return ave_loss, accuracy


def train(model, train_loader, valid_loader, optimizer, lossFunction, epoch_r=0):
    valid_accus = []
    for epoch_i in range(NUM_EPOCH):
        print(f'[ Epoch {epoch_r} + {epoch_i}]')

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, train_loader, optimizer, lossFunction)
        elapse = (time.time() - start) / 60
        print(
            f'  - (Training)  loss:{train_loss:8.5f}, accuracy:{train_accu:3.5f} elapse:{elapse:4.3f}min')

        start = time.time()
        valid_loss, valid_accu = eval_epoch(
            model, valid_loader, lossFunction)
        elapse = (time.time() - start) / 60
        print(
            f'  - (Validation)  loss:{valid_loss:8.5f}, accuracy:{valid_accu:3.5f} elapse:{elapse:4.3f}min')

        valid_accus += [valid_accu]
        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'params': MODEL_PARAMS,
            'dataloaders': (train_loader, valid_loader),
            'epoch': epoch_i + epoch_r,
        }
        if SAVE_DATA is not None:
            if valid_accu >= max(valid_accus):
                torch.save(checkpoint, SAVE_DATA)
        if SAVE_LOG is not None:
            with open(SAVE_LOG, 'a') as logf:
                logf.write(
                    f'{epoch_i+epoch_r},{train_loss:8.5f},{train_accu:3.5f},training\n')
                logf.write(
                    f'{epoch_i+epoch_r},{valid_loss:8.5f},{valid_accu:3.5f},validation\n')


@click.command()
@click.argument('resume', type=click.BOOL)
def main(resume):
    if not resume:
        model, train_loader, valid_loader = init()
        epoch_r = 0
    else:
        model, train_loader, valid_loader, epoch_r = resumeFrom(SAVE_DATA)
        print(f'resume state from {SAVE_DATA} to {DEVICE}, epoch:{epoch_r}')

    optimizer = torch.optim.Adam(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    lossFunction = MultiTaskLoss(AUXILIARY_WEIGHT)

    train(model, train_loader, valid_loader, optimizer, lossFunction, epoch_r)


if __name__ == '__main__':
    main()
