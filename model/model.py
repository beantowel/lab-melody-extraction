import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from configs.modelConfigs import DEVICE, AUXILIARY_WEIGHT, LABEL_BLUR_MATRIX, UNVOICE_LABEL_SMOOTH, SEMINOTE_RESOLUTION, USE_MODEL, PRETRAINED_MODEL


class MultiDNN(nn.Module):

    def __init__(
            self,
            d_feature=1025,
            d_hidden=(2048, 2048, 1024),
            n_batch=64,
            n_pitch=120,
            n_voice=2,
            dropout=0.5,
            device=DEVICE):
        super(MultiDNN, self).__init__()

        self.device = device
        self.dnn = nn.Sequential(
            # nn.BatchNorm1d(n_batch),
            nn.Linear(d_feature, d_hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.BatchNorm1d(n_batch),
            nn.Linear(d_hidden[0], d_hidden[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.BatchNorm1d(n_batch),
            nn.Linear(d_hidden[1], d_hidden[2]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pitchLayer = nn.Sequential(
            nn.Linear(d_hidden[2], n_pitch),
            nn.ReLU()
        )
        self.voiceLayer = nn.Sequential(
            nn.Linear(d_hidden[2], n_voice),
            nn.ReLU()
        )

        self.initParams()
        self.to(self.device)

    def forward(self, feature_vec):
        dnnOut = self.dnn(feature_vec)
        pitchLogit = self.pitchLayer(dnnOut)
        voiceLogit = self.voiceLayer(dnnOut)
        return pitchLogit, voiceLogit

    def loadPretrained(self, save_data):
        checkpoint = torch.load(save_data, map_location=self.device)
        model_stat_dict = checkpoint['model']
        self.load_state_dict(model_stat_dict, strict=False)
        for module in (self.dnn, self.pitchLayer, self.voiceLayer):
            for param in module.parameters():
                param.requires_grad = False  # fix parameter

    def initParams(self):
        for layer in self.dnn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
        nn.init.xavier_normal_(self.pitchLayer[0].weight)
        nn.init.xavier_normal_(self.voiceLayer[0].weight)


class MultiDNN_RNN(MultiDNN):

    def __init__(
            self,
            d_feature=1025,
            d_hidden=(2048, 2048, 1024),
            l_seq=16,
            n_rnnLayer=1,
            n_pitch=120,
            n_voice=2,
            dropout=0.5,
            device=DEVICE):
        super(MultiDNN_RNN, self).__init__(
            d_feature,
            d_hidden,
            l_seq,
            n_pitch,
            n_voice,
            dropout,
            device)
        self.d_feature = d_feature
        self.l_seq = l_seq
        self.n_rnnLayer = n_rnnLayer

        self.pitchLSTM = nn.LSTM(
            input_size=d_hidden[2],
            hidden_size=d_hidden[2] // 2,
            num_layers=self.n_rnnLayer,
            batch_first=True,
            dropout=dropout,
            bidirectional=True)
        self.voiceLSTM = nn.LSTM(
            input_size=d_hidden[2],
            hidden_size=d_hidden[2] // 2,
            num_layers=self.n_rnnLayer,
            batch_first=True,
            dropout=dropout,
            bidirectional=True)

        self.loadPretrained(PRETRAINED_MODEL)
        self.to(self.device)

    def forward(self, feature_vec):
        # feature_vec:(nBatch, l_seq, d_feature)
        dnnOut = self.dnn(feature_vec)
        pitchRnnOut, _ = self.pitchLSTM(dnnOut)
        voiceRnnOut, _ = self.voiceLSTM(dnnOut)
        pitchLogit = self.pitchLayer(pitchRnnOut)
        voiceLogit = self.voiceLayer(voiceRnnOut)
        return pitchLogit, voiceLogit


def multiTaskLoss(output, label):
    # label(nBatch, len_seq)
    # pitch(nBatch, len_seq, nPitchClass) -> (x, nPitchClass)
    # voice(nBatch, len_seq, nVoiceClass) -> (y, nVoiceClass)
    pitchLogit, voiceLogit = output
    nPc, nVc = pitchLogit.shape[-1], voiceLogit.shape[-1]
    voiceMask = (label >= 0).byte()

    # make target with label blurring
    # blurredLabel(nVoicingFrame, nPitchClass)
    blurOnehot = F.one_hot(label.abs(), num_classes=nPc).type(torch.float)
    blurOnehot = torch.matmul(blurOnehot, LABEL_BLUR_MATRIX)
    blurOnehot[label < 0, :] = UNVOICE_LABEL_SMOOTH
    pitchLoss = -(blurOnehot * F.log_softmax(pitchLogit, dim=-1)).sum(dim=-1)
    # pitchLoss = pitchLoss.masked_select(voiceMask).mean()
    # pitchLoss = pitchLoss if not torch.isnan(pitchLoss) else 0
    pitchLoss = pitchLoss.mean()

    voiceLoss = F.cross_entropy(
        voiceLogit.view(-1, nVc), voiceMask.view(-1).long()) * SEMINOTE_RESOLUTION
    loss = pitchLoss + AUXILIARY_WEIGHT * voiceLoss
    return loss


class AccuracyCounter(object):

    def __init__(self):
        # (correct, total), accuracy=correct/total
        self.paCount = np.array([0, 0])
        self.vaCount = np.array([0, 0])
        self.oaCount = np.array([0, 0])

    def update(self, output, label):
        pitchLogit, voiceLogit = output
        nPc, nVc = pitchLogit.shape[-1], voiceLogit.shape[-1]
        pitchLogit = pitchLogit.detach().view(-1, nPc)
        voiceLogit = voiceLogit.detach().view(-1, nVc)
        label = label.view(-1)

        pitch = pitchLogit.argmax(1)
        voice = voiceLogit.argmax(1).byte()  # 0-unvoice 1-voice

        # negative value small enough for correct pitch accuracy counting
        pitch.masked_fill_(~voice, -SEMINOTE_RESOLUTION)
        oaCount = torch.abs(label-pitch) < (SEMINOTE_RESOLUTION / 2)
        oaCount[label < 0] = 1
        paCount = torch.abs(label[label > 0] -
                            pitch[label > 0]) < (SEMINOTE_RESOLUTION / 2)
        paCount = paCount.cpu().numpy().astype(int)
        label[label >= 0] = 1
        label[label < 0] = 0
        vaCount = torch.eq(label, voice.long())
        oaCount = (oaCount & vaCount).cpu().numpy().astype(int)
        vaCount = vaCount.cpu().numpy().astype(int)

        self.paCount += np.array([paCount.sum(), paCount.size])
        self.vaCount += np.array([vaCount.sum(), vaCount.size])
        self.oaCount += np.array([oaCount.sum(), oaCount.size])

    def getAccuacy(self):
        pa = self.paCount[0] / self.paCount[1]
        va = self.vaCount[0] / self.vaCount[1]
        oa = self.oaCount[0] / self.oaCount[1]
        return pa, va, oa


if USE_MODEL == 'MultiDNN':
    MODEL_CLASS = MultiDNN
elif USE_MODEL == 'MultiDNN_RNN':
    MODEL_CLASS = MultiDNN_RNN
else:
    raise(ValueError(USE_MODEL))
