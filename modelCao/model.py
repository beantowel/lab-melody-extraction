import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from configs.modelCaoConfigs import DEVICE
from configs.modelFengConfigs import SAVE_DATA as Feng_SAVE_DATA


class ScopeMultiDNN(nn.Module):

    def __init__(self, d_feature=1025, d_hidden=(2048, 2048, 1024), n_stage=8, n_pitch=120, n_voice=2, dropout=0.5, device=DEVICE, pretrained=Feng_SAVE_DATA):
        super(ScopeMultiDNN, self).__init__()
        self.d_feature = d_feature
        self.d_hidden = d_hidden
        self.n_stage = n_stage
        self.n_pitch = n_pitch
        self.n_voice = n_voice
        self.device = device

        self.scope = 1 << n_stage
        # bit-reversal permutation
        self.permutation = torch.LongTensor([
            int('{:0{width}b}'.format(i, width=n_stage)[::-1], 2)
            for i in range(self.scope)
        ]).to(device)

        self.dnn = nn.Sequential(
            nn.Linear(d_feature, d_hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden[0], d_hidden[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden[1], d_hidden[2]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pitchLayer = nn.Sequential(
            nn.Linear(d_hidden[-1], n_pitch),
            nn.ReLU()
        )
        self.voiceLayer = nn.Sequential(
            nn.Linear(d_hidden[-1], n_voice),
            nn.ReLU()
        )

        self.butterflyArithm = nn.Sequential(
            nn.Linear(n_pitch * 2, n_pitch * 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_pitch * 2, n_pitch * 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

        self.loadPretrained(pretrained)
        self.to(self.device)

    def forward(self, feature_vec):
        # feature_vec: (nBatch, scope, d_feature)
        assert feature_vec.shape[1] == self.scope, f'{feature_vec.shape} mismatch {self.n_stage}'
        # dnnIn: (nBatch*scope, d_feature)
        dnnIn = feature_vec.view(-1, self.d_feature)
        dnnOut = self.dnn(dnnIn)
        pitchLogit = self.pitchLayer(dnnOut)
        voiceLogit = self.voiceLayer(dnnOut)

        transIn = pitchLogit.view(-1, self.scope, self.n_pitch)
        transIn = transIn[:, self.permutation]
        tPitchLogit = self.butterflyTransform(transIn)
        tPitchLogit = tPitchLogit[:, self.permutation].view(-1, self.n_pitch)
        return tPitchLogit, voiceLogit

    def loadPretrained(self, save_data):
        checkpoint = torch.load(save_data, map_location=self.device)
        model_stat_dict = checkpoint['model']
        self.load_state_dict(model_stat_dict, strict=False)
        for module in (self.dnn, self.pitchLayer, self.voiceLayer):
            for param in module.parameters():
                param.requires_grad = False

    def butterflyOp(self, ax, ay, k, i):
        # rootN = 1 << (i+1)
        # theta = 2 * k / rootN * np.pi
        # ax, ay: (nBatch, x)
        # out: (nBatch, x * 2)
        dim1 = ax.shape[1]
        opIn = torch.stack((ax, ay), dim=-1).view(-1, dim1 * 2)
        opOut = self.butterflyArithm(opIn)
        return opOut[:, :dim1], opOut[:, dim1:]

    def butterflyTransform(self, a):
        for i in range(self.n_stage):
            stride = 1 << i
            groups = 1 << (self.n_stage - i - 1)
            for j in range(groups):
                for k in range(stride):
                    x = j * 2 * stride + k
                    y = x + stride
                    a[:, x], a[:, y] = self.butterflyOp(a[:, x], a[:, y], k, i)
        return a


def MultiTaskLoss(auxiliary_weight=0.5):
    def lossFunction(output, label):
        # (nBatch, 1<<nStage)
        label = label.view(-1)
        pitchLogit, voiceLogit = output
        voiceLabel = (label > 0).to(torch.long)

        pitchLoss = F.cross_entropy(pitchLogit, label, ignore_index=-1)
        voiceLoss = F.cross_entropy(voiceLogit, voiceLabel)
        loss = pitchLoss + auxiliary_weight * voiceLoss
        return loss
    return lossFunction
