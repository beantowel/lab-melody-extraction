import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiDNN(nn.Module):

    def __init__(self, d_feature=1025, d_hidden=(2048, 2048, 1024), n_pitch=120, n_voice=2, dropout=0.5):
        super(MultiDNN, self).__init__()

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
            nn.Linear(d_hidden[2], n_pitch),
            nn.ReLU()
        )
        self.voiceLayer = nn.Sequential(
            nn.Linear(d_hidden[2], n_voice),
            nn.ReLU()
        )

        for i in range(0, 9, 3):
            nn.init.xavier_normal_(self.dnn[i].weight)
        nn.init.xavier_normal_(self.pitchLayer[0].weight)
        nn.init.xavier_normal_(self.voiceLayer[0].weight)

    def forward(self, feature_vec):
        dnnOut = self.dnn(feature_vec)
        pitchLogit = self.pitchLayer(dnnOut)
        voiceLogit = self.voiceLayer(dnnOut)
        return pitchLogit, voiceLogit


def MultiTaskLoss(auxiliary_weight=0.5):
    def lossFunction(output, label):
        pitchLogit, voiceLogit = output
        voiceLabel = (label > 0).to(torch.long)

        pitchLoss = F.cross_entropy(pitchLogit, label, ignore_index=-1)
        voiceLoss = F.cross_entropy(voiceLogit, voiceLabel)
        loss = pitchLoss + auxiliary_weight * voiceLoss
        return loss
    return lossFunction
