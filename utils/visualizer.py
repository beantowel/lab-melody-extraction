import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from configs.configs import PLOTS_DIR
# from configs.modelFengConfigs import SAVE_LOG, MODEL_NAME
from configs.modelCaoConfigs import SAVE_LOG, MODEL_NAME


def logPlot(mode, logfile, model, savedir):
    # 'epoch,loss,accuracy,mode\n'
    df = pd.read_csv(logfile)
    mask = (df['mode'] == mode)
    # remove duplicate epoch cause of non-latest chekpoint
    # keep epoch in ascending order
    r_epoch = df['epoch'][mask]
    for i, m in enumerate(mask):
        if m and (r_epoch.loc[i] > min(r_epoch.loc[i:])):
            mask[i] = False

    epoch = df['epoch'][mask]
    loss, accuracy = df['loss'][mask], df['accuracy'][mask]

    fig, ax1 = plt.subplots()
    plt.title(mode)

    color = 'tab:red'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss', color=color)
    ax1.plot(epoch, loss, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    # we already handled the x-label with ax1
    ax2.set_ylabel('accuracy', color=color)
    ax2.plot(epoch, accuracy, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    savefile = os.path.join(savedir, f'{model}_{mode}.svg')
    plt.savefig(savefile)
    return plt


def show(logfile=SAVE_LOG, model=MODEL_NAME, savedir=PLOTS_DIR):
    plt = logPlot('training', logfile, model, savedir)
    plt.show()
    plt = logPlot('validation', logfile, model, savedir)
    plt.show()
