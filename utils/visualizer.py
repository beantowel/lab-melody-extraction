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
    epoch = df['epoch']
    loss, accuracy = df['loss'], df['accuracy']
    mask = (df['mode'] == mode)

    fig, ax1 = plt.subplots()
    plt.title(mode)

    color = 'tab:red'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss', color=color)
    ax1.plot(epoch[mask], loss[mask], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    # we already handled the x-label with ax1
    ax2.set_ylabel('accuracy', color=color)
    ax2.plot(epoch[mask], accuracy[mask], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    savefile = os.path.join(savedir, f'{model}_{mode}.svg')
    plt.savefig(savefile)
    return plt


def main(logfile=SAVE_LOG, model=MODEL_NAME, savedir=PLOTS_DIR):
    plt = logPlot('training', logfile, model, savedir)
    plt.show()
    plt = logPlot('validation', logfile, model, savedir)
    plt.show()


if __name__ == '__main__':
    main()
