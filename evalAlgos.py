import click
import numpy as np
import pandas as pd

from utils.dataset import *
from utils.algorithmsWrapper import *
from utils.preprocessor import EmphasisF
from utils.metrics import evalAlgo, Metrics_Saver
from configs.configs import EVAL_RESULT_DIR, FORCE_EVAL
from configs.modelConfigs import MODEL_NAME, TRAIN_RATIO

loaders = {
    # 'adc2004_vocal_emp': Adc2004_vocal_Dataset(transform=EmphasisF()),
    'adc2004_vocal': Adc2004_vocal_Dataset(),
    'adc2004_instrumental': Adc2004_instrumental_Dataset(),
    'medleyDB_vocal': MedleyDB_vocal_Dataset(),
    'medleyDB_instrumental': MedleyDB_instrumental_Dataset(),
    'Orchset': Orchset_Dataset(),
    'RWC_Popular': RWC_Popular_Dataset(),
    'RWC_Royalty_Free': RWC_Royalty_Free_Dataset(),
    'mirex05_vocal': MIREX_05_vocal_Dataset(),
    'mirex05_instrumental': MIREX_05_instrumental_Dataset(),
    'iKala': IKala_Dataset(),
}
algos = {
    'Salamon': melodia,
    'Bittner': deepsalience,
    'Hsieh': lambda x: melodicSegnet(x, mode='std'),
    'Kum': JDC,
    'Lu': SemanticSeg,
    f'{MODEL_NAME}': algoProposed,
    # 'PeakSel': algoPeakSel,
    # 'Durrieu': separateLeadStereo,
    # 'Basaran': SFNMF_CRNN,
    # 'Bosch': sourceFilterContoursMelody,
    # 'Kum': MCDNN,
}


@click.command()
@click.option('--force', default=FORCE_EVAL, type=click.BOOL, help='overwrite evaluation results')
@click.option('--dataset', default=None, type=click.STRING, help='using specific dataset')
@click.option('--algorithm', default=None, type=click.STRING, help='using specific algorithm')
def main(force, dataset, algorithm):
    if dataset is None:
        evalLoader = loaders
    else:
        evalLoader = {dataset: loaders[dataset]}
    if algorithm is None:
        evalAlgos = algos
    else:
        evalAlgos = {algorithm: algos[algorithm]}

    for dName, loader in evalLoader.items():
        print(f'loader:{dName}')
        saver = Metrics_Saver(dName)
        # run incremental evaluation by default
        saver.load(EVAL_RESULT_DIR)
        for aName, algo in evalAlgos.items():
            # avoid duplicate evaluation
            if (aName not in saver.algoNames) or force:
                if force:
                    print(f're-eval algo:{aName}')
                else:
                    print(f'algo:{aName}')

                metrics, titles = evalAlgo(loader, algo)
                print(f'{aName} average result:')
                print(['VR', 'VFA', 'RPA', 'RCA', 'OA'])
                print(np.mean(metrics, axis=0))

                if force and (aName in saver.algoNames):
                    saver.reWriteResult(aName, metrics, titles)
                else:
                    saver.addResult(aName, metrics, titles)
                # save result every iter
                saver.dump(EVAL_RESULT_DIR)
            else:
                print(f'! skipping algo:{aName}')
        saver.writeFullResults(EVAL_RESULT_DIR)
        saver.writeAveResults(EVAL_RESULT_DIR)
        saver.saveViolinPlot(EVAL_RESULT_DIR)


if __name__ == '__main__':
    main()
