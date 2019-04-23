import click
import numpy as np
import pandas as pd

from utils.dataset import MedleyDB_Dataset, Adc2004_Dataset, RWC_PR_Dataset, Orchset_Dataset, Adc2004_vocal_Dataset, MedleyDB_vocal_Dataset
from utils.algorithmsWrapper import melodia, separateLeadStereo,\
    sourceFilterContoursMelody, MCDNN, deepsalience, melodicSegnet, algoFeng, SFNMF_CRNN, JDC, algoCao
from utils.metrics import evalAlgo, Metrics_Saver
from configs.configs import EVAL_RESULT_DIR, FORCE_EVAL

loaders = {
    'adc2004': Adc2004_Dataset(),
    'adc2004_vocal': Adc2004_vocal_Dataset(),
    # 'medleyDB': MedleyDB_Dataset(),
    'medleyDB_vocal': MedleyDB_vocal_Dataset(),
    'Orchset': Orchset_Dataset(),
    # 'RWC_PR': RWC_PR_Dataset(),
}
algos = {
    # 'Salamon': melodia,
    # 'Bittner': deepsalience,
    # 'Hsieh': melodicSegnet,  # lambda x: melodicSegnet(x, None),
    # 'Kum': JDC,
    # 'Feng': algoFeng,
    'Cao': algoCao,
    # 'Durrieu': separateLeadStereo,
    # 'Basaran': SFNMF_CRNN,
    # 'Bosch': sourceFilterContoursMelody,
    # 'Kum': MCDNN,
}


@click.command()
@click.option('--force_eval', default=FORCE_EVAL, type=click.BOOL)
@click.option('--dataset', default=None, type=click.STRING)
def main(force_eval, dataset):
    if dataset is None:
        evalLoader = loaders
    else:
        evalLoader = {dataset: loaders[dataset]}

    for dName, loader in evalLoader.items():
        print(f'loader:{dName}')
        saver = Metrics_Saver(dName)
        # run incremental evaluation by default
        saver.load(EVAL_RESULT_DIR)
        for aName, algo in algos.items():
            # avoid duplicate evaluation
            if (aName not in saver.algoNames) or force_eval:
                if force_eval:
                    saver.removeResult(aName)
                    print(f're-eval algo:{aName}')
                else:
                    print(f'algo:{aName}')
                metrics, titles = evalAlgo(loader, algo)
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
