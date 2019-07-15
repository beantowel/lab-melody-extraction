import numpy as np

from utils import metrics
from utils.preprocessor import EmphasisF
from configs.configs import EVAL_RESULT_DIR


def writeMel(output, times, freqs):
    # write melody
    np.savetxt(output, np.array([times, freqs]).T,
               fmt=['%.3f', '%.4f'], delimiter='\t')


def editViolinPlot(
        dnames=['adc2004_vocal', 'mirex05_vocal'],
        algoNames=['Salamon', 'Bittner', 'Hsieh', 'Lu', 'MultiDNN', 'MultiDNN_RNN']):
    for key in dnames:
        saver = metrics.Metrics_Saver(key)
        saver.load(EVAL_RESULT_DIR)
        saver.saveViolinPlot(EVAL_RESULT_DIR, order=algoNames)
