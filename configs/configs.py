import os

# algorithmsWrapper subprocess call log configuration
# stdout, stderr
VERBOSE_OUT = True
VERBOSE_ERR = True

MEL_EXT_HOME = os.environ["HOME"] + '/FDU/MIR'
# dataset location
DATASET_BASE_DIRS = {
    'MedleyDB': f'{MEL_EXT_HOME}/dataset/melodyExtraction/MedleyDB',
    'RWC': f'{MEL_EXT_HOME}/dataset/melodyExtraction/RWC',
    'adc2004': f'{MEL_EXT_HOME}/dataset/melodyExtraction/adc2004_full_set',
    'Orchset': f'{MEL_EXT_HOME}/dataset/melodyExtraction/Orchset',
    'MIREX-05': f'{MEL_EXT_HOME}/dataset/melodyExtraction/MIREX-05',
    'iKala': f'{MEL_EXT_HOME}/dataset/melodyExtraction/iKala'
}


def MEDLEYDB_INIT():
    # set medleydb path for medleydb python package
    os.environ['MEDLEYDB_PATH'] = f'{MEL_EXT_HOME}/dataset/melodyExtraction/MedleyDB'


# algorithm location
ALGO_BASE_DIRS = {
    'TmpDir': '/tmp/Lab_MelExt',
    'SLS': f'{MEL_EXT_HOME}/Projects/separateLeadStereo',
    'MCDNN': f'{MEL_EXT_HOME}/Projects/MelodyExtraction_MCDNN',
    'SFC': f'{MEL_EXT_HOME}/Projects/SourceFilterContoursMelody/src',
    'deepSalience': f'{MEL_EXT_HOME}/Projects/ismir2017-deepsalience/predict',
    'melodicSegnet': f'{MEL_EXT_HOME}/Projects/Melody-extraction-with-melodic-segnet',
    'JDC': f'{MEL_EXT_HOME}/Projects/melodyExtraction_JDC',
    'SFNMF-CRNN': f'{MEL_EXT_HOME}/Projects/ismir2018_dominant_melody_estimation/predict',
    'SemanticSeg': f'{MEL_EXT_HOME}/Projects/Vocal-Melody-Extraction',
    'algoProposed': f'{MEL_EXT_HOME}/MirLabs/Lab_MelExt',
}


def ALGO_WRAPPER_INIT():
    # make temporary directory for algoWrapper
    try:
        os.mkdir(ALGO_BASE_DIRS['TmpDir'])
    except FileExistsError:
        pass


# evaluation settings
EVAL_RESULT_DIR = './data/evalResult'
FORCE_EVAL = False

# visualization settings
PLOTS_DIR = './data/plots'
