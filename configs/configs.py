import os

# algorithmsWrapper subprocess call log configuration
# stdout, stderr
VERBOSE_OUT = True
VERBOSE_ERR = True

# dataset location
DATASET_BASE_DIRS = {
    'MedleyDB': '/home/beantowel/FDU/MIR/dataset/melodyExtraction/MedleyDB',
    'RWC': '/home/beantowel/FDU/MIR/dataset/melodyExtraction/RWC',
    'adc2004': '/home/beantowel/FDU/MIR/dataset/melodyExtraction/adc2004_full_set',
    'Orchset': '/home/beantowel/FDU/MIR/dataset/melodyExtraction/Orchset',
}


def MEDLEYDB_INIT():
    # set medleydb path for medleydb python package
    os.environ['MEDLEYDB_PATH'] = '/home/beantowel/FDU/MIR/dataset/melodyExtraction/MedleyDB'


# algorithm location
ALGO_BASE_DIRS = {
    'TmpDir': '/tmp/Lab_MelExt',
    'SLS': '/home/beantowel/FDU/MIR/Projects/separateLeadStereo',
    'MCDNN': '/home/beantowel/FDU/MIR/Projects/MelodyExtraction_MCDNN',
    'SFC': '/home/beantowel/FDU/MIR/Projects/SourceFilterContoursMelody/src',
    'deepSalience': '/home/beantowel/FDU/MIR/Projects/ismir2017-deepsalience/predict',
    'melodicSegnet': '/home/beantowel/FDU/MIR/Projects/Melody-extraction-with-melodic-segnet',
    'JDC': '/home/beantowel/FDU/MIR/Projects/melodyExtraction_JDC',
    'SFNMF-CRNN': '/home/beantowel/FDU/MIR/Projects/ismir2018_dominant_melody_estimation/predict',
    'algoFeng': '/home/beantowel/FDU/MIR/MirLabs/Lab_MelExt',
    'algoCao': '/home/beantowel/FDU/MIR/MirLabs/Lab_MelExt',
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
