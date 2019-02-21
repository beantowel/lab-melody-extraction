import numpy as np
import pandas as pd

from utils.dataLoader import MedleyDB_Loader, Adc2004_Loader, RWC_Loader
from utils.algorithmsWrapper import melodia, seprateLeadStereo,\
    sourceFilterContoursMelody, MCDNN, deepsalience
from utils.metrics import getMetric


def evalAlgo(loader, algo):
    metrics = []
    for data in loader:
        print(f'eval:{data["title"]}')
        est = algo(data['wav'])
        metrics.append(getMetric(data['gt'], est))
    metrics = np.array(metrics)
    return metrics


loaders = {
    'adc2004': Adc2004_Loader(),
    'medleyDB': MedleyDB_Loader(),
    'RWC': RWC_Loader(),
}
algos = {
    'Salamon': melodia,
    'Durrieu': seprateLeadStereo,
    # 'Bosch': sourceFilterContoursMelody,
    # 'Kum': MCDNN,
    'Bittner': deepsalience,
}

for dName, loader in loaders.items():
    print(f'loader:{dName}')
    outputFile = f'data/{dName}.csv'
    rows = []
    for aName, algo in algos.items():
        print(f'algo:{aName}')
        metrics = evalAlgo(loader, algo)
        metrics = np.mean(metrics, axis=0)
        row = {}
        row['author'] = aName
        row['VR'], row['VFA'], row['RPA'], row['RCA'], row['OA'] = metrics
        rows.append(row)
        print(f'metric:{row}')
    df = pd.DataFrame(rows)
    df.to_csv(outputFile)
