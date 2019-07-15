from mir_eval.melody import evaluate
import matplotlib.pyplot as plt
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')


def evalAlgo(loader, algo):
    titles = []
    metrics = []
    for i, data in enumerate(loader):
        print(f'[{i + 1}/{len(loader)}] eval:{data["title"]}')
        est = algo(data['wav'])
        metrics.append(getMetric(data['gt'], est))
        titles.append(data['title'])
    return np.array(metrics), titles


def getMetric(ref, est):
    ref_time, ref_freq = ref
    est_time, est_freq = est

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = evaluate(ref_time, ref_freq, est_time, est_freq)
    VR, VFA, RPA, RCA, OA = scores['Voicing Recall'], scores['Voicing False Alarm'], scores[
        'Raw Pitch Accuracy'], scores['Raw Chroma Accuracy'], scores['Overall Accuracy']
    return VR, VFA, RPA, RCA, OA


class Metrics_Saver():
    def __init__(self, datasetName):
        self.datasetName = datasetName
        self.algoNames = []
        self.metricsList = []
        self.titlesList = []

    def addResult(self, algoName, metrics, titles):
        self.algoNames.append(algoName)
        self.metricsList.append(metrics)
        self.titlesList.append(titles)

    def removeResult(self, algoName):
        try:
            # remove all match algoName result
            while True:
                idx = self.algoNames.index(algoName)
                self.algoNames.pop(idx)
                self.metricsList.pop(idx)
                self.titlesList.pop(idx)
        except ValueError:
            print(f'all {algoName} result removed')

    def reWriteResult(self, algoName, metrics, titles):
        assert algoName in self.algoNames, f'{algoName} not in {self.algoNames}'
        idx = self.algoNames.index(algoName)
        self.metricsList[idx] = metrics
        self.titlesList[idx] = titles

    def writeFullResults(self, dirname):
        fullOutputFile = os.path.join(dirname, f'{self.datasetName}_full.csv')
        cols = ['title', 'author', 'VR', 'VFA', 'RPA', 'RCA', 'OA']
        df = pd.DataFrame(columns=cols)
        for algoName, metrics, titles in zip(self.algoNames, self.metricsList, self.titlesList):
            n = len(titles)
            head = np.array([titles, [algoName] * n]).T
            headDf = pd.DataFrame(data=head, columns=cols[:2])
            metricDf = pd.DataFrame(data=metrics, columns=cols[2:])
            algoDf = pd.concat([headDf, metricDf], axis=1)
            df = pd.concat([df, algoDf], ignore_index=True)
        df.to_csv(fullOutputFile)
        print(f'results written to {fullOutputFile}')

    def writeAveResults(self, dirname):
        aveOutputFile = os.path.join(dirname, f'{self.datasetName}.csv')
        columns = ['author', 'VR', 'VFA', 'RPA', 'RCA', 'OA']
        df = pd.DataFrame(columns=columns)
        for algoName, metrics in zip(self.algoNames, self.metricsList):
            data = np.hstack(
                [algoName, np.mean(metrics, axis=0)]).reshape(1, 6)
            df = pd.concat([df, pd.DataFrame(data=data, columns=columns)])
        df.to_csv(aveOutputFile)
        print(f'results written to {aveOutputFile}')

    def saveViolinPlot(self, dirname, order=None):
        pltOutputFile = os.path.join(dirname, f'{self.datasetName}.svg')
        axisNames = ['VR', 'VFA', 'RPA', 'RCA', 'OA']
        if order is not None:
            algoNames = list(filter(lambda x: x in self.algoNames, order))
            metricsList = [
                self.metricsList[self.algoNames.index(aName)] for aName in algoNames]
        else:
            algoNames = self.algoNames
            metricsList = self.metricsList

        pos = np.arange(len(algoNames), dtype=int) + 1
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
        fig.delaxes(axes[1, 2])

        for i, axis in enumerate(axes.flatten()[:5]):
            data = [metrics[:, i] for metrics in metricsList]
            axis.violinplot(data, pos, showmeans=True, showextrema=True)
            axis.set_title(axisNames[i])
            plt.setp(axis.get_xticklabels(), rotation=45)

        plt.suptitle(self.datasetName, fontsize=20)
        plt.setp(axes, xticks=pos, xticklabels=algoNames)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(pltOutputFile, quality=100)
        print(f'violin plot written to {pltOutputFile}')

    def dump(self, dirname):
        dumpFile = os.path.join(dirname, f'{self.datasetName}.pkl')
        with open(dumpFile, 'wb') as f:
            pickle.dump((self.datasetName, self.algoNames,
                         self.metricsList, self.titlesList), f, pickle.HIGHEST_PROTOCOL)
        print(f'saver object written to {dumpFile}')

    def load(self, dirname):
        dumpFile = os.path.join(dirname, f'{self.datasetName}.pkl')
        try:
            with open(dumpFile, 'rb') as f:
                self.datasetName, self.algoNames, self.metricsList, self.titlesList = pickle.load(
                    f)
            print(f'saver object loaded from {dumpFile}')
        except FileNotFoundError:
            print(f'saver object {dumpFile} not found, set to empty')
