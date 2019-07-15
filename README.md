# Automatic Melody Extraction

snippets for [MIREX AME task](https://www.music-ir.org/mirex/wiki/2018:Audio_Melody_Extraction)

## Evaluation on state-of-the arts results

### Algorithms

- Salamon, [melodia](https://github.com/justinsalamon/audio_to_midi_melodia)
- Durrieu, [separateLeadStereo](https://github.com/HENDRIX-ZT2/separateLeadStereo)
- ~~Bosch, [SourceFilterContoursMelody](https://github.com/juanjobosch/SourceFilterContoursMelody)~~
- ~~Kum, [MelodyExtraction_MCDNN](https://github.com/keums/MelodyExtraction_MCDNN)~~
- Bittner, [ismir2017-deepsalience](https://github.com/rabitt/ismir2017-deepsalience)
- Hsieh, [Melody-extraction-with-melodic-segnet](https://github.com/bill317996/Melody-extraction-with-melodic-segnet)
- Kum, [melodyExtraction_JDC](https://github.com/keums/melodyExtraction_JDC)
- Basaran, [ismir2018_dominant_melody_estimation](https://github.com/dogacbasaran/ismir2018_dominant_melody_estimation)
- Lu, [Vocal-Melody-Extraction](https://github.com/s603122001/Vocal-Melody-Extraction)

### Datasets

- MedleyDB (vocal/instrumental)
- RWC Popular, RWC Royalty free
- ADC 2004 (vocal/instrumental)
- Orchset (instrumental)
- Mirex-05 (vocal/instrumental)
- iKala

## Requirements

Except for the requirements needed by the algorithms repos above, the code use Python 3.6 or higher version, and requirements is listed in `requirements.txt`. You could install the requirements with command `pip install -r requirements.txt`.

## Configuration

Since this repo does not include any dataset files and algorithms repos, you should have download them independently and configure the path to the datasets and algorithms in the `configs/configs.py`.

## Usage

### train.py

train the proposed model.

```sh
Usage: train.py [OPTIONS]

Options:
  --resume BOOLEAN  resume training state from .pkl file
  --debug BOOLEAN   use small dataset to debug faster
  --help            Show this message and exit.
```

### predict.py

inference using the trained model. Pretrained models can be downloaded from [google drive](https://drive.google.com/open?id=1TScpMr2sTsushiqB_3k0ILySLGzJSN0W), download and put them in the `data/` folder.

```sh
Usage: predict.py [OPTIONS] AUDIOFILE [MELFILE]

Options:
  --model TEXT   pretrained model path
  --cpu BOOLEAN  use when have no cuda support
  --help         Show this message and exit.
```

example: `python predict.py <path-to-input-audio-file> <path-of-output-csv-file>`

### evalAlgos.py

evaluate state-of-the-art algorithms on various datasets.

```sh
Usage: evalAlgos.py [OPTIONS]

Options:
  --force BOOLEAN   overwrite evaluation results
  --dataset TEXT    using specific dataset
  --algorithm TEXT  using specific algorithm
  --help            Show this message and exit.
```

### algorithmsCLI.py

a convenient command line interface for calling algorithms.

```sh
Usage: algorithmsCLI.py [OPTIONS] ALGO AUDIOFILE [MELFILE]

Options:
  --help  Show this message and exit.
```
