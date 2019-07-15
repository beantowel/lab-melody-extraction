#!/bin/sh

WAV="~/Music/Lithium Flower.mp3"
DSTDIR="~/Music/exp/"

ALGO="Proposed"
MEL="$DSTDIR""$ALGO.mel"
cd ~/FDU/MIR/MirLabs/Lab_MelExt/
python ~/FDU/MIR/MirLabs/Lab_MelExt/algorithmsCLI.py "$ALGO" "$WAV" "$MEL"
python ~/FDU/MIR/MirLabs/Lab_MelExt/scripts/melToWave.py "$MEL" "$DSTDIR"
python ~/FDU/MIR/MirLabs/Lab_MelExt/scripts/makeStereo.py "$WAV" "$MEL.wav" "$DSTDIR"
rm "$MEL.wav"
WAV_NAME=${WAV##*/}
STEREO="$DSTDIR"stereo_"${WAV_NAME%.*}"_"$ALGO.mel_"
ffmpeg -i "$STEREO".wav -acodec mp3 "$STEREO".mp3
rm "$STEREO".wav
