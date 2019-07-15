#!~/anaconda3/bin python

import os
import click
import librosa


@click.command()
@click.argument('src', nargs=1, type=click.Path(exists=True))
@click.argument('sr', nargs=1, type=click.INT)
@click.argument('dst', nargs=1, type=click.Path())
def transFiles(src, sr, dst):
    head, tail = os.path.split(src)
    output = os.path.join(dst, tail + f'_{sr//1000}K.wav')

    y, sr = librosa.load(src, sr=sr)
    wav = y[::-1], sr
    click.echo('write to: ' + output)
    librosa.output.write_wav(output, *wav)


script_dir = os.path.dirname(__file__)
if __name__ == "__main__":
    transFiles()
