#!~/anaconda3/bin python

import os
import click
from midi2audio import FluidSynth


@click.command()
@click.argument('src', nargs=-1, type=click.Path(exists=True))
@click.argument('dst', nargs=1, type=click.Path())
def transFiles(src, dst):
    for filename in src:
        head, tail = os.path.split(filename)
        output = os.path.join(dst, tail + '.wav')

        click.echo('transform: ' + filename)
        click.echo('write to: ' + output)
        fs.midi_to_audio(filename, output)


fs = FluidSynth('/usr/share/sounds/sf2/FluidR3_GM.sf2')
script_dir = os.path.dirname(__file__)
if __name__ == "__main__":
    transFiles()
