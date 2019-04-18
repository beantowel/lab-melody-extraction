import mido
import os
import random
import numpy as np


def getPort(idx=0):
    portNames = mido.get_output_names()
    port = portNames[idx]
    print('avaliable ports:', portNames)
    print('select:', port)
    return port


def playMidiFile(filename='01-AchGottundHerr.mid', mid=None):
    if mid == None:
        mid = mido.MidiFile(filename)
    with mido.open_output(getPort()) as port:
        for msg in mid.play():
            print(msg)
            port.send(msg)


def initMessages(channel, program):
    control_values = [
        [121, 0],
        [64, 0],
        [91, 28],
        [10, 51],
        [7, 100]
    ]
    msgs = []
    for control, value in control_values:
        msgs.append(mido.Message('control_change', channel=channel, control=control, value=value))
    msgs.append(mido.Message('program_change', channel=channel, program=program))
    return msgs


def genScores(channel, time):
    tQuarter = 240 #0.3191495
    score = []
    t = 0
    while t < time:
        note = random.randint(60 - 24, 60 + 24)
        tNote =  tQuarter * random.randint(1, 4) # np.random.choice(np.arange(1, 5), p=[0.4, 0.3, 0.2, 0.1])
        tRest =  tQuarter * 0 # np.random.choice(np.arange(0, 5), p=[0.6, 0.2, 0.1, 0.09, 0.01])
        score.append(mido.Message('note_on', channel=channel, note=note, velocity=96, time=tRest))
        score.append(mido.Message('note_off', channel=channel, note=note, velocity=0, time=tNote))
        t += (tRest + tNote)
    return score



def genMidi(time, channel_programs):
    mid = mido.MidiFile()
    metaTrack = mido.MidiTrack()
    metaTrack.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    metaTrack.append(mido.MetaMessage('key_signature', key='C', time=0))
    metaTrack.append(mido.MetaMessage('set_tempo', tempo=638299, time=0))
    metaTrack.append(mido.MetaMessage('track_name', name='generated midi', time=0))
    metaTrack.append(mido.MetaMessage('end_of_track', time=1))
    mid.tracks.append(metaTrack)

    for i, (c, p) in enumerate(channel_programs):
        mid.tracks.append(mido.MidiTrack(initMessages(c, p)))
        mid.tracks[-1].extend(genScores(c, time))
        mid.tracks[-1].name = 'Instrument' + str(i)
    return mid


channel_programs = [[0,0],[1,41],[2,71],[3,72]]
if __name__ == '__main__':
    mid = genMidi(20 * 240, channel_programs)
    playMidiFile(mid=mid)
    playMidiFile()
