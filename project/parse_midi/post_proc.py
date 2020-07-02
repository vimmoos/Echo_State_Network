import csv
import enum as e
import subprocess as sp

import numpy as np

# import project.esn.test as tesn
import project.music_gen.data_types as dt

# from pprint import pprint

net_out = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 1., 0., 1., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 1., 0., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 1., 0., 1., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 1., 0., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 1., 0., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 1., 0., 1., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 1., 0., 0., 0., 0., 0.],
                    [0., 1., 0., 1., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 1., 0., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0.]])


class Note_value(e.Enum):
    HI_HAT_CLOSE = 42
    HI_HAT_OPEN = 46
    BASS_DRUM = 35
    CRASH = 49
    SNARE = 38
    HIG_TOM = 50
    MID_TOM = 48
    FLOOR_TOM = 43
    RIDE = 51


abs_to_midi_note = {
    k: v
    for k, v in zip(range(len(dt.Abs_note)), [ent.value for ent in Note_value])
}


def csv_header(clock_cycle, channel, offset, tempo):
    return [["0", "0", "Header", "0", "1", clock_cycle],
            ["1", "0", "Start_track"], ["1", "0", "Channel_prefix", channel],
            ["1", "0", "SMPTE_offset", offset, "0", "0", "0", "0"],
            ["1", "0", "Tempo", tempo]]


def csv_footer(time_stamp, track_n=1):
    return [[str(track_n), str(time_stamp), "End_track"],
            ["0", "0", "End_of_file"]]


def create_csv_line(
    note,
    clock_time,
    event="Note_on_c",
    track_n=1,
    channel=9,  # channel 9 is reserved for drums
    velocity=100,
):
    return [
        str(x) for x in [track_n, clock_time, event, channel, note, velocity]
    ]


def row_to_csv(row,
               time_stamp: int,
               t_offset: int,
               pred=lambda x: x > 0) -> tuple:
    notes_hit = [idx for idx, val in enumerate(row) if pred(val)]
    time_stamp += t_offset

    if not notes_hit:
        return None, time_stamp

    notes_conv = [abs_to_midi_note[note_idx] for note_idx in notes_hit]

    return (
        [create_csv_line(note, time_stamp - t_offset)
         for note in notes_conv] + [
             create_csv_line(note, time_stamp, event="Note_off_c", velocity=0)
             for note in notes_conv
         ], time_stamp)


def output_to_csv(output_matrix, t_offset: int = 120):
    time_stamp = 0
    csv_f = []

    for r in output_matrix:
        csv_lines, time_stamp = row_to_csv(r, time_stamp, t_offset)
        if csv_lines:
            csv_f += csv_lines

    return csv_f, time_stamp


def final_to_csv(filename: str):
    csv_body, time_stamp = output_to_csv(net_out)
    with open(filename, 'w', newline='') as file_:
        writer = csv.writer(file_)
        writer.writerows(
            csv_header(480, 9, 33, 500000) + csv_body + csv_footer(time_stamp))


def csv2midi(filename: str, listen=False):
    fnanme_csv, fname_midi = filename + ".csv", filename + ".midi"
    final_to_csv(fnanme_csv)
    sp.run(["csvmidi", fnanme_csv, fname_midi])
    if listen:
        sp.run(["xdg-open", fname_midi])
