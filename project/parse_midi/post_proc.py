import csv
import functools as ft
import subprocess as sp
from dataclasses import dataclass
from enum import Enum

import numpy as np

import project.music_gen.data_types as dt
#import project.test.esn_t as tesn
import project.test.music_test as mt

#tot_out = tesn.test_generated()

#net_out, tempo = tot_out["desired"], tot_out["tempo"]


class Note_MIDI(Enum):
    HI_HAT_CLOSE = 42
    HI_HAT_OPEN = 46
    BASS_DRUM = 35
    CRASH = 49
    SNARE = 38
    HIG_TOM = 50
    MID_TOM = 48
    FLOOR_TOM = 43
    RIDE = 51


enum_values = lambda t: [ent.value for ent in t]

dict_from_iterables = lambda it0, it1: {k: v for k, v in zip(it0, it1)}

midi_tempo = lambda bpm: 6 * 10**7 // bpm

csv_header = lambda clock_cycle, channel, offset, tempo: (
    [["0", "0", "Header", "0", "1", clock_cycle], ["1", "0", "Start_track"],
     ["1", "0", "Channel_prefix", channel],
     ["1", "0", "SMPTE_offset", offset, "0", "0", "0", "0"],
     ["1", "0", "Tempo", tempo]])

csv_footer = lambda time_stamp, track_n=1: (
    [[str(track_n), str(time_stamp), "End_track"], ["0", "0", "End_of_file"]])

# NOTE channel 9 is reserved for drums in MIDI
create_csv_line = lambda note, clock_time, event="Note_on_c", track_n=1, channel=9, velocity=100: (
    [str(x) for x in [track_n, clock_time, event, channel, note, velocity]])


@dataclass
class CSV_Out:
    net_out: np.ndarray
    filename: str
    t_offset: int = 120
    midi_note: dict = None
    time_stamp: int = 0
    bpm: int = 120

    def __post_init__(self):
        if not self.midi_note:
            self.midi_note = dict_from_iterables(enum_values(dt.Abs_note),
                                                 enum_values(Note_MIDI))

    def __call__(self, clock_cycle=480, channel=9, offset=33, tempo=None):
        return self.net_to_csv(clock_cycle, channel, offset,
                               midi_tempo(self.bpm if not tempo else tempo))

    def row_to_csv(self, row, pred=lambda x: x > 0) -> list:
        notes_hit = [idx for idx, val in enumerate(row) if pred(val)]
        self.time_stamp += self.t_offset

        if not notes_hit:
            return None

        notes_converted = [self.midi_note[note_idx] for note_idx in notes_hit]

        return ([
            create_csv_line(note, self.time_stamp - self.t_offset)
            for note in notes_converted
        ] + [
            create_csv_line(
                note, self.time_stamp, event="Note_off_c", velocity=0)
            for note in notes_converted
        ])

    def body_to_csv(self) -> list:
        return list(
            ft.reduce(
                lambda x, y: x + y,
                [ret for r in self.net_out if (ret := self.row_to_csv(r))]))

    def net_to_csv(self, *args, **kwargs) -> str:
        with open(self.filename + ".csv", 'w', newline='') as file_:
            writer = csv.writer(file_)
            writer.writerows(
                csv_header(*args, **kwargs) + self.body_to_csv() +
                csv_footer(self.time_stamp))
        return self.filename + ".csv"


# Other parameters of the dataclass are not passed atm...
def net2midi(net_out: np.ndarray, filename: str, bpm: int = 120, listen=False):
    fname_midi = filename + ".midi"
    csv_out = CSV_Out(net_out, filename, bpm)
    fname_csv = csv_out()
    sp.run(["csvmidi", fname_csv, fname_midi])
    if listen:
        sp.run(["xdg-open", fname_midi])


# pad_measures = lambda dividend, divisor: (dividend if not (
#     rem := dividend % divisor) else dividend - rem, dividend // divisor)

# quarter_offset = lambda d, **kwargs: ({
#     k: pad_measures(v, k) if k != 4 else v
#     for k, v in d.items()
# }) if not kwargs else ({
#     k: (k * new_v, new_v)
#     if (new_v := kwargs.get(str(k), None)) and k != 4 else pad_measures(v, k)
#     if k != 4 else v
#     for k, v in d.items()
# })

# measure_offset = lambda step, base=0, lim=300: {
#     k: v
#     for k, v in zip([2**i for i in range(2, 6)],
#                     [base + j for j in range(0, lim, step)])
# }

if __name__ == "__main__":
    net2midi(mt.new, "new", bpm= 170)