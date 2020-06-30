import csv
import enum as e
from pprint import pprint

import numpy as np

import project.esn.test as tesn
import project.music_gen.data_types as dt


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


def generate_header(clock_cycle, channel, offset, tempo):
    row_list = [["0", "0", "Header", "0", "1", clock_cycle],
                ["1", "0", "Start_track"],
                ["1", "0", "Channel_prefix", channel],
                ["1", "0", "Title_t", "STANDARD Drum"],
                ["1", "0", "Key_signature", "0", "major"],
                ["1", "0", "SMPTE_offset", offset, "0", "0", "0", "0"],
                ["1", "0", "Tempo", tempo]]
    return row_list


def row_to_csv(output_row, time_stamp):
    notes = np.where(output_row == 1.)
    print(numpy(notes))
    for i in range(1, len(notes) + 1):
        print(dt.Abs_note(i))


def output_to_csv(output_matrix, interval):
    time_stamp = 0
    data = []
    for x in output_matrix:
        row_to_csv(x, time_stamp)


def final_to_csv():
    csv_header = generate_header(480, 7, 33, 500000)
    with open('go.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_header)


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    (array, mse) = tesn.test_generated()
    (array, mse) = (array[5:6], mse)
    # pprint((array, mse))
    print(array)
    row_to_csv(array, 120)
