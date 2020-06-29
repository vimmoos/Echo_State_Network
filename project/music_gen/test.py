from project.music_gen.core import *
from project.music_gen.data_types import *

t_len = 3

hit_quarter = note_replicator(Tempo.QUARTER, Abs_note.HI_HAT_CLOSE,
                              list(Quarters))

snares = note_replicator(Tempo.QUARTER, Abs_note.SNARE,
                         [Quarters.TWO, Quarters.FOUR])

basses = note_replicator(Tempo.QUARTER, Abs_note.BASS_DRUM,
                         [Quarters.ONE, Quarters.THREE])

std_groove = hit_quarter + snares + basses

bass0 = note_generator(Note(Tempo.QUARTER, Abs_note.BASS_DRUM, Quarters.ONE),
                       t_len)
bass2 = note_generator(
    Note(Tempo.EIGHTH, Abs_note.BASS_DRUM, Quarters.THREE, 0.5), t_len)

basss = bass0 + bass2

test = hit_quarter + snares + basss

test_g = hit_quarter[1] | test * 2 | ((hit_quarter + basss) * 2)[3]

test3 = test * 2

test4 = test[2]

# def quartes_gen(tempo: Tempo):
#     for x in range(tempo.value):
#         yield (x, Quarters(int(x / (tempo.value / 4)) + 1))

# def measure_generator(max_note: int, note: Note):
#     while True:
#         yield [[
#             note.val if idx == note.note.value else 0
#             for idx in range(max_note)
#         ] for _ in range(note.tempo.value)]

# def pippo():
#     a = mona()
#     b = gna()
#     for i, j in zip(a, b):
#         yield i + j + "last"

# def mona():
#     a = test()
#     for i in a:
#         yield i + "normal"

# def gna():
#     a = test()
#     for i in a:
#         newa = test()
#         for i in newa:
#             yield i + "nested"

# def test():
#     for i in (1, 2, 3):
#         yield str(i) + "base"

len_ = 1

ride = note_generator(Note(Tempo.QUARTER, Abs_note.FLOOR_TOM, Quarters.FOUR),
                      len_)

bass = note_generator(Note(Tempo.QUARTER, Abs_note.BASS_DRUM, Quarters.ONE),
                      len_)

charl = note_replicator(Tempo.EIGHTH, Abs_note.HI_HAT_CLOSE,
                        [Quarters.ONE, Quarters.THREE])

ttest = bass + charl + ride
pprint(list(~ttest))
