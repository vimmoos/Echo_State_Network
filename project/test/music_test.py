from pprint import pprint as p

from project.music_gen.core import *
from project.music_gen.data_types import *

# def defn(tempo:int,abs_n,quarter):
#     return note_generator(Note(Tempo(tempo),abs_n,Quarters(quarter)))


# hit_8t = note_replicator(Tempo.EIGHTH,Abs_note.HI_HAT_CLOSE,list(Quarters))
# bass_4t_1q = defn(4,Abs_note.BASS_DRUM,1)
# snare_4t_1q = defn(4,Abs_note.SNARE,2)
# snare_4t_4q = defn(4,Abs_note.SNARE,4)
# bass_16t_2q = defn(16,Abs_note.BASS_DRUM,2) // (2,[0,0,0,1])
# bass_16t_3q =  defn(16,Abs_note.BASS_DRUM,3) // (2,[0,1,0,0])
# hito_8t_4q = defn(8,)

# first_A = hit_8t + bass_4t_1q + snare_4t_1q + snare_4t_4q + bass_16t_2q + bass_16t_3q


# second_A = first_A +



# patter_a =



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
    Note(Tempo.EIGHTH, Abs_note.BASS_DRUM, Quarters.THREE, 1),
    t_len) // (3, [0, 1])

basss = bass0 + bass2

classic = (hit_quarter * 3) + (snares * 3) + basss

test_g = hit_quarter[1] | classic * 2 | ((hit_quarter + basss) * 2)[3]

test = classic | std_groove * 2 | test_g

test3 = test * 2

test4 = test[2]

len_ = 1

ride = note_generator(Note(Tempo.QUARTER, Abs_note.RIDE, Quarters.FOUR), len_)

bass = note_generator(Note(Tempo.QUARTER, Abs_note.BASS_DRUM, Quarters.ONE),
                      len_)

charl = note_replicator(Tempo.EIGHTH, Abs_note.HI_HAT_CLOSE,
                        [Quarters.ONE, Quarters.THREE])

ttest = bass + charl + ride

n0 = note_replicator(Tempo.QUARTER, Abs_note.MID_TOM, [Quarters.TWO])
n1 = note_replicator(Tempo.QUARTER, Abs_note.CRASH, [Quarters.TWO])
n2 = note_replicator(Tempo.QUARTER, Abs_note.HI_HAT_CLOSE,
                     [Quarters.ONE, Quarters.THREE])

add = n0 + n1 + n2

n3 = note_replicator(Tempo.EIGHTH, Abs_note.SNARE, [Quarters.THREE])
n4 = note_replicator(Tempo.EIGHTH, Abs_note.HIG_TOM, [Quarters.FOUR]) * 2
n5 = note_replicator(Tempo.QUARTER, Abs_note.HI_HAT_OPEN, [Quarters.ONE]) * 2

add1 = (n0 + n3 + n4) | (n5 + n1 + n2)

test_patterns = [ttest, add, add1]

# pprint(list((add1)()))
