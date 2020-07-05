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

n0 = note_replicator(Tempo.EIGHTH, Abs_note.MID_TOM, [Quarters.TWO])
n1 = note_replicator(Tempo.QUARTER, Abs_note.CRASH, [Quarters.TWO])
n2 = note_replicator(Tempo.QUARTER, Abs_note.HI_HAT_CLOSE,
                     [Quarters.ONE, Quarters.THREE])

add = n0 + n1 + n2

n3 = note_replicator(Tempo.EIGHTH, Abs_note.SNARE, [Quarters.THREE])
n4 = note_replicator(Tempo.EIGHTH, Abs_note.HIG_TOM, [Quarters.FOUR]) * 2
n5 = note_replicator(Tempo.EIGHTH, Abs_note.HI_HAT_OPEN, [Quarters.ONE]) * 2 // (1,[0,1])

add1 = (n0 + n3 + n4) | (n5 + n1 + n2)


Q1_note1 = note_replicator(Tempo.SIXTEENTH, Abs_note.BASS_DRUM, [Quarters.ONE]) // (1,[1,0,0,1])
Q1_note2 = note_replicator(Tempo.SIXTEENTH, Abs_note.SNARE, [Quarters.ONE]) // (1, [0,1,1,0])

Q2_note1 = note_replicator(Tempo.SIXTEENTH, Abs_note.SNARE, [Quarters.TWO]) // (2,[1,0,0,0])
Q2_note2 = note_replicator(Tempo.SIXTEENTH, Abs_note.BASS_DRUM, [Quarters.TWO]) // (2,[0,0,1,0])
Q2_note3 = note_replicator(Tempo.SIXTEENTH, Abs_note.HI_HAT_OPEN, [Quarters.TWO]) // (2,[1,0,1,0])

Q3_note1 = note_replicator(Tempo.SIXTEENTH, Abs_note.HI_HAT_OPEN, [Quarters.THREE]) // (3,[1,0,1,0])
Q3_note2 = note_replicator(Tempo.SIXTEENTH, Abs_note.BASS_DRUM, [Quarters.THREE]) // (3, [0,1,1,0])

Q4_note1 = note_replicator(Tempo.SIXTEENTH, Abs_note.HI_HAT_CLOSE, [Quarters.FOUR]) // (4,[1,0,1,0])
Q4_note2 = note_replicator(Tempo.SIXTEENTH, Abs_note.SNARE, [Quarters.FOUR]) // (4,[1,0,0,1])

red_hot = (Q1_note1 + Q1_note2 + Q2_note1 + Q2_note2 + Q2_note3 + Q3_note1 + Q3_note2 + Q4_note1 + Q4_note2) * 20


reg_1 = note_replicator(Tempo.SIXTEENTH, Abs_note.BASS_DRUM , [Quarters.ONE]) // (1, [1,0,0,0])
reg_2 = note_replicator(Tempo.SIXTEENTH, Abs_note.HI_HAT_CLOSE , [Quarters.ONE]) // (1, [1,0,1,0])
reg_3 = note_replicator(Tempo.SIXTEENTH, Abs_note.BASS_DRUM , [Quarters.TWO]) // (2, [1,0,0,0])
reg_4 = note_replicator(Tempo.SIXTEENTH, Abs_note.SNARE , [Quarters.TWO]) // (2, [1,0,0,0])
reg_5 = note_replicator(Tempo.SIXTEENTH, Abs_note.HI_HAT_CLOSE, [Quarters.TWO]) // (2,[1,1,0,1])
reg_6 = note_replicator(Tempo.SIXTEENTH, Abs_note.HI_HAT_CLOSE, [Quarters.THREE]) // (3, [1,0,1,0])
reg_7 = note_replicator(Tempo.SIXTEENTH, Abs_note.BASS_DRUM , [Quarters.FOUR]) // (4, [1,0,0,0])
reg_8 = note_replicator(Tempo.SIXTEENTH, Abs_note.SNARE , [Quarters.FOUR]) // (4, [1,0,0,0])
reg_9 = note_replicator(Tempo.SIXTEENTH, Abs_note.HI_HAT_CLOSE, [Quarters.FOUR]) // (4,[1,1,0,1])
reg_10 = note_replicator(Tempo.SIXTEENTH, Abs_note.HI_HAT_CLOSE, [Quarters.THREE]) // (3, [1,0,1,1])
reg_11 = note_replicator(Tempo.SIXTEENTH, Abs_note.HI_HAT_CLOSE, [Quarters.FOUR]) // (4,[0,0,1,0])
reg_12 = note_replicator(Tempo.SIXTEENTH, Abs_note.SNARE, [Quarters.ONE]) // (1, [0,0,1,0])
reg_13 = note_replicator(Tempo.SIXTEENTH, Abs_note.SNARE, [Quarters.THREE]) // (3, [1,0,0,1])
reg_14 = note_replicator(Tempo.SIXTEENTH, Abs_note.HI_HAT_CLOSE, [Quarters.FOUR]) // (4, [1,0,1,0])
reg_15 = note_replicator(Tempo.SIXTEENTH, Abs_note.SNARE, [Quarters.FOUR]) // (4, [0,0,1,0])


reg_m1 = reg_1 + reg_2 + reg_3 + reg_4 + reg_5 + reg_13 + reg_7 + reg_8 + reg_9
reg_m2 = reg_2 + reg_3 + reg_4 + reg_5 + reg_10 + reg_7 + reg_8 + reg_11
reg_m3 = reg_2 + reg_3 + reg_4 + reg_5 + reg_6 + reg_7 + reg_8 + reg_9
reg_m4 = reg_2 + reg_12 + reg_3 + reg_4 + reg_5 + reg_10 + reg_13 + reg_14 + reg_15

reggae = (reg_m1 | reg_m2 | reg_m3 | reg_m4) *10

ste_1 = note_replicator(Tempo.EIGHTH, Abs_note.BASS_DRUM, [Quarters.ONE])// (1,[1,0])
ste_2 = note_replicator(Tempo.EIGHTH, Abs_note.BASS_DRUM, [Quarters.TWO])// (2,[1,0])
ste_3 = note_replicator(Tempo.EIGHTH, Abs_note.BASS_DRUM, [Quarters.THREE])// (3,[1,0])
ste_4 = note_replicator(Tempo.EIGHTH, Abs_note.BASS_DRUM, [Quarters.FOUR])// (4,[1,0])
ste_5 = note_replicator(Tempo.EIGHTH, Abs_note.HI_HAT_CLOSE, [Quarters.ONE, Quarters.TWO, Quarters.THREE])
ste_6 = note_replicator(Tempo.EIGHTH, Abs_note.HI_HAT_CLOSE, [Quarters.FOUR])// (4, [1,0])
ste_7 = note_replicator(Tempo.EIGHTH, Abs_note.CRASH, [Quarters.FOUR])// (4, [0,1])
ste_8 = note_replicator(Tempo.EIGHTH, Abs_note.HIG_TOM, [Quarters.FOUR])// (4, [0,1])
ste_9 = note_replicator(Tempo.EIGHTH, Abs_note.HI_HAT_CLOSE, [Quarters.FOUR])
ste_10 = note_replicator(Tempo.EIGHTH, Abs_note.FLOOR_TOM, [Quarters.TWO]) // (2,[0,1])
ste_11 = note_replicator(Tempo.EIGHTH, Abs_note.BASS_DRUM, [Quarters.FOUR]) // (4,[0,1])
ste_12 = note_replicator(Tempo.EIGHTH, Abs_note.FLOOR_TOM, [Quarters.THREE]) // (3,[1,0])


ste_m1 = ste_1 + ste_2 + ste_3 + ste_4 + ste_5 + ste_6 + ste_7 + ste_8
ste_m2 = ste_1 + ste_2 + ste_3 + ste_4 + ste_5 + ste_9 + ste_10 + ste_11
ste_m3 = ste_1 + ste_2 + ste_3 + ste_4 + ste_5 + ste_9
ste_m4 = ste_1 + ste_2 + ste_3 + ste_4 + ste_5 + ste_6 + ste_7 + ste_8 + ste_10 + ste_12

steward = (ste_m1 | ste_m2 | ste_m3 | ste_m4) * 10

all = red_hot | reggae | steward

p(all.len())

test_patterns = [ttest, add, add1, red_hot, reggae, steward]


my_add = list((steward)())

new=np.array([[0,0,0,0,0,0,0,0,0]])
for a in my_add:
    new = np.concatenate((new, a), axis=0)
new = new[1:]
