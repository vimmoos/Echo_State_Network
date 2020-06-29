import functools as ft
import itertools as it

import numpy as np

import project.esn.utils as ut
from project.music_gen.data_types import *


@ut.mydataclass(init=True, repr=True)
class gNote():
    note: Note
    _generator: callable
    tempo: Tempo
    pattern_len: int = 1

    def __add__(self, other):
        return check_instance(other, gNote, lambda x: note_zipper(self, x),
                              f"cannot add!! {other}")

    def __or__(self, other):
        return check_instance(other, gNote, lambda x: note_concat(self, x),
                              f"cannot zip !! {other}")

    def __mul__(self, other):
        return check_instance(other, int, lambda x: note_replic(self, x),
                              f"cannot replic with {other}")

    def __getitem__(self, key):
        return check_instance(key, int, lambda x: note_slice(self, x),
                              f"cannot slice with {key}")

<<<<<<< HEAD
=======
    def __floordiv__(self,other):
        return check_instance(other,tuple,lambda x: note_pattern(self,other),
                              f"cannote apply the patter {other}")


    # def __sub__(self,other):
    #     return check_instance(other,,lambda x: )
>>>>>>> c8c756563bd2a47d2db249f0a2c657cde1cbcd4f
    def len(self):
        return int((self.pattern_len * 4) * (self.tempo.value / len(Quarters)))

    def __invert__(self):
        return note_sampler(self)

    def __call__(self):
        return self._generator(self.note, self.pattern_len)


gNote_ = lambda note=None, pattern_len=None, tempo=None: lambda fun: (
    lambda n, p: gNote(_generator=fun, note=n, pattern_len=p, tempo=n.tempo)
) if note is None else lambda: gNote(
    _generator=fun, note=note, pattern_len=pattern_len, tempo=tempo)


def note_sampler(note: gNote):
    for x in note():
        for el in x:
            yield el


def note_replicator(tempo: Tempo,
                    note: Abs_note,
                    quarters: list,
                    pattern_len=1):
    return note_zipper.reduce(
        [note_generator(Note(tempo, note, x), pattern_len) for x in quarters])


''' implementation of gNote operators
'''




def merge_tempos_f(tempo: Tempo):
    @np.vectorize
    def merge_tempos(genNote: gNote):
        def inner():
            for x in genNote():
                if genNote.tempo.value < tempo.value:
                    yield np.vstack(
                        (x, [[0 for _ in range(len(Abs_note))] for _ in range(
                            int((tempo.value - genNote.tempo.value) /
                                len(Quarters)))]))
                else:
                    yield x

        return inner()

    return merge_tempos


def note_reduce(f: callable, notes: list):
    return ft.reduce(f, notes)


def reducer(fun):
    setattr(fun, "reduce", lambda li: note_reduce(fun, li))
    return fun


def note_map(f: callable, notes: list):
    return map(f, notes)


def mapper(fun):
    setattr(fun, "map", lambda li: note_map(fun, li))
    return fun


def note_slice(gnote: gNote, n: int):
    @gNote_(note="slice", pattern_len=n, tempo=gnote.tempo)
    def gsliced(note: Note, pattern_len: int):
        return it.islice(gnote(), n * len(Quarters), None)

    return gsliced()


def note_replic(gnote: gNote, n: int):
    @gNote_(note="replic",
            pattern_len=gnote.pattern_len * n,
            tempo=gnote.tempo)
    def greplic(note: Note, pattern_len: int):
        for _ in range(n):
            for x in gnote():
                yield x

    return greplic()


@gNote_()
def note_generator(note: Note, pattern_len: int):
    for _ in range(pattern_len):
        for x in Quarters:
            yield np.array([
                np.array([
                    note.val
                    if idx == note.note.value and x == note.quarter else 0
                    for idx in range(len(Abs_note))
                ]) for t in range(int(note.tempo.value / len(Quarters)))
            ])


def note_pattern(gnote:gNote,qua_pat:tuple):
    quarter,pattern = qua_pat
    if not isinstance(quarter,Quarters):
        quarter = Quarters(quarter)
    c_pattern = np.array([np.ones(len(Abs_note)) if x != 0 else np.zeros(len(Abs_note)) for x in pattern])
    @gNote_(note="pattern",
            pattern_len= gnote.pattern_len,
            tempo=gnote.tempo)
    def gpatter(note:Note,pattern_len:int):
        for i,x in enumerate(gnote()):
            yield x if (i%4) == quarter.value else x * c_pattern

    return gpatter()





@reducer
@mapper
def note_concat(gNote_0: gNote, gNote_1: gNote):
    max_tempo = max_tempos(gNote_0.tempo, gNote_1.tempo)
    merge_t = merge_tempos_f(max_tempo)

    @gNote_(note="concated",
            pattern_len=(gNote_0.pattern_len + gNote_1.pattern_len),
            tempo=max_tempo)
    def gconcat(note: Note, pattern_len: int):
        return it.chain(*merge_t([gNote_0, gNote_1]))

    return gconcat()


@reducer
@mapper
def note_zipper(gNote_0: gNote, gNote_1: gNote):
    max_tempo = max_tempos(gNote_0.tempo, gNote_1.tempo)
    merge_t = merge_tempos_f(max_tempo)

    @gNote_(note="zipped", pattern_len=gNote_0.pattern_len, tempo=max_tempo)
    def gzipped(note: Note, pattern_len: int):
        for (x, y) in it.zip_longest(
                *merge_t([gNote_0, gNote_1]),
                fillvalue= np.zeros((len(Abs_note),int(max_tempo.value/len(Quarters))))):
            yield x + y

    return gzipped()
