import enum as e
import project.esn.utils as ut
import numpy as np
import types as t
import itertools as it


class AutoName(e.Enum):
    def _generate_next_value_(name, start, count, last_values):
        count += 2
        return 2**count


@e.unique
class Tempo(AutoName):
    QUARTER = e.auto()
    EIGHTH = e.auto()
    SIXTEENTH = e.auto()
    THIRTY_SECOND = e.auto()


def max_tempos(t0: Tempo, t1: Tempo):
    return t0 if t0.value > t1.value else t1


@e.unique
class Quarters(e.Enum):
    ONE = e.auto()
    TWO = e.auto()
    THREE = e.auto()
    FOUR = e.auto()


@e.unique
class Abs_note(e.Enum):
    HI_HAT_CLOSE = e.auto()
    HI_HAT_OPEN = e.auto()
    BASS_DRUM = e.auto()
    CRASH = e.auto()
    SNARE = e.auto()
    HIG_TOM = e.auto()
    MID_TOM = e.auto()
    FLOOR_TOM = e.auto()
    RIDE = e.auto()


@ut.mydataclass(init=True, repr=True)
class Note():
    tempo: Tempo
    note: Abs_note
    quarter: Quarters
    val: float = 1.0


def check_instance(obj, typ, fun, string):
    if isinstance(obj, typ):
        return fun(obj)
    raise ValueError(string)


@ut.mydataclass(init=True, repr=True)
class gNote():
    note: Note
    _generator: callable
    pattern_len: int
    tempo: Tempo

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

    def __call__(self):
        return self._generator(self.note, self.pattern_len)


def gNote_(note=None, pattern_len=None, tempo=None):
    def decorator(fun):
        return (lambda n, p: gNote(
            _generator=fun, note=n, pattern_len=p, tempo=n.tempo
        )) if note is None else lambda: gNote(
            _generator=fun, note=note, pattern_len=pattern_len, tempo=tempo)

    return decorator


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


def note_concat(gNote_0: gNote, gNote_1: gNote):
    max_tempo = max_tempos(gNote_0.tempo, gNote_1.tempo)
    merge_t = merge_tempos_f(max_tempo)

    @gNote_(note="concated",
            pattern_len=(gNote_0.pattern_len + gNote_1.pattern_len),
            tempo=max_tempo)
    def gconcat(note: Note, pattern_len: int):
        return it.chain(*merge_t([gNote_0, gNote_1]))

    return gconcat()


def note_zipper(gNote_0: gNote, gNote_1: gNote):
    max_tempo = max_tempos(gNote_0.tempo, gNote_1.tempo)
    merge_t = merge_tempos_f(max_tempo)

    @gNote_(note="zipped", pattern_len=gNote_0.pattern_len, tempo=max_tempo)
    def gzipped(note: Note, pattern_len: int):
        for (x, y) in it.zip_longest(
                *merge_t([gNote_0, gNote_1]),
                fillvalue=np.array([
                    np.array([0 for _ in range(len(Abs_note))])
                    for _ in range(int(max_tempo.value / len(Quarters)))
                ])):
            yield x + y

    return gzipped()


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


def note_sampler(note: gNote):
    for x in note():
        for el in x:
            yield el


t_len = 3

hit0 = note_generator(Note(Tempo.QUARTER, Abs_note.HI_HAT_CLOSE, Quarters.ONE),
                      t_len)
hit1 = note_generator(Note(Tempo.QUARTER, Abs_note.HI_HAT_CLOSE, Quarters.TWO),
                      t_len)
hit2 = note_generator(
    Note(Tempo.QUARTER, Abs_note.HI_HAT_CLOSE, Quarters.THREE), t_len)
hit3 = note_generator(
    Note(Tempo.QUARTER, Abs_note.HI_HAT_CLOSE, Quarters.FOUR), t_len)

hit_quarter = hit0 + hit1 + hit2 + hit3

snare1 = note_generator(Note(Tempo.QUARTER, Abs_note.SNARE, Quarters.TWO),
                        t_len)
snare3 = note_generator(Note(Tempo.QUARTER, Abs_note.SNARE, Quarters.FOUR),
                        t_len)

snares = snare1 + snare3

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
