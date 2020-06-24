import enum as e
import project.esn.utils as ut
import numpy as np
import types as t


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

    def __add__(self, other):
        if isinstance(other, gNote) and other.pattern_len == self.pattern_len:
            return note_zipper(self, other)
        raise ValueError(
            f"cannot add!! {other} {self.pattern_len} {other.pattern_len}")

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


def merge_tempos(arr0: np.array, arr1: np.array):
    if len(arr0) < len(arr1):
        tmp = arr0
        arr0 = arr1
        arr1 = tmp

    return arr0, np.vstack((arr1, [[0 for _ in range(len(arr0[0]))]
                                   for _ in range(len(arr0) - len(arr1))]))


def gNote_(note=None, pattern_len=None):
    def decorator(fun):
        ret_fun = lambda n, p: gNote(_generator=fun, note=n, pattern_len=p)
        if note == None and pattern_len == None:
            return ret_fun
        if note != None and pattern_len != None:
            return lambda: ret_fun(note, pattern_len)
        return lambda n: ret_fun(
            n, pattern_len) if note == None else lambda p: ret_fun(note, p)

    return decorator


def note_slice(gnote: gNote, n: int):
    @gNote_(note="slice", pattern_len=n)
    def gsliced(note: Note, pattern_len: int):
        gen = gnote()
        for _ in range(n * len(Quarters)):
            yield next(gen)

    return gsliced()


def note_replic(gnote: gNote, n: int):
    @gNote_(note="replic", pattern_len=gnote.pattern_len * n)
    def greplic(note: Note, pattern_len: int):
        for _ in range(n):
            for x in gnote():
                yield x

    return greplic()


def note_concat(gNote_0: gNote, gNote_1: gNote):
    @gNote_(note="concated",
            pattern_len=(gNote_0.pattern_len + gNote_1.pattern_len))
    def gconcat(note: Note, pattern_len: int):
        for x in gNote_0():
            yield x
        for x in gNote_1():
            yield x

    return gconcat()


def note_zipper(gNote_0: gNote, gNote_1: gNote):
    @gNote_(note="zipped", pattern_len=gNote_0.pattern_len)
    def gzipped(note: Note, pattern_len: int):
        for (x, y) in zip(gNote_0(), gNote_1()):
            if len(x) != len(y):
                (x, y) = merge_tempos(x, y)

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


def quartes_gen(tempo: Tempo):
    for x in range(tempo.value):
        yield (x, Quarters(int(x / (tempo.value / 4)) + 1))


def measure_generator(max_note: int, note: Note):
    while True:
        yield [[
            note.val if idx == note.note.value else 0
            for idx in range(max_note)
        ] for _ in range(note.tempo.value)]


def pippo():
    a = mona()
    b = gna()
    for i, j in zip(a, b):
        yield i + j + "last"


def mona():
    a = test()
    for i in a:
        yield i + "normal"


def gna():
    a = test()
    for i in a:
        newa = test()
        for i in newa:
            yield i + "nested"


def test():
    for i in (1, 2, 3):
        yield str(i) + "base"


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
bass2 = note_generator(Note(Tempo.EIGHTH, Abs_note.BASS_DRUM, Quarters.THREE),
                       t_len)

basss = bass0 + bass2

test = hit_quarter + snares + basss

test_g = hit_quarter[1] | test * 2 | ((hit_quarter + basss) * 2)[3]

test3 = test * 2

test4 = test[2]
