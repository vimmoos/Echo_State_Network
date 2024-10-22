import enum as e

import project.esn.utils as ut


class AutoName(e.Enum):
    def _generate_next_value_(name, start, count, last_values):
        count += 2
        return 2**count


class IndexName(e.Enum):
    def _generate_next_value_(name, start, count, last_values):
        return count


@e.unique
class Tempo(AutoName):
    """Tempo enum,define all the possible tempo

    """
    QUARTER = e.auto()
    EIGHTH = e.auto()
    SIXTEENTH = e.auto()
    THIRTY_SECOND = e.auto()


def max_tempos(t0: Tempo, t1: Tempo):
    return t0 if t0.value > t1.value else t1


@e.unique
class Quarters(e.Enum):
    """ Quarters enum, define all the possible quarter
    """
    ONE = e.auto()
    TWO = e.auto()
    THREE = e.auto()
    FOUR = e.auto()


@e.unique
class Abs_note(IndexName):
    """Abs_note enum, define all the possible note

    """
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
    """Note class, a note is represented as follows : It has a Tempo, a
    Quarters (in which the note is played), a value which is the
    intensity of the note, finally the abstract note

    """
    tempo: Tempo
    note: Abs_note
    quarter: Quarters
    val: float = 1.0


def check_instance(obj, typ, fun, string):
    if isinstance(obj, typ):
        return fun(obj)
    raise ValueError(string)
