import project.parse_midi.matrix.converters as conv
import project.parse_midi.matrix.core as core
import project.parse_midi.matrix.utils as u

default_columns = [
    "track", "time", "event", "channel", "note", "velocity", "arg1", "arg2"
]
default_drop = ["arg1", "arg2", "track", "channel", "event"]
default_filter = ".*Note_on_c.*"

# an example of a proc_dict
example_proc_dict = {
    "column": default_columns,
    "midi": "~/NN/resources/csv/34time13.csv",
    "n_note": 9,
    "body_drop": default_drop,
    "filter_event": default_filter,
    "agg_func": u.agg_to_literal_arr,
    "convert_funcs": [conv.note_to_binary, conv.note_to_velocity]
}
