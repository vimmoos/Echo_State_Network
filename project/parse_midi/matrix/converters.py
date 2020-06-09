import pandas as pd
import project.parse_midi.matrix.utils as u
import ast

def note_to(reduce_fun,row_fun,row,lenght):
    """General converter from row to list. First loop over the lenght and
    if the row is not empty call the row_fun which must return a
    list. After that we loop over the just created list and apply the
    reduce_function for each elem (because we will have a structure
    like [[0],[a,b,b],...,[a,a,b]])

    """
    return [reduce_fun(tup)
            for tup in
            [row_fun(row,index)
             if not row.empty else [0]
             for index in range (lenght)]]

def note_to_binary (row,lenght):
    """Reduce a row to a list with 0s and 1s corresponding to the event
    note

    """
    return note_to(lambda x: 1 if 1 in x else 0,
                   lambda row,index: [1 if index == elem else 0
                                 for elem in ast.literal_eval (row.note)],
                   row,lenght)

def note_to_velocity (row,lenght):
    """Reduce a row to a list with 0s and velocities corresponding to the event
    note

    """
    return note_to(u.get_first_non_zero,
                   lambda row,index:[velocity if index == note else 0
                                for (note,velocity) in
                                zip (ast.literal_eval (row.note),ast.literal_eval (row.velocity))],
                   row,lenght)


def convert_to_matrix (df,func,n_note):
    """
    Simple convert to matrix function [UNUSED]
    """
    return [func (row,n_note) for _,row in df.iterrows ()]

def convert_to_full_matrix (df,func,n_note):
    """General converter from df to matrix
    First get the clock (which is the gcd of all the event, in this
    way we are sure that any event will happen at time x*clock) this
    is used to encode the temporal information in the lenght of the
    matrix. After that we loop over the list of all clocks in the
    "song"/df and we apply the func with the row if there is at least
    one row with that time otherwise we apply the func with an empty
    Series

    """
    clock = u.get_min_clock (df.index.map (lambda x:int (x)))
    return [func (df.loc[time],n_note)
            if time in df.index else
            func (pd.Series (),n_note)
            for time in [x*clock
                      for x in range (int (df.index [-1]/clock)+1)]]
