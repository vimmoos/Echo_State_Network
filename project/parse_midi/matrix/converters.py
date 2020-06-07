import pandas as pd
import project.parse_midi.matrix.utils as u
import ast

def note_to(reduce_fun,row_fun,row,lenght):
    return [reduce_fun(tup)
            for tup in
            [row_fun(row,index)
             if not row.empty else [0]
             for index in range (lenght)]]

def note_to_binary (row,lenght):
    return note_to(lambda x: 1 if 1 in x else 0,
                   lambda row,index: [1 if index == elem else 0
                                 for elem in ast.literal_eval (row.note)],
                   row,lenght)

def note_to_velocity (row,lenght):
    return note_to(u.get_first_non_zero,
                   lambda row,index:[velocity if index == note else 0
                                for (note,velocity) in
                                zip (ast.literal_eval (row.note),ast.literal_eval (row.velocity))],
                   row,lenght)


def convert_to_matrix (df,func,n_note):
    return [func (row,n_note) for _,row in df.iterrows ()]

def convert_to_full_matrix (df,func,n_note):
    clock = u.get_min_clock (df.index.map (lambda x:int (x)))
    return [func (df.loc[time],n_note)
            if time in df.index else
            func (pd.Series (),n_note)
            for time in [x*clock
                      for x in range (int (df.index [-1]/clock)+1)]]
