
import pandas as pd
import functools as ft
import math as m
import numpy as np

# General utils
# They are self explanatory one-liners
def spy (fun,payload):
    ret = fun ()
    print (ret,payload)
    return ret

def filter_reg(df,column,regexp):
    return df [df [column].str.match (regexp)]


def filter_top_frequency (df,column,n):
    return filter_reg(df,column,
                      '|'.join(df[column].value_counts().index [:n]))

def dict_top_frequency (df,column,n):
    top_frequency = df [column].value_counts ().index [:n]
    return {k:v for (k,v) in zip(top_frequency,range (len (top_frequency)))}

def translate_from_dict (df,column,t_dict):
    return  pd.DataFrame(df[col]  if col != column else
                         df [column].map (t_dict)
                          for col in df).transpose()

def get_min_clock (time_stamp):
    return ft.reduce (lambda x,y:m.gcd (x,y),time_stamp)

def agg_to_literal_arr (column):
    return "[%s]" % ",".join (map (lambda val: str (val),column.values))


def get_notes (midi):
    return midi [midi["event"].str.match ('.*Note.*')]

def get_meta_event (midi):
    return midi [midi ["event"].str.match('.*Note.*').apply (lambda x:not (x))]

def get_first_non_zero (arr):
    return ft.reduce (lambda x,y: x if x != 0 else y,arr)
