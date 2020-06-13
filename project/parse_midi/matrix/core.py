
import pandas as pd
import project.parse_midi.matrix.utils as u
import project.parse_midi.matrix.converters as conv


def assemble_proc_dict (proc_dict):
    """Takes a proc_dict which encode the information needed to perform
    the parsing and it returns another dict containing a prefiled flow
    which only need to be patch with the correct input.

    """
    return {"midi_csv": pd.read_csv (proc_dict ["midi"],names=proc_dict ["column"]),
            "original_body": (lambda midi: u.get_notes (midi)),
            "header": (lambda midi: u.get_meta_event (midi)),

            "filtered_body": (lambda body:u.filter_reg (
                u.filter_top_frequency(
                    body,"event",proc_dict ["n_note"])
                ,"event",proc_dict ["filter_event"])),

            "trans_dict": (lambda body:u.dict_top_frequency(body,"note",proc_dict["n_note"])),

            "new_body" :(lambda body,trans_dict: u.translate_from_dict (
                body.drop (proc_dict ["body_drop"],axis=1)
                ,"note",trans_dict)
                         .groupby ("time")
                         .agg(proc_dict ["agg_func"])),

            "matrixs" : (lambda body :[conv.convert_to_full_matrix ( body,fun,proc_dict ["n_note"])
                         for fun in proc_dict ["convert_funcs"]])}


def eval_aproc_dict (proc_dict):
    """Takes an already assembled proc_dict and it patches the entries
    and return the result of the parsing which consists in a dict:
    {:original_body  body
    :header  header
    :new_body new_body
    :trans_dict trans_dict
    :matrixs matrixs}

    """
    body = proc_dict ["original_body"] (proc_dict ["midi_csv"])
    filt_body = proc_dict ["filtered_body"] (body)
    trans_dict = proc_dict ["trans_dict"] (filt_body)
    new_body = proc_dict ["new_body"] (filt_body,trans_dict)
    return {"original_body": body,
            "header": proc_dict ["header"] (proc_dict ["midi_csv"]),
            "new_body": new_body,
            "trans_dict": trans_dict,
            "matrixs" : proc_dict ["matrixs"] (new_body)}

def exec_proc_dict(proc_dict):
    """Takes a proc_dict and return the result of the midi parsing

    """
    return eval_aproc_dict(assemble_proc_dict(proc_dict))
