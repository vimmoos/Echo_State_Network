import pickle
from enum import Enum
from pprint import pprint
from random import randint

import numpy as np

import project.esn.core as core
import project.esn.transformer as transf
import project.PSO.config as c
import project.esn.core as core
import random as r
from pprint import pprint


def bind_enum_idx(idx: float, e: Enum) -> int:
    return int(idx) if int(idx) < len(list(e)) else int(idx) - 1


def map_params(**PSO_kwargs) -> dict:
    for k, v in PSO_kwargs.items():
        if k in ["t_squeeze", "squeeze_o"]:
            PSO_kwargs[k] = list(transf.Squeezers)[bind_enum_idx(
                v, transf.Squeezers)].value
        if k == "transformer":
            PSO_kwargs[k] = list(transf.Transformers)[bind_enum_idx(
                v, transf.Transformers)]
        if k == "reservoir":
            PSO_kwargs[k] = list(c.Res)[bind_enum_idx(v, c.Res)]
    return PSO_kwargs


def res_name(conf: dict) -> list:
    return [
        str(v) for k, v in conf.items()
        if k not in ["desired", "output", "input"]
    ]


def run_net_load_matrix(load_param: c.Res, **kwargs) -> dict:
    run_dict = core.Run(**{
        **c.esn_gen,
        **kwargs
    }).load((c.path + load_param.value), randint(0, 9)).__enter__()()
    with open(c.path_esn + "_".join(res_name(run_dict)), "wb") as f:
        pickle.dump(run_dict, f)
    return run_dict


def run_network(out_transf: transf.Transformer = transf.Transformers.sig_prob,
                **PSO_kwargs) -> tuple:
    PSO_kwargs = map_params(**PSO_kwargs)
    load_param = PSO_kwargs.pop("reservoir") if PSO_kwargs.get(
        "reservoir", None) else None
    run_dict = (run_net_load_matrix(load_param, **PSO_kwargs)
                if load_param else core.Run(**{
                    **c.esn_gen,
                    **PSO_kwargs
                }).__enter__()())
    pprint(f"dumped conf : {PSO_kwargs}")
    return ((raw_out := run_dict["output"]),
            out_transf.value(0.8,
                             transf._identity)(raw_out), run_dict["desired"])


check_default = lambda obj, default, call, *args, **kwargs: (
    obj if obj is not None else
    (default(*args, **kwargs)
     if hasattr(default, "__call__") and call else default))


def enum_rand_idx(e: Enum, velocity: bool) -> float:
    return np.random.uniform(0 if not velocity else -len(list(e)),
                             len(list(e)))


def check_transformer(dims: dict):
    if "transformer" in (keys := dims.keys()):
        if "t_param" not in keys:
            dims["t_param"] = (0.05, 1)
        if "t_squeeze" not in keys:
            dims["t_squeeze"] = transf.Squeezers


def distribute_dimensions(dims: dict, velocity: bool = False) -> float:
    check_transformer(dims)
    ret = []
    for k, v in dims.items():
        if k in ["t_squeeze", "squeeze_o"] and v.__name__ == "Squeezers":
            ret.append(enum_rand_idx(transf.Squeezers, velocity))
        elif k == "transformer" and v.__name__ == "Transformers":
            ret.append(enum_rand_idx(transf.Transformers, velocity))
        elif k == "reservoir" and v.__name__ == "Res":
            ret.append(enum_rand_idx(c.Res, velocity))
        elif isinstance(v, tuple):
            up, low = v
            ret.append(
                np.random.uniform(up, low) if not velocity else np.random.
                uniform(-np.abs(up - low), np.abs(up - low)))
    return np.array(ret)
