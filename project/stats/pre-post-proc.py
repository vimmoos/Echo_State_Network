import pickle as p
from os import listdir
from os.path import isfile, join
import project.esn.transformer as t
from collections import ChainMap

path = "/home/vimmoos/NN/resources/esn/"

experiment = (f for f in listdir(path) if isfile(join(path, f)))

# data = (pic.load(open(f).__enter__()) for f in experiment)

data = [
    p.load(open(path + x, "rb")) for x in experiment
]

process_data = [[{
    **{
        trans.name: [{
            **{
                "param": (val := ((param * 2) / 10) + 0.2)
            },
            **{
                "output": trans.value(val, t._identity)(y["output"])
            }
        } for param in range(5)]
        for trans in list(t.Transformers)
    },
    **{k: v
       for k, v in y.items() if k != "output"}
} for y in x] for x in data]

compute_metrics = {}

import matplotlib.pyplot as pl
import scipy.signal as s
import scipy.fft as f
pl.figure(1)
pl.plot(process_data[0][0]["sig_prob"][3]["output"][:200, 2])

pl.figure(0)
pl.plot(data[0][0]["desired"][:200, 2])
pl.show()

pl.plot(
    s.correlate(process_data[0][0]["sig_prob"][3]["output"][:5000],
                process_data[0][0]["desired"][:5000]))

pl.figure(1)
pl.plot(
    s.correlate(process_data[0][0]["desired"], process_data[0][0]["desired"]))
pl.show()

a = f.fftn(process_data[0][0]["pow_prob"][0]["output"][:500, 0])
c = f.fftn(process_data[0][0]["sig_prob"][0]["output"][:500, 0])
d = f.fftn(process_data[0][0]["threshold"][0]["output"][:500, 0])
e = f.fftn(process_data[0][0]["identity"][0]["output"][:500, 0])
b = f.fftn(process_data[0][0]["desired"][:500, 0])

pl.figure(0)
pl.plot(f.fftn(process_data[0][0]["desired"][:500]))
pl.figure(1)
pl.plot(f.fftn(process_data[0][0]["sig_prob"][0]["output"][:500]))
pl.figure(2)
pl.plot(f.fftn(process_data[0][0]["threshold"][0]["output"][:500]))
pl.show()

pl.figure(0)
pl.plot(np.log(np.abs(scipy.fft.fftshift(b))**2))
pl.figure(1)
pl.plot(np.log(np.abs(scipy.fft.fftshift(a))**2))
pl.figure(2)
pl.plot(np.log(np.abs(scipy.fft.fftshift(c))**2))
pl.figure(3)
pl.plot(np.log(np.abs(scipy.fft.fftshift(d))**2))
pl.figure(4)
pl.plot(np.log(np.abs(scipy.fft.fftshift(e))**2))

pl.show()
