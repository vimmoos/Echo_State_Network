# NN
This repository, is the code base for the Neural Network course.
The documentation for the code can be found in the html folder, you can visualized with a broswer , Therefore it's necessary to clone the git and then open the index.html file
which can be found in html/project/index.html
a simple snippet of code to run the network is : 
``` python
import project.test.music_test as tmusic
import numpy as np
import project.esn.core as c
def test_generated():
    train_len = test_len = 6200
    init_len = 3200
    music = (tgen.test_patterns[2] * 300)
    data = c.Data(np.array(list(~(tmusic.all * 10))), (tmusic.all * 10).tempo,
              init_len,train_len,test_len)
    with c.Run(
            **
        {
            "data": data,
            "reservoir": 5000,
            "in_out": 9,
            "leaking_rate": 0.3,
            "reg": 1e-8,
            "transformer": ta.Transformers.pow_prob,
            "t_param": 1,
            "density": 0.04
            "spectral_radius": 1.3
            "t_squeeze": np.tanh,
            "noise" : 0
        }) as gen:
        return gen()
```
