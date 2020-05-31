import numpy as np

class abs_ESN (object):
    input_size = None
    output_size = None
    reservoir_size = None
    leaking_rate = 1
    generators = {}
    trainers = {}
    W_in = None
    W_res = None
    W_out = None
    activations = None
    states = None
    predicted = None
    data = None

    def __init__(self,input_size,output_size,
                      reservoir_size,
                      leaking_rate = 1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.reservoir_size = reservoir_size
        self.leaking_rate = leaking_rate
        self.activations = np.zeros ((reservoir_size,1))

    def set_up (self,generator,trainer):
        self.generate_Ws = self.generators.get(generator, "NOT FOUND")
        self.train_output = self.trainers.get(trainer,"NOT FOUND")

    # this function must generate the W_in and the W_res matrix
    def generate_Ws():
        pass

    # this function should just calculate the W_out matrix NOTE: this
    # is not of online learning !!
    def train_output (self):
        pass

    # this function should just run and collect the resulting states
    def run_and_collect (self):
        pass

    # perform the offline training
    def train (self):
        self.run_and_collect ()
        self.train_output ()

    # simply set the data and initialize the states and predicted matrixs/vectors
    def set_data (self,data):
        self.data = data
        self.states = np.zeros((1+self.input_size+self.reservoir_size,
                             data.train_len - data.init_len))
        self.predicted = np.zeros ((self.output_size,data.test_len))

    # compute the Mean-Square-Error given an error_len used to decide
    # how many timestep of the testing data we are considering
    def compute_MSE (self,error_len):
        train_len = self.data.train_len + 1
        return np.sum (np.square
                    (self.data.vector[train_len : train_len + error_len] -
                            self.predicted[0,0:error_len])) / error_len
