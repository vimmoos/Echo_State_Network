from matplotlib.pyplot import *
from scipy import linalg
from abs_ESN import *


class Data (object):
    train_len = 0
    test_len = 0
    init_len = 0
    vector = None
    target = None

    def __init__ (self,train_len,test_len,init_len,
                       vector):
        super ().__init__ ()
        self.train_len = train_len
        self.test_len = test_len
        self.init_len = init_len
        self.vector = vector
        self.target = vector [None,init_len+1:train_len+1]


def normalize_spectral_radius (matrix):
    rho_W = max (abs (linalg.eig (matrix) [0]))
    matrix *= 1.25 / rho_W


class Echo_state_network(abs_ESN):

    def __init__(self,input_size,output_size,
                      reservoir_size,
                      leaking_rate):
        super().__init__(input_size,output_size,
                         reservoir_size,leaking_rate)
        self.generators = {"random" : lambda : self.generate_random_Ws(42)}
        self.trainers = {"ridge" : self.train_output_ridge}

    # generates the input weight matrix and the reservoir matrix
    def generate_random_Ws (self,seed):
        np.random.seed (seed)
        self.W_in = np.random.rand (self.reservoir_size,1+ self.input_size)-0.5
        self.W_res = np.random.rand (self.reservoir_size,self.reservoir_size)- 0.5
        normalize_spectral_radius (self.W_res)

    # train the output layer using the ridge regression to calculate
    # the output weight matrix
    def train_output_ridge (self):
        reg = 1e-8 # regularization coefficient
        self.W_out = np.dot (np.dot (self.data.target,self.states.T),
                          linalg.inv (np.dot (self.states,self.states.T) +
                                      reg * np.eye (1+self.input_size+self.reservoir_size)))


    # run for the training len and collect the resulting states only
    # afte t >= init_len
    def run_and_collect (self):
        for time in range (self.data.train_len):
            u = self.data.vector[time]
            update = np.tanh (np.dot (self.W_in,np.vstack ((1,u))) +
                           np.dot (self.W_res,self.activations))
            self.activations = ((1-self.leaking_rate) * self.activations +
                                self.leaking_rate * update)
            if time >= self.data.init_len: # done to remove potentially
                                          # weird starts of the data
                self.states [:,time-self.data.init_len] = np.vstack ((1,u,self.activations)) [:,0]

    # this function runs the network in a generative mode so it takes
    # as input the previous output
    def run_trained (self,error_len):
        u = self.data.vector [self.data.train_len]
        for time in range (self.data.test_len):
            update = np.tanh (np.dot (self.W_in,np.vstack ((1,u))) +
                           np.dot (self.W_res,self.activations))
            self.activations = ((1-self.leaking_rate)* self.activations +
                                self.leaking_rate * update)
            y = np.dot (self.W_out,np.vstack ((1,u,self.activations)))
            self.predicted [:,time] = y
            u = y
        return self.compute_MSE (error_len)
