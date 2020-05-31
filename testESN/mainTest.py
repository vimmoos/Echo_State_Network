from EchoStateNetwork import *


if __name__ == "__main__":
    txt = np.loadtxt('MackeyGlass_t17.txt')
    test_data = Data(2000,2000,100,txt)

    network = Echo_state_network(1,1,100,0.3)

    network.set_up("random","ridge")

    network.generate_Ws()

    network.set_data(test_data)

    network.train()

    mse = network.run_trained(500)

    print('MSE = ' + str( mse ))

    # plot some signals
    figure(1).clear()
    plot(test_data.vector[test_data.train_len+1:test_data.train_len+test_data.test_len+1], 'g' )
    plot( network.predicted.T, 'b' )
    title('Target and generated signals $y(n)$ starting at $n=0$')
    legend(['Target signal', 'Free-running predicted signal'])

    figure(2).clear()
    plot( network.states[0:20,0:200].T )
    title('Some reservoir activations $\mathbf{x}(n)$')

    show()
