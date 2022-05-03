""" Complete Predator-Prey Task (Classical Approach)
This implements the complete Predator-Prey task within classical computing using a
restricted boltzmann machine.
"""

from data import data
from rbms import sampler
# from dbms import dbm

TRAIN_SIZE = 100
TEST_SIZE = 30
WIDTH = HEIGHT = 100
MAX_SPEED = 5
# EPOCHS = 20
# LEARNING_RATE = 0.1

def main():
    # Generate a data for training and testing given the width and the height of the 
    # coordinate plane and the maximum speed
    dataset = data.Data(TRAIN_SIZE, TEST_SIZE, MAX_SPEED, WIDTH, HEIGHT)
    print(dataset.train_data.shape)
    print(dataset.test_data.shape)
    print(dataset.train_data_bin.shape)
    print(dataset.test_data_bin.shape)
    print(dataset.test_data_bin_answers.shape)
    print(dataset.test_data_bin_answers[0])

    # Initialize the RBM sampler, which will be used in the DBM
    rbm_sampler = sampler.SamplerRBM()

#     # Define visible and hidden dimensions of the DBM
#     visible_dim = len(dataset.train_data_bin[0])
#     hidden_dim = 15

#     # Initialize the DBM given the sampler and the dimensions
#     dbm = movement_rbm.MovementRBM(rbm_sampler, visible_dim, hidden_dim)

#     # Train the RBM for a given number of epochs with a given learning rate
#     rbm.train(dataset.test_data_bin, EPOCHS, 0.1)

#     # Test the RBM
#     rbm.test(dataset.test_data_bin, dataset.test_data_bin_answers)

#     # Run an example
#     rbm.run_example(dataset.test_data_bin[0], dataset.test_data_bin_answers[0])

    return


if __name__ == "__main__":
    main()