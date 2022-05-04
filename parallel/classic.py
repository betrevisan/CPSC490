""" Complete Predator-Prey Task (Classical Approach)
This implements the complete Predator-Prey task within classical computing using a
restricted boltzmann machine.
"""

from data import data
from rbms import rbm
from samplers import classic_sampler

TRAIN_SIZE = 100
TEST_SIZE = 30
MAX_SPEED = 5
WIDTH = HEIGHT = 100
EPOCHS = 20
LEARNING_RATE = 0.1

def main():
    # Generate a data for training and testing given the width and the height of the 
    # coordinate plane and the maximum speed
    dataset = data.Data(TRAIN_SIZE, TEST_SIZE, MAX_SPEED, WIDTH, HEIGHT)

    # Initialize the RBM sampler, which will be used in the RBM
    rbm_sampler = classic_sampler.SamplerRBM()

    # Define visible and hidden dimensions of the RBM
    visible_dim = len(dataset.train_data_bin[0])
    hidden_dim = 15

    # Initialize the RBM given the sampler and the dimensions
    rbm_model = rbm.RBM(rbm_sampler, visible_dim, hidden_dim)

    # Train the RBM for a given number of epochs with a given learning rate
    rbm_model.train(dataset.test_data_bin, EPOCHS, 0.1)

    # Test the RBM
    rbm_model.test(dataset.test_data_bin, dataset.test_data_bin_answers)

    # Run an example
    rbm_model.run_example(dataset.test_data_bin[0], dataset.test_data_bin_answers[0])

    return


if __name__ == "__main__":
    main()