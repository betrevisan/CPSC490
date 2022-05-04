""" Predator-Prey Task for Attention Allocation (Classic Approach)
This implements the Predator-Prey task for attention allocation within classic
computing using a restricted boltzmann machine.
"""

from data import attention_data
from rbms import attention_rbm
from samplers import classic_sampler

TRAIN_SIZE = 100
TEST_SIZE = 30
WIDTH = HEIGHT = 100
EPOCHS = 20
LEARNING_RATE = 0.1

def main():
    # Generate a data for training and testing given the width and the height of the 
    # coordinate plane
    dataset = attention_data.AttentionData(TRAIN_SIZE, TEST_SIZE, WIDTH, HEIGHT)

    # Initialize the RBM sampler 
    rbm_sampler = classic_sampler.SamplerRBM()

    # Define visible and hidden dimensions of the RBM
    visible_dim = len(dataset.train_data_bin[0])
    hidden_dim = 15

    # Initialize the RBM given the sampler and the dimensions
    rbm = attention_rbm.AttentionRBM(rbm_sampler, visible_dim, hidden_dim)

    # Train the RBM for a given number of epochs with a given learning rate
    rbm.train(dataset.test_data_bin, EPOCHS, 0.1)

    # Test the RBM
    rbm.test(dataset.test_data_bin, dataset.test_data_bin_answers)

    # Run an example
    rbm.run_example(dataset.test_data_bin[0], dataset.test_data_bin_answers[0])

if __name__ == "__main__":
    main()
    